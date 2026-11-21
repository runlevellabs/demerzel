#!/usr/bin/env python3
"""The hermetic smoke runner: harness checks against a scripted model.

Each task is a fixture repository, a scripted model transcript, and an
oracle. For each task the runner:

1. copies the fixture repository to a fresh scratch directory;
2. starts `demerzel-mock` on the transcript and reads its base URL;
3. writes a user config file that points an `openai-compat` instance
   at the mock, and pins the route with `DEMERZEL_` variables; a
   transcript with a `judge` member also gets a judge instance on the
   mock;
4. runs `demerzel --auto -- <prompt>` in the scratch repository, with
   scratch HOME and XDG directories and a minimal environment;
5. closes the mock's stdin and reads its count of unused replies,
   refused requests, and, for a judged task, judge requests;
6. runs the oracle in the final state.

No model and no network. The mock serves the replies by request
order, so a prompt change does not break a task; a harness change that
alters the final state, leaves a reply unused, or makes a request that
the mock refuses, does.
`bench/README.md` documents the task layout, the three file formats,
the isolation, and `DEMERZEL_EVAL_BANK`.
"""

import argparse
import contextlib
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
DEFAULT_TASKS = EVAL_DIR / "tasks"
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "demerzel"
DEFAULT_MOCK = REPO_ROOT / "target" / "release" / "demerzel-mock"

# Names the private bank. Only `--bank` reads it, so the public set
# stays the default wherever the variable is set.
BANK_VARIABLE = "DEMERZEL_EVAL_BANK"
# Names the task directory, for the oracle only, so an oracle can read
# data outside `repo/`. The agent never gets it.
TASK_DIR_VARIABLE = "DEMERZEL_EVAL_TASK_DIR"

SCHEMA = 1
INDEX_FIELDS = frozenset({"schema", "tasks"})
ENTRY_FIELDS = frozenset({"name", "source_step", "summary"})
TASK_FIELDS = frozenset({"schema", "prompt", "oracle"})
TRANSCRIPT_FIELDS = frozenset({"schema", "replies"})
TRANSCRIPT_OPTIONAL = frozenset({"judge"})
CALL_FIELDS = frozenset({"id", "name", "input"})
JUDGE_FIELDS = frozenset({"model", "requests", "answers"})
JUDGE_OPTIONAL = frozenset({"sequence", "checks", "rules"})
RULE_FIELDS = frozenset({"event", "ask", "action"})
RULE_OPTIONAL = frozenset({"tool", "threshold", "reason"})
# The members of each wire answer type of the judge route.
ANSWER_FIELDS = {
    "noul": frozenset({"type", "noul"}),
    "choice": frozenset({"type", "choice", "probabilities", "confidence"}),
    "score": frozenset({"type", "score", "probabilities", "confidence", "legend"}),
}
SUMMARY_FIELDS = frozenset({"served", "unused", "refused"})
JUDGED_SUMMARY_FIELDS = SUMMARY_FIELDS | {"judged"}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# `synthetic`, or a milestone and a plan step: m3-7, m3-8a, m2.5-8f.
SOURCE_STEP = re.compile(r"synthetic|m[0-9]+(?:\.[0-9]+)?-[0-9]+[a-z]?")
# The mock's parser stops at 128 levels; stay well below it.
MAX_DEPTH = 100
MOCK_URL = re.compile(r"http://127\.0\.0\.1:[0-9]{1,5}")

AGENT_TIMEOUT = 30.0
ORACLE_TIMEOUT = 30.0
MOCK_START_TIMEOUT = 10.0
MOCK_SUMMARY_TIMEOUT = 5.0
STOP_GRACE = 3.0
TAIL_LINES = 20
TAIL_BYTES = 64 * 1024
TAIL_WIDTH = 300
MOCK_LINE_BYTES = 64 * 1024

# The only parent variables a run keeps. Every other variable stays
# out: each DEMERZEL_ variable of the parent, each provider key, and
# anything else that could change the prompt or the route.
KEPT_VARIABLES = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
INSTANCE = "smoke"
MODEL = "scripted"
# The judge instance of a judged task. Its key variable has no
# DEMERZEL_ prefix, so it cannot meet a variable the binary reads.
JUDGE_INSTANCE = "judge"
JUDGE_KEY_VARIABLE = "SMOKE_JUDGE_KEY"
JUDGE_KEY = "fake-smoke-judge-key"

EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_INTERRUPTED = 130
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# os.waitid can wait without a reap, so the group ID stays reserved
# while the stop sweeps the group. Python 3.13 adds it on macOS.
HAVE_WAITID = hasattr(os, "waitid") and hasattr(os, "WNOWAIT")


class InvalidTask(Exception):
    """A task set or a task that does not validate."""


class Stopped(BaseException):
    """SIGTERM or SIGHUP reached the runner. `code` is the exit status."""

    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


# --- Validation ---------------------------------------------------------


def unique_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"the key {key!r} occurs twice in one object")
        document[key] = value
    return document


def no_constant(name):
    raise ValueError(f"{name} is not a JSON number")


def check_values(where: str, value) -> None:
    """Refuse a value that the OS or the mock cannot take: a string with
    a NUL or that is not UTF-8, a number that is not finite, and nesting
    deeper than MAX_DEPTH."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)) and depth > MAX_DEPTH:
            raise InvalidTask(f"{where}: the nesting is deeper than {MAX_DEPTH} levels")
        if isinstance(item, dict):
            stack.extend((key, depth) for key in item.keys())
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise InvalidTask(f"{where}: a number is out of range")
        elif isinstance(item, str):
            if "\0" in item:
                raise InvalidTask(f"{where}: a string holds a NUL character")
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise InvalidTask(f"{where}: a string is not valid UTF-8") from None


def load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as file:
            document = json.load(file, object_pairs_hook=unique_keys, parse_constant=no_constant)
    except FileNotFoundError:
        raise InvalidTask(f"{path}: the file does not exist") from None
    except (OSError, ValueError) as error:
        raise InvalidTask(f"{path}: {error}") from None
    except RecursionError:
        raise InvalidTask(f"{path}: the nesting is deeper than {MAX_DEPTH} levels") from None
    check_values(str(path), document)
    return document


def check_fields(where: str, value, fields: frozenset, optional: frozenset = frozenset()) -> dict:
    """An object with exactly `fields`, any of `optional`, and `schema: 1`,
    when `schema` is one of them."""
    if not isinstance(value, dict):
        raise InvalidTask(f"{where}: expected a JSON object")
    unknown = sorted(set(value) - fields - optional)
    if unknown:
        raise InvalidTask(f"{where}: unknown field {', '.join(map(repr, unknown))}")
    missing = sorted(fields - set(value))
    if missing:
        raise InvalidTask(f"{where}: missing field {', '.join(map(repr, missing))}")
    if "schema" in fields and not (type(value["schema"]) is int and value["schema"] == SCHEMA):
        raise InvalidTask(f"{where}: schema {value['schema']!r} is not supported; expected {SCHEMA}")
    return value


def check_text(where: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidTask(f"{where}: expected a non-empty string")
    return value


def check_number(where: str, value) -> None:
    if type(value) not in (int, float):
        raise InvalidTask(f"{where}: expected a number")


def check_numbers(where: str, value) -> dict:
    """A non-empty object of numbers."""
    if not isinstance(value, dict) or not value:
        raise InvalidTask(f"{where}: expected a non-empty object of numbers")
    for name, number in value.items():
        check_number(f"{where}.{name}", number)
    return value


def load_index(root: Path) -> list[dict]:
    """The index entries, after the checks that cover the whole directory."""
    if not root.is_dir():
        raise InvalidTask(f"{root}: the task directory does not exist")
    path = root / "index.json"
    index = check_fields(str(path), load_json(path), INDEX_FIELDS)
    entries = index["tasks"]
    if not isinstance(entries, list) or not entries:
        raise InvalidTask(f"{path}: tasks: expected a non-empty list")
    names = []
    for position, entry in enumerate(entries):
        where = f"{path}: tasks[{position}]"
        check_fields(where, entry, ENTRY_FIELDS)
        for field in sorted(ENTRY_FIELDS):
            check_text(f"{where}.{field}", entry[field])
        if not NAME.fullmatch(entry["name"]):
            raise InvalidTask(f"{where}.name: {entry['name']!r} is not a plain directory name")
        if not SOURCE_STEP.fullmatch(entry["source_step"]):
            raise InvalidTask(
                f"{where}.source_step: {entry['source_step']!r} is not 'synthetic' "
                "or m<milestone>-<step>, such as m3-8a"
            )
        if entry["name"] in names:
            raise InvalidTask(f"{where}.name: {entry['name']!r} is listed twice")
        names.append(entry["name"])
    # A hidden directory, such as the `.git` of a bank repository, is
    # not a task.
    unlisted = sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and not child.name.startswith(".") and child.name not in names
    )
    if unlisted:
        raise InvalidTask(f"{path}: the index does not list {', '.join(unlisted)}")
    return entries


def check_task(task_dir: Path) -> tuple[dict, int, dict | None]:
    """The task file of one task, its reply count, and its judge member,
    after the checks of its directory."""
    if not task_dir.is_dir():
        raise InvalidTask(f"{task_dir}: the task directory does not exist")
    path = task_dir / "task.json"
    task = check_fields(str(path), load_json(path), TASK_FIELDS)
    check_text(f"{path}: prompt", task["prompt"])
    oracle = task["oracle"]
    if not isinstance(oracle, list) or not oracle:
        raise InvalidTask(f"{path}: oracle: expected a non-empty list")
    for position, part in enumerate(oracle):
        check_text(f"{path}: oracle[{position}]", part)
    replies, judge = check_transcript(task_dir / "transcript.json")
    if not (task_dir / "repo").is_dir():
        raise InvalidTask(f"{task_dir / 'repo'}: the fixture directory does not exist")
    return task, replies, judge


def check_transcript(path: Path) -> tuple[int, dict | None]:
    """Tool-call replies, then exactly one text reply, which is last.

    This shape is an early check: the run asks for each reply in turn
    and ends at the text. The proof that the run used every reply is
    the mock's count at the end of the run. Return the reply count and
    the judge member, or None without one.
    """
    transcript = check_fields(str(path), load_json(path), TRANSCRIPT_FIELDS, TRANSCRIPT_OPTIONAL)
    judge = transcript.get("judge")
    if "judge" in transcript:
        check_judge(f"{path}: judge", judge)
    replies = transcript["replies"]
    if not isinstance(replies, list) or not replies:
        raise InvalidTask(f"{path}: replies: expected a non-empty list")
    last = len(replies) - 1
    for position, reply in enumerate(replies):
        where = f"{path}: replies[{position}]"
        if not isinstance(reply, dict) or len(reply) != 1 or not set(reply) <= {"text", "tool_calls"}:
            raise InvalidTask(f"{where}: expected exactly one of 'text' or 'tool_calls'")
        if "text" in reply:
            if not isinstance(reply["text"], str):
                raise InvalidTask(f"{where}.text: expected a string")
            if position != last:
                raise InvalidTask(f"{where}: only the last reply can be text")
            continue
        if position == last:
            raise InvalidTask(f"{where}: the last reply must be text")
        calls = reply["tool_calls"]
        if not isinstance(calls, list) or not calls:
            raise InvalidTask(f"{where}.tool_calls: expected a non-empty list")
        ids = set()
        for index, call in enumerate(calls):
            call_where = f"{where}.tool_calls[{index}]"
            check_fields(call_where, call, CALL_FIELDS)
            check_text(f"{call_where}.id", call["id"])
            if call["id"] in ids:
                # The harness joins each tool result to its call by id.
                raise InvalidTask(f"{call_where}.id: {call['id']!r} occurs twice in one reply")
            ids.add(call["id"])
            check_text(f"{call_where}.name", call["name"])
            if not isinstance(call["input"], dict):
                raise InvalidTask(f"{call_where}.input: expected a JSON object")
    return len(replies), judge


def check_judge(where: str, judge) -> None:
    """The judge member: the reply model, the expected number of judge
    requests, one wire answer per question id, and optional per-request
    overrides, check overrides, and rules."""
    check_fields(where, judge, JUDGE_FIELDS, JUDGE_OPTIONAL)
    check_text(f"{where}.model", judge["model"])
    if type(judge["requests"]) is not int or judge["requests"] < 0:
        raise InvalidTask(f"{where}.requests: expected a non-negative integer")
    answers = judge["answers"]
    if not isinstance(answers, dict) or (not answers and not judge.get("sequence")):
        raise InvalidTask(f"{where}.answers: expected a non-empty object")
    for question, answer in answers.items():
        check_answer(f"{where}.answers.{question}", answer)
    if "sequence" in judge:
        sequence = judge["sequence"]
        if not isinstance(sequence, list):
            raise InvalidTask(f"{where}.sequence: expected a list of answer objects")
        for index, step in enumerate(sequence):
            step_where = f"{where}.sequence[{index}]"
            if not isinstance(step, dict):
                raise InvalidTask(f"{step_where}: expected a JSON object")
            for question, answer in step.items():
                check_answer(f"{step_where}.{question}", answer)
    if "checks" in judge:
        checks = judge["checks"]
        if not isinstance(checks, dict):
            raise InvalidTask(f"{where}.checks: expected a JSON object")
        for name, action in checks.items():
            check_text(f"{where}.checks.{name}", action)
    if "rules" in judge:
        rules = judge["rules"]
        if not isinstance(rules, list):
            raise InvalidTask(f"{where}.rules: expected a list of rule objects")
        for index, rule in enumerate(rules):
            rule_where = f"{where}.rules[{index}]"
            check_fields(rule_where, rule, RULE_FIELDS, RULE_OPTIONAL)
            check_text(f"{rule_where}.event", rule["event"])
            check_text(f"{rule_where}.ask", rule["ask"])
            check_text(f"{rule_where}.action", rule["action"])
            if "tool" in rule:
                check_text(f"{rule_where}.tool", rule["tool"])
            if "reason" in rule:
                check_text(f"{rule_where}.reason", rule["reason"])
            if "threshold" in rule:
                check_number(f"{rule_where}.threshold", rule["threshold"])
                if not (0 < rule["threshold"] < 1):
                    raise InvalidTask(f"{rule_where}.threshold: expected a number strictly between 0 and 1")


def check_answer(where: str, answer) -> None:
    """One wire answer of the judge route, by its `type`."""
    kind = answer.get("type") if isinstance(answer, dict) else None
    if kind not in ANSWER_FIELDS:
        raise InvalidTask(f"{where}: expected an object with a type of {', '.join(ANSWER_FIELDS)}")
    check_fields(where, answer, ANSWER_FIELDS[kind])
    if kind == "noul":
        check_number(f"{where}.noul", answer["noul"])
        return
    check_number(f"{where}.confidence", answer["confidence"])
    probabilities = check_numbers(f"{where}.probabilities", answer["probabilities"])
    if kind == "choice":
        check_text(f"{where}.choice", answer["choice"])
        return
    check_number(f"{where}.score", answer["score"])
    levels = {str(level) for level in range(len(probabilities))}
    legend = answer["legend"]
    if set(probabilities) != levels:
        raise InvalidTask(f"{where}.probabilities: expected the levels 0 to {len(levels) - 1} as keys")
    if not isinstance(legend, dict) or set(legend) != levels or not all(isinstance(text, str) for text in legend.values()):
        raise InvalidTask(f"{where}.legend: expected one string for each level of probabilities")


def validate(root: Path) -> tuple[list[dict], list[str]]:
    """Every index entry and every error. An empty error list is a valid set."""
    try:
        entries = load_index(root)
    except InvalidTask as error:
        return [], [str(error)]
    errors = []
    for entry in entries:
        try:
            check_task(root / entry["name"])
        except InvalidTask as error:
            errors.append(str(error))
    return entries, errors


# --- Processes ----------------------------------------------------------


@contextlib.contextmanager
def signals_blocked():
    """Hold SIGINT, SIGTERM, and SIGHUP until the block ends, so a second
    signal cannot abandon a child or a scratch directory half-way."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class Child:
    """One child process in its own session: its process ID is also its group ID.

    Python opens every pipe end as non-inheritable, and `close_fds`
    stays on, so no other child holds the parent's end of a pipe.
    """

    def __init__(self, argv, *, cwd: Path, env: dict, stdin, stdout, stderr):
        self.process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=stdin, stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
        )
        self.buffer = b""
        self.stopped = False

    def wait_exit(self, timeout: float) -> bool:
        """Wait up to `timeout` for the leader to exit. Where the
        platform allows, leave it unreaped for the group sweep."""
        if self.process.returncode is not None:
            return True
        if not HAVE_WAITID:
            try:
                self.process.wait(timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                return False
        deadline = time.monotonic() + timeout
        delay = 0.001
        while True:
            flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
            if os.waitid(os.P_PID, self.process.pid, flags) is not None:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.05)

    def signal_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except (ProcessLookupError, PermissionError):
            # The group is empty: nothing is left to signal.
            pass

    def stop(self, first: signal.Signals) -> int:
        """Stop the leader with `first`, then SIGKILL after a grace period.
        Then kill whatever is left in the group, and reap the leader."""
        if self.stopped:
            return self.process.returncode
        with signals_blocked():
            if not self.wait_exit(0):
                self.signal_group(first)
                if not self.wait_exit(STOP_GRACE):
                    self.signal_group(signal.SIGKILL)
            # A background child of the leader can outlive it.
            self.signal_group(signal.SIGKILL)
            self.process.wait()
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    pipe.close()
            self.stopped = True
        return self.process.returncode

    def read_line(self, timeout: float) -> str | None:
        """The next stdout line, or None at end of file or after `timeout`."""
        deadline = time.monotonic() + timeout
        descriptor = self.process.stdout.fileno()
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or len(self.buffer) > MOCK_LINE_BYTES:
                return None
            ready, _, _ = select.select([descriptor], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(descriptor, 4096)
            if not chunk:
                return None
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8", "replace")


def run_child(argv, cwd: Path, env: dict, out, err, timeout: float, first: signal.Signals) -> int | None:
    """Run one child to its end and sweep its group. The exit status, or
    None after a timeout. Raise OSError or ValueError when it cannot start."""
    child = None
    timed_out = False
    try:
        child = Child(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        timed_out = not child.wait_exit(timeout)
    finally:
        if child is not None:
            child.stop(first)
    return None if timed_out else child.process.returncode


def clip(line: str) -> str:
    return line if len(line) <= TAIL_WIDTH else line[:TAIL_WIDTH] + " [cut]"


def tail(path: Path, label: str) -> list[str]:
    """The last lines of a log file, bounded in lines, bytes, and width."""
    try:
        with open(path, "rb") as file:
            size = file.seek(0, os.SEEK_END)
            file.seek(max(0, size - TAIL_BYTES))
            data = file.read(TAIL_BYTES)
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").rstrip("\n").splitlines()
    if size > TAIL_BYTES and len(lines) > 1:
        lines = lines[1:]  # The first line can start mid-line.
    lines = lines[-TAIL_LINES:]
    return [f"{label}:"] + [f"  | {clip(line)}" for line in lines] if lines else []


# --- One task -----------------------------------------------------------


def run_env(scratch: Path) -> dict[str, str]:
    """The child environment: the kept variables and scratch homes only."""
    env = {name: os.environ[name] for name in KEPT_VARIABLES if name in os.environ}
    env.setdefault("PATH", os.defpath)
    for name, directory in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        path = scratch / directory
        path.mkdir()
        env[name] = str(path)
    return env


def write_config(config_home: Path, url: str, judge: dict | None = None) -> None:
    """The user config file. A judged task also gets a judge instance on
    the mock's judge route, with its key in JUDGE_KEY_VARIABLE."""
    directory = config_home / "demerzel"
    directory.mkdir()
    # A JSON string of ASCII text is also a TOML basic string.
    text = (
        f"provider = {json.dumps(INSTANCE)}\n"
        f"\n[providers.{INSTANCE}]\n"
        'kind = "openai-compat"\n'
        f"base_url = {json.dumps(url + '/v1')}\n"
        f"model = {json.dumps(MODEL)}\n"
    )
    if judge is not None:
        text += (
            f"\n[providers.{JUDGE_INSTANCE}]\n"
            'kind = "typesafe"\n'
            f"base_url = {json.dumps(url + '/judge')}\n"
            f"model = {toml_string(judge['model'])}\n"
            f"api_key_env = {json.dumps(JUDGE_KEY_VARIABLE)}\n"
            f"\n[judgments]\n"
            f"instance = {json.dumps(JUDGE_INSTANCE)}\n"
        )
        checks = judge.get("checks") or {}
        if checks:
            text += "\n[judgments.checks]\n"
            for name, action in sorted(checks.items()):
                text += f"{toml_string(name)} = {toml_string(action)}\n"
        for rule in judge.get("rules") or []:
            text += "\n[[judgments.rules]]\n"
            text += f"event = {toml_string(rule['event'])}\n"
            if "tool" in rule:
                text += f"tool = {toml_string(rule['tool'])}\n"
            text += f"ask = {toml_string(rule['ask'])}\n"
            if "threshold" in rule:
                text += f"threshold = {rule['threshold']}\n"
            text += f"action = {toml_string(rule['action'])}\n"
            if "reason" in rule:
                text += f"reason = {toml_string(rule['reason'])}\n"
    (directory / "config.toml").write_text(text, encoding="utf-8")


def toml_string(text: str) -> str:
    """A TOML basic string of any text: each control character, quote,
    and backslash as a `\\u` escape."""
    return '"' + "".join(
        f"\\u{ord(c):04x}" if c < " " or c in '"\\\x7f' else c for c in text
    ) + '"'


def agent_env(env: dict, url: str, judge: dict | None = None) -> dict:
    """Variables outrank the project file: its `provider`, `model`, and
    `base_url` cannot move the run off the mock. A judged task also gets
    the judge's fake key."""
    pinned = dict(env, DEMERZEL_PROVIDER=INSTANCE, DEMERZEL_MODEL=MODEL, DEMERZEL_BASE_URL=url + "/v1")
    if judge is not None:
        pinned[JUDGE_KEY_VARIABLE] = JUDGE_KEY
    return pinned


def check_summary(line: str | None, total: int, judge: dict | None = None) -> str | None:
    """Why the mock's count fails the task, or None when the run used
    every reply, the mock refused no request, and a judged task sent
    the judge requests its transcript expects."""
    if line is None:
        return "the mock printed no count line"
    try:
        summary = json.loads(line)
    except ValueError:
        summary = None
    fields = SUMMARY_FIELDS if judge is None else JUDGED_SUMMARY_FIELDS
    if (
        not isinstance(summary, dict)
        or set(summary) != fields
        or not all(type(summary[field]) is int and summary[field] >= 0 for field in fields)
    ):
        return f"the count line of the mock is malformed: {clip(line)!r}"
    if summary["unused"] or summary["refused"] or summary["served"] != total:
        return (
            f"the mock served {summary['served']} of {total} replies; "
            f"{summary['unused']} unused; {summary['refused']} refused"
        )
    if judge is not None and summary["judged"] != judge["requests"]:
        return f"the mock answered {summary['judged']} judge requests; the transcript expects {judge['requests']}"
    return None


def run_task(task_dir: Path, binary: Path, mock: Path) -> tuple[str | None, list[str]]:
    """Run one task. Return the failed stage and its detail, or (None, [])."""
    scratch = None
    try:
        try:
            # Held signals keep a directory from leaking between its
            # creation and the assignment.
            with signals_blocked():
                scratch = Path(tempfile.mkdtemp(prefix="demerzel-smoke-"))
        except OSError as error:
            return "setup", [f"the scratch directory was not made: {error}"]
        return run_in(scratch, task_dir, binary, mock)
    finally:
        if scratch is not None:
            with signals_blocked():
                shutil.rmtree(scratch, ignore_errors=True)


def run_in(scratch: Path, task_dir: Path, binary: Path, mock_path: Path) -> tuple[str | None, list[str]]:
    task, total, judge = check_task(task_dir)
    try:
        repo = scratch / "repo"
        shutil.copytree(task_dir / "repo", repo, symlinks=True)
        env = run_env(scratch)
        logs = scratch / "logs"
        logs.mkdir()
    except OSError as error:
        return "setup", [f"the scratch directory was not filled: {error}"]
    mock_err_path = logs / "mock.stderr"
    mock = None
    try:
        try:
            with open(mock_err_path, "wb") as mock_err:
                # The mock serves until its stdin reaches end of file, so
                # it also ends when the runner dies.
                mock = Child(
                    [str(mock_path), str(task_dir / "transcript.json")], cwd=scratch, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=mock_err,
                )
        except (OSError, ValueError) as error:
            return "mock start", [f"the mock did not start: {error}"]
        line = mock.read_line(MOCK_START_TIMEOUT)
        if line is None or not MOCK_URL.fullmatch(line):
            exited = mock.wait_exit(0)
            status = mock.stop(signal.SIGTERM)
            if line is not None:
                reason = f"the first line of the mock is not a loopback base URL: {clip(line)!r}"
            elif exited:
                reason = f"the mock exited with status {status} before it printed a line"
            else:
                reason = f"the mock printed no line in {MOCK_START_TIMEOUT:g} s"
            return "mock start", [reason] + tail(mock_err_path, "mock stderr")
        url = line
        try:
            write_config(Path(env["XDG_CONFIG_HOME"]), url, judge)
        except OSError as error:
            return "setup", [f"the config file was not written: {error}"]

        out_path, err_path = logs / "agent.stdout", logs / "agent.stderr"
        try:
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                status = run_child(
                    [str(binary), "--auto", "--", task["prompt"]], repo, agent_env(env, url, judge),
                    out, err, AGENT_TIMEOUT, signal.SIGINT,
                )
        except (OSError, ValueError) as error:
            return "agent exit", [f"the agent did not start: {error}"]

        # End of file on stdin asks the mock for its count, then ends it.
        try:
            mock.process.stdin.close()
        except OSError:
            pass
        count_error = check_summary(mock.read_line(MOCK_SUMMARY_TIMEOUT), total, judge)
        mock.wait_exit(STOP_GRACE)
        mock.stop(signal.SIGTERM)
        logs_tail = (
            ([count_error] if count_error else [])
            + tail(err_path, "agent stderr")
            + tail(out_path, "agent stdout")
            + tail(mock_err_path, "mock stderr")
        )
        if status is None:
            return "timeout", [f"the agent ran past {AGENT_TIMEOUT:g} s"] + logs_tail
        if status != 0:
            return "agent exit", [f"the agent exited with status {status}"] + logs_tail
        if count_error:
            return "mock count", logs_tail
    finally:
        if mock is not None:
            mock.stop(signal.SIGTERM)

    oracle_path = logs / "oracle.output"
    oracle_env = dict(env, **{TASK_DIR_VARIABLE: str(task_dir)})
    try:
        with open(oracle_path, "wb") as output:
            status = run_child(
                task["oracle"], repo, oracle_env, output, subprocess.STDOUT, ORACLE_TIMEOUT,
                signal.SIGTERM,
            )
    except (OSError, ValueError) as error:
        return "oracle", [f"the oracle did not start: {error}"]
    if status is None:
        return "oracle timeout", [f"the oracle ran past {ORACLE_TIMEOUT:g} s"] + tail(oracle_path, "oracle output")
    if status != 0:
        return "oracle", [f"the oracle exited with status {status}"] + tail(oracle_path, "oracle output")
    return None, []


# --- Command line -------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run the hermetic smoke tasks. Each task runs the one-shot CLI against a "
            "scripted model and checks the final state of its fixture repository."
        ),
    )
    parser.add_argument(
        "--binary", type=Path, default=DEFAULT_BINARY, metavar="PATH",
        help="Use this demerzel binary. The default is target/release/demerzel.",
    )
    parser.add_argument(
        "--mock", type=Path, default=DEFAULT_MOCK, metavar="PATH",
        help="Use this mock server binary. The default is target/release/demerzel-mock.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--tasks", type=Path, metavar="DIR",
        help="Use the tasks in this directory. The default is the public smoke set.",
    )
    source.add_argument(
        "--bank", action="store_true",
        help=f"Use the tasks in the directory that {BANK_VARIABLE} names.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Validate the index and every task. Do not run a task.",
    )
    parser.add_argument("names", nargs="*", metavar="NAME", help="Run only the tasks with these names.")
    return parser.parse_args(argv)


def task_root(args) -> Path | None:
    if args.tasks is not None:
        return args.tasks
    if args.bank:
        bank = os.environ.get(BANK_VARIABLE, "")
        return Path(bank) if bank else None
    return DEFAULT_TASKS


def plural(count: int) -> str:
    return f"{count} task{'' if count == 1 else 's'}"


def main(argv=None) -> int:
    args = parse_args(argv)
    root = task_root(args)
    if root is None:
        print(f"smoke: --bank needs {BANK_VARIABLE} to name the bank directory", file=sys.stderr)
        return EXIT_INVALID
    entries, errors = validate(root)
    known = [entry["name"] for entry in entries]
    errors += [f"{root}: the index does not list {name}" for name in args.names if name not in known]
    if errors:
        for error in errors:
            print(f"smoke: {error}", file=sys.stderr)
        return EXIT_INVALID
    selected = [entry for entry in entries if not args.names or entry["name"] in args.names]
    if args.check:
        print(f"{plural(len(entries))} in {root}: valid")
        return 0
    for label, path in (("binary", args.binary), ("mock", args.mock)):
        if not (path.is_file() and os.access(path, os.X_OK)):
            print(
                f"smoke: the {label} {path} is not an executable file; "
                "build it with cargo build --release --locked",
                file=sys.stderr,
            )
            return EXIT_INVALID
    binary, mock = args.binary.resolve(), args.mock.resolve()

    width = max(len(entry["name"]) for entry in selected)
    started = time.monotonic()
    failed = 0
    try:
        for entry in selected:
            task_started = time.monotonic()
            stage, detail = run_task((root / entry["name"]).resolve(), binary, mock)
            seconds = time.monotonic() - task_started
            if stage is None:
                print(f"PASS  {entry['name']:<{width}}  {seconds:6.2f} s", flush=True)
                continue
            failed += 1
            print(f"FAIL  {entry['name']:<{width}}  {seconds:6.2f} s  stage: {stage}", flush=True)
            for line in detail:
                print(f"      {line}", flush=True)
    except KeyboardInterrupt:
        print("smoke: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Stopped as stop:
        print(f"smoke: stopped by a signal; exit status {stop.code}", file=sys.stderr)
        return stop.code
    wall = time.monotonic() - started
    print(
        f"{plural(len(selected))}: {len(selected) - failed} passed, {failed} failed, "
        f"{wall:.2f} s wall time",
        flush=True,
    )
    return EXIT_FAILED if failed else 0


def stopped(signum, frame):
    raise Stopped(128 + signum)


if __name__ == "__main__":
    # A cancelled CI job sends SIGTERM, and a closed terminal sends
    # SIGHUP: stop the children on both paths too.
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGHUP, stopped)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERRUPTED)
    except Stopped as stop:
        sys.exit(stop.code)
