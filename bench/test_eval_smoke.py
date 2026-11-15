"""The smoke runner must isolate each run and fail on any broken stage.

Stand-in executables replace the mock and the agent, so these tests
need no Rust build.
"""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from unittest.mock import patch


BENCH = Path(__file__).resolve().parent
SMOKE = BENCH / "eval" / "smoke.py"
spec = importlib.util.spec_from_file_location("smoke", SMOKE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

ORACLE = ["sh", "-c", 'diff -r . "$DEMERZEL_EVAL_TASK_DIR/expected"']
TRANSCRIPT = {
    "schema": 1,
    "replies": [
        {"tool_calls": [{"id": "c1", "name": "create", "input": {"path": "out.txt", "content": "done\n"}}]},
        {"text": "done"},
    ],
}
JUDGE = {
    "model": "judge-9.9.9",
    "requests": 2,
    "answers": {"outcome": {"type": "noul", "noul": 0.25}, "claim": {"type": "noul", "noul": 0.75}},
}
JUDGED_TRANSCRIPT = dict(TRANSCRIPT, judge=JUDGE)
MOCK_URL = "http://127.0.0.1:9"

# The mock stand-in follows the mock's protocol: the base URL, then
# serve until stdin ends, then the count line. `mode` breaks one step.
MOCK = """
import json, os, sys, time
with open({pid_file!r}, "w") as f:
    f.write(str(os.getpid()))
mode = {mode!r}
if mode == "exit":
    sys.exit(5)
if mode == "silent":
    time.sleep(60)
print("not a url" if mode == "garbage" else {url!r}, flush=True)
if mode == "garbage":
    time.sleep(60)
sys.stdin.read()
document = json.load(open(sys.argv[1]))
replies = len(document["replies"])
judge = document.get("judge")
counts = {{"served": replies, "unused": 0, "refused": 0}}
if (judge is not None and mode != "three-keys") or mode == "four-keys":
    counts["judged"] = (judge or {{}}).get("requests", 0) + (mode == "judged-off")
if mode == "malformed":
    print("{{not json", flush=True)
elif mode == "unused":
    print(json.dumps({{"served": replies - 1, "unused": 1, "refused": 0}}), flush=True)
elif mode == "refused":
    print(json.dumps({{"served": replies, "unused": 0, "refused": 1}}), flush=True)
elif mode == "two-keys":
    print(json.dumps({{"served": replies, "unused": 0}}), flush=True)
elif mode == "negative":
    print(json.dumps({{"served": replies, "unused": 0, "refused": -1}}), flush=True)
elif mode != "no-count":
    print(json.dumps(counts), flush=True)
"""

# The agent stand-in: record the process ID, the environment, the
# arguments, and the config file, check that the config points at the
# mock, then act.
AGENT = """
import json, os, signal, subprocess, sys, time
with open(os.path.join(os.environ["XDG_CONFIG_HOME"], "demerzel", "config.toml")) as f:
    config = f.read()
with open({record!r} + ".tmp", "w") as f:
    json.dump({{"pid": os.getpid(), "env": dict(os.environ), "argv": sys.argv, "config": config}}, f)
os.replace({record!r} + ".tmp", {record!r})
if {url!r} + "/v1" not in config:
    sys.exit(3)
{action}
"""

WRITE = 'open("out.txt", "w").write("done\\n")'
FAIL = "sys.exit(4)"
CHILD = """
child = subprocess.Popen(["sleep", "60"])
with open({record!r} + ".child", "w") as f:
    f.write(str(child.pid))
"""
HANG = CHILD + "time.sleep(60)\n"
BACKGROUND = CHILD + WRITE + "\n"
FLOOD = """
sys.stderr.write("x" * 2_000_000 + "\\n")
for number in range(50_000):
    sys.stderr.write(f"line {{number}}\\n")
sys.exit(4)
"""


def executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_task(root: Path, name: str, transcript=TRANSCRIPT, oracle=ORACLE) -> Path:
    task = root / name
    write_json(task / "task.json", {"schema": 1, "prompt": "write out.txt", "oracle": oracle})
    write_json(task / "transcript.json", transcript)
    (task / "repo").mkdir()
    (task / "repo" / "keep.txt").write_text("keep\n")
    (task / "expected").mkdir()
    (task / "expected" / "keep.txt").write_text("keep\n")
    (task / "expected" / "out.txt").write_text("done\n")
    return task


def make_set(root: Path, names=("one",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tasks = [{"name": name, "source_step": "synthetic", "summary": "a test task"} for name in names]
    write_json(root / "index.json", {"schema": 1, "tasks": tasks})
    for name in names:
        make_task(root, name)
    return root


def poll(path: Path, timeout: float = 10.0) -> str:
    """The content of a file that a slow child writes, once it is there."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(FileNotFoundError):
            text = path.read_text()
            if text:
                return text
        time.sleep(0.02)
    raise AssertionError(f"{path} did not appear")


def gone(pid: int, timeout: float = 3.0) -> bool:
    """True when no process has `pid`, after a reaper had time to act."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class SmokeTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.tasks = make_set(self.tmp / "tasks")
        self.mock_pid = self.tmp / "mock.pid"
        self.record = self.tmp / "agent.json"
        self.mock = self.make_mock("normal")
        self.agent = self.make_agent(WRITE)
        for name, value in (("AGENT_TIMEOUT", 10.0), ("MOCK_START_TIMEOUT", 5.0), ("STOP_GRACE", 1.0)):
            patcher = patch.object(smoke, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_mock(self, mode: str) -> Path:
        body = MOCK.format(pid_file=str(self.mock_pid), mode=mode, url=MOCK_URL)
        return executable(self.tmp / "mock", body)

    def make_agent(self, action: str) -> Path:
        action = action.format(record=str(self.record))
        return executable(self.tmp / "agent", AGENT.format(record=str(self.record), url=MOCK_URL, action=action))

    def argv(self, *args: str) -> list[str]:
        return ["--binary", str(self.agent), "--mock", str(self.mock), "--tasks", str(self.tasks), *args]

    def run_smoke(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = smoke.main(self.argv(*args))
        return status, output.getvalue()

    def check(self, root: Path) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return smoke.main(["--check", "--tasks", str(root)])

    def agent_record(self) -> dict:
        return json.loads(poll(self.record))

    def assert_mock_stopped(self):
        self.assertTrue(gone(int(poll(self.mock_pid))), "the mock is still running")


class RunTests(SmokeTestCase):
    def test_passing_task_passes(self):
        status, output = self.run_smoke()
        self.assertEqual(status, 0, output)
        self.assert_mock_stopped()

    def test_agent_gets_the_prompt_in_one_shot_auto_mode(self):
        self.run_smoke()
        self.assertEqual(self.agent_record()["argv"][1:], ["--auto", "--", "write out.txt"])

    def test_broken_oracle_fails_the_run(self):
        write_json(self.tasks / "one" / "task.json", {"schema": 1, "prompt": "p", "oracle": ["false"]})
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: oracle", output)

    def test_missing_oracle_program_fails_the_task(self):
        write_json(self.tasks / "one" / "task.json", {"schema": 1, "prompt": "p", "oracle": ["/absent/oracle"]})
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: oracle", output)

    def test_oracle_timeout_fails_the_task(self):
        write_json(self.tasks / "one" / "task.json", {"schema": 1, "prompt": "p", "oracle": ["sleep", "30"]})
        with patch.object(smoke, "ORACLE_TIMEOUT", 0.3):
            status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: oracle timeout", output)

    def test_final_state_change_fails_the_oracle(self):
        (self.tasks / "one" / "expected" / "out.txt").write_text("other\n")
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)

    def test_agent_exit_status_fails_the_run(self):
        self.agent = self.make_agent(FAIL)
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: agent exit", output)
        self.assert_mock_stopped()

    def test_mock_count_fails_the_task(self):
        # The agent exits 0 in each case; only the mock's count fails it.
        for mode in ("unused", "refused", "no-count", "malformed", "two-keys", "negative"):
            with self.subTest(mode=mode):
                self.mock = self.make_mock(mode)
                status, output = self.run_smoke()
                self.assertEqual(status, smoke.EXIT_FAILED, output)
                self.assertIn("stage: mock count", output)
                self.assert_mock_stopped()

    def test_judged_task_passes_when_the_judged_count_matches(self):
        write_json(self.tasks / "one" / "transcript.json", JUDGED_TRANSCRIPT)
        status, output = self.run_smoke()
        self.assertEqual(status, 0, output)

    def test_judged_count_fails_the_task(self):
        write_json(self.tasks / "one" / "transcript.json", JUDGED_TRANSCRIPT)
        for mode in ("judged-off", "three-keys", "refused"):
            with self.subTest(mode=mode):
                self.mock = self.make_mock(mode)
                status, output = self.run_smoke()
                self.assertEqual(status, smoke.EXIT_FAILED, output)
                self.assertIn("stage: mock count", output)

    def test_judged_count_without_a_judge_member_is_malformed(self):
        self.mock = self.make_mock("four-keys")
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: mock count", output)
        self.assertIn("malformed", output)

    def test_judge_model_is_any_text_in_the_config(self):
        path = self.tmp / "config-home"
        path.mkdir()
        model = 'a "quoted" \\ model\x7f\u00e9'
        smoke.write_config(path, MOCK_URL, dict(JUDGE, model=model))
        config = tomllib.loads((path / "demerzel" / "config.toml").read_text())
        self.assertEqual(config["providers"][smoke.JUDGE_INSTANCE]["model"], model)

    def test_mock_without_a_url_fails_without_a_hang(self):
        for mode in ("silent", "garbage", "exit"):
            with self.subTest(mode=mode):
                self.mock_pid.unlink(missing_ok=True)
                self.mock = self.make_mock(mode)
                started = time.monotonic()
                with patch.object(smoke, "MOCK_START_TIMEOUT", 0.5):
                    status, output = self.run_smoke()
                self.assertEqual(status, smoke.EXIT_FAILED, output)
                self.assertIn("stage: mock start", output)
                self.assertLess(time.monotonic() - started, 5)
                self.assertFalse(self.record.exists(), "the agent ran without a mock")
                self.assert_mock_stopped()

    def test_timeout_fails_and_leaves_no_child(self):
        self.agent = self.make_agent(HANG)
        with patch.object(smoke, "AGENT_TIMEOUT", 0.5):
            status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: timeout", output)
        self.assertTrue(gone(self.agent_record()["pid"]), "the agent is still running")
        self.assertTrue(gone(int(poll(Path(f"{self.record}.child")))), "the agent's child is still running")
        self.assert_mock_stopped()

    def test_background_child_dies_with_its_leader(self):
        self.agent = self.make_agent(BACKGROUND)
        status, output = self.run_smoke()
        self.assertEqual(status, 0, output)
        self.assertTrue(gone(int(poll(Path(f"{self.record}.child")))), "the background child survived")

    def test_one_long_line_still_shows(self):
        self.agent = self.make_agent('sys.stderr.write("y" * 200_000); sys.exit(4)')
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED)
        self.assertIn("y" * 100, output)
        self.assertLess(len(output), 5_000)

    def test_scratch_failure_fails_the_task(self):
        with patch.object(smoke.tempfile, "tempdir", str(self.tmp / "absent")):
            status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertIn("stage: setup", output)

    def test_failure_detail_is_bounded(self):
        self.agent = self.make_agent(FLOOD)
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED)
        self.assertLess(len(output), 20_000)

    def test_bad_binary_fails_its_task_and_the_run_continues(self):
        self.tasks = make_set(self.tmp / "two", names=("one", "two"))
        self.agent = self.tmp / "not-a-program"
        self.agent.write_text("this is not a program\n")
        self.agent.chmod(0o755)
        status, output = self.run_smoke()
        self.assertEqual(status, smoke.EXIT_FAILED, output)
        self.assertEqual(output.count("stage: agent exit"), 2, output)

    def test_child_environment_is_isolated(self):
        hostile = {
            "DEMERZEL_PROVIDER": "junk",
            "DEMERZEL_MODEL": "junk",
            "DEMERZEL_BASE_URL": "http://junk.invalid",
            "DEMERZEL_LOG": "trace",
            "DEMERZEL_EVAL_BANK": str(self.tmp),
            "ANTHROPIC_API_KEY": "fake-key-for-test",
            smoke.JUDGE_KEY_VARIABLE: "fake-parent-judge-key",
            "XDG_CONFIG_HOME": str(self.tmp / "user-config"),
        }
        for judged in (False, True):
            with self.subTest(judged=judged):
                transcript = JUDGED_TRANSCRIPT if judged else TRANSCRIPT
                write_json(self.tasks / "one" / "transcript.json", transcript)
                self.record.unlink(missing_ok=True)
                with patch.dict(os.environ, hostile):
                    status, output = self.run_smoke()
                self.assertEqual(status, 0, output)
                record = self.agent_record()
                env = record["env"]
                pinned = {name: value for name, value in env.items() if name.startswith("DEMERZEL_")}
                self.assertEqual(pinned, {
                    "DEMERZEL_PROVIDER": smoke.INSTANCE,
                    "DEMERZEL_MODEL": smoke.MODEL,
                    "DEMERZEL_BASE_URL": MOCK_URL + "/v1",
                })
                self.assertNotIn("ANTHROPIC_API_KEY", env)
                for name in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
                    self.assertNotIn(env[name], (os.environ.get(name), hostile["XDG_CONFIG_HOME"]), name)
                # The judge config and its key reach a judged task only.
                config = tomllib.loads(record["config"])
                if judged:
                    self.assertEqual(env[smoke.JUDGE_KEY_VARIABLE], smoke.JUDGE_KEY)
                    self.assertEqual(config["judgments"], {"instance": smoke.JUDGE_INSTANCE})
                    self.assertEqual(config["providers"][smoke.JUDGE_INSTANCE], {
                        "kind": "typesafe",
                        "base_url": MOCK_URL + "/judge",
                        "model": JUDGE["model"],
                        "api_key_env": smoke.JUDGE_KEY_VARIABLE,
                    })
                else:
                    self.assertNotIn(smoke.JUDGE_KEY_VARIABLE, env)
                    self.assertNotIn("judgments", config)
                    self.assertEqual(set(config["providers"]), {smoke.INSTANCE})

    def test_names_select_a_subset(self):
        self.tasks = make_set(self.tmp / "two", names=("one", "two"))
        write_json(self.tasks / "two" / "task.json", {"schema": 1, "prompt": "p", "oracle": ["false"]})
        self.assertEqual(self.run_smoke("one")[0], 0)
        self.assertEqual(self.run_smoke("absent")[0], smoke.EXIT_INVALID)

    def test_missing_binary_is_invalid(self):
        self.agent = self.tmp / "absent"
        self.assertEqual(self.run_smoke()[0], smoke.EXIT_INVALID)


class SignalTests(SmokeTestCase):
    def assert_signal_stops_the_run(self, sig: signal.Signals):
        self.agent = self.make_agent(HANG)
        scratch_root = self.tmp / f"scratch-{sig.name}"
        scratch_root.mkdir()
        runner = subprocess.Popen(
            [sys.executable, str(SMOKE), *self.argv()],
            env=dict(os.environ, TMPDIR=str(scratch_root)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        child = int(poll(Path(f"{self.record}.child")))
        runner.send_signal(sig)
        self.assertEqual(runner.wait(timeout=10), 128 + sig)
        for pid in (self.agent_record()["pid"], child, int(poll(self.mock_pid))):
            self.assertTrue(gone(pid), f"process {pid} survived")
        self.assertEqual(list(scratch_root.iterdir()), [])

    def test_sigterm_mid_task_leaves_no_child_and_no_scratch(self):
        self.assert_signal_stops_the_run(signal.SIGTERM)

    def test_sighup_mid_task_leaves_no_child_and_no_scratch(self):
        self.assert_signal_stops_the_run(signal.SIGHUP)


class ValidationTests(SmokeTestCase):
    def assert_invalid(self, root: Path):
        self.assertEqual(self.check(root), smoke.EXIT_INVALID)
        self.assertNotEqual(smoke.validate(root)[1], [])

    def test_valid_set_passes(self):
        self.assertEqual(self.check(self.tasks), 0)

    def test_unknown_task_field(self):
        task = json.loads((self.tasks / "one" / "task.json").read_text())
        write_json(self.tasks / "one" / "task.json", dict(task, timeout=5))
        self.assert_invalid(self.tasks)

    def test_unknown_index_entry_field(self):
        index = json.loads((self.tasks / "index.json").read_text())
        index["tasks"][0]["owner"] = "x"
        write_json(self.tasks / "index.json", index)
        self.assert_invalid(self.tasks)

    def test_wrong_schema(self):
        for path in ("index.json", "one/task.json", "one/transcript.json"):
            with self.subTest(path=path):
                root = make_set(self.tmp / path.replace("/", "-"))
                document = json.loads((root / path).read_text())
                for schema in (2, True, "1"):
                    write_json(root / path, dict(document, schema=schema))
                    self.assert_invalid(root)

    def test_json_the_mock_or_the_os_refuses(self):
        cases = {
            "duplicate key": '{"schema": 1, "schema": 1, "prompt": "p", "oracle": ["true"]}',
            "NaN": '{"schema": NaN, "prompt": "p", "oracle": ["true"]}',
            "Infinity": '{"schema": -Infinity, "prompt": "p", "oracle": ["true"]}',
            "NUL": '{"schema": 1, "prompt": "a\\u0000b", "oracle": ["true"]}',
            "lone surrogate": '{"schema": 1, "prompt": "\\ud800", "oracle": ["true"]}',
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                (self.tasks / "one" / "task.json").write_text(text)
                self.assert_invalid(self.tasks)
        write_json(self.tasks / "one" / "task.json", {"schema": 1, "prompt": "p", "oracle": ["true"]})
        transcript = json.dumps(TRANSCRIPT).replace('"done\\n"', '"\\ud800"')
        (self.tasks / "one" / "transcript.json").write_text(transcript)
        self.assert_invalid(self.tasks)

    def test_source_step_pattern(self):
        index = json.loads((self.tasks / "index.json").read_text())
        for step, valid in (
            ("synthetic", True), ("m3-7", True), ("m3-8a", True), ("m2.5-8f", True),
            ("7", False), ("m3", False), ("M3-7", False), ("m3-8ab", False), ("m3-", False),
        ):
            with self.subTest(step=step):
                index["tasks"][0]["source_step"] = step
                write_json(self.tasks / "index.json", index)
                self.assertEqual(self.check(self.tasks), 0 if valid else smoke.EXIT_INVALID)

    def write_input_value(self, text: str):
        """A transcript whose first call input holds `"x": <text>`."""
        transcript = json.dumps(TRANSCRIPT).replace('"input": {', '"input": {"x": %s, ' % text, 1)
        (self.tasks / "one" / "transcript.json").write_text(transcript)

    def test_number_out_of_range(self):
        self.write_input_value("1e400")
        self.assert_invalid(self.tasks)
        self.write_input_value("1e300")
        self.assertEqual(self.check(self.tasks), 0)

    def test_nesting_limit(self):
        # The input object sits at level 6 of the transcript.
        for extra, valid in ((smoke.MAX_DEPTH - 6, True), (smoke.MAX_DEPTH - 5, False), (2000, False)):
            with self.subTest(levels=6 + extra):
                self.write_input_value("[" * extra + "]" * extra)
                self.assertEqual(self.check(self.tasks), 0 if valid else smoke.EXIT_INVALID)

    def test_unlisted_directory(self):
        make_task(self.tasks, "stray")
        self.assert_invalid(self.tasks)

    def test_hidden_directory_is_not_a_task(self):
        (self.tasks / ".git").mkdir()
        self.assertEqual(self.check(self.tasks), 0)

    def test_listed_task_is_missing(self):
        index = json.loads((self.tasks / "index.json").read_text())
        index["tasks"].append({"name": "absent", "source_step": "m3-7", "summary": "s"})
        write_json(self.tasks / "index.json", index)
        self.assert_invalid(self.tasks)

    def test_missing_fixture_repository(self):
        (self.tasks / "one" / "repo" / "keep.txt").unlink()
        (self.tasks / "one" / "repo").rmdir()
        self.assert_invalid(self.tasks)

    def test_bad_transcript_shapes(self):
        call = TRANSCRIPT["replies"][0]
        text = {"text": "done"}
        shapes = {
            "text before a call": [text, call],
            "two texts": [call, text, text],
            "no text": [call],
            "empty": [],
            "both kinds": [dict(call, text="x")],
            "empty calls": [{"tool_calls": []}, text],
            "input not an object": [{"tool_calls": [dict(call["tool_calls"][0], input=[])]}, text],
            "unknown call field": [{"tool_calls": [dict(call["tool_calls"][0], extra=1)]}, text],
            "unknown reply field": [{"tool_calls": call["tool_calls"], "extra": 1}, text],
            "one id twice in a reply": [{"tool_calls": call["tool_calls"] * 2}, text],
        }
        for label, replies in shapes.items():
            with self.subTest(shape=label):
                write_json(self.tasks / "one" / "transcript.json", {"schema": 1, "replies": replies})
                self.assert_invalid(self.tasks)

    def test_judge_member_shapes(self):
        score = {"type": "score", "score": 0.5, "confidence": 0.5,
                 "probabilities": {"0": 0.5, "1": 0.5}, "legend": {"0": "low", "1": "high"}}
        choice = {"type": "choice", "choice": "fix", "confidence": 0.5, "probabilities": {"fix": 1}}
        valid = dict(JUDGE, requests=0, answers=dict(JUDGE["answers"], risk=score, kind=choice))
        write_json(self.tasks / "one" / "transcript.json", dict(TRANSCRIPT, judge=valid))
        self.assertEqual(self.check(self.tasks), 0)
        noul = JUDGE["answers"]["outcome"]
        shapes = {
            "null": None,
            "a list": [],
            "unknown field": dict(JUDGE, extra=1),
            "no model": {"requests": 1, "answers": JUDGE["answers"]},
            "empty model": dict(JUDGE, model=""),
            "negative requests": dict(JUDGE, requests=-1),
            "fractional requests": dict(JUDGE, requests=1.5),
            "boolean requests": dict(JUDGE, requests=True),
            "answers a list": dict(JUDGE, answers=[noul]),
            "no answers": dict(JUDGE, answers={}),
            "answer not an object": dict(JUDGE, answers={"outcome": "yes"}),
            "unknown answer type": dict(JUDGE, answers={"outcome": {"type": "yes_no", "yes": 0.5}}),
            "unknown answer field": dict(JUDGE, answers={"outcome": dict(noul, extra=1)}),
            "missing answer field": dict(JUDGE, answers={"risk": {k: v for k, v in score.items() if k != "legend"}}),
            "answer not a number": dict(JUDGE, answers={"outcome": dict(noul, noul="0.5")}),
            "score level gap": dict(JUDGE, answers={"risk": dict(score, probabilities={"0": 0.5, "2": 0.5})}),
            "score legend short": dict(JUDGE, answers={"risk": dict(score, legend={"0": "low"})}),
        }
        for label, judge in shapes.items():
            with self.subTest(shape=label):
                write_json(self.tasks / "one" / "transcript.json", dict(TRANSCRIPT, judge=judge))
                self.assert_invalid(self.tasks)
        # The strict JSON rules hold inside the member too.
        text = json.dumps(JUDGED_TRANSCRIPT)
        for label, broken in (
            ("duplicate key", text.replace('"requests": 2', '"requests": 2, "requests": 2')),
            ("NaN", text.replace('"noul": 0.25', '"noul": NaN')),
        ):
            with self.subTest(case=label):
                self.assertNotEqual(broken, text)
                (self.tasks / "one" / "transcript.json").write_text(broken)
                self.assert_invalid(self.tasks)

    def test_public_set_passes_check(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(smoke.main(["--check"]), 0)


class BankTests(SmokeTestCase):
    def main(self, *args: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return smoke.main(list(args))

    def test_bank_flag_reads_the_variable(self):
        with patch.dict(os.environ, {smoke.BANK_VARIABLE: str(self.tasks)}):
            self.assertEqual(self.main("--bank", "--check"), 0)
            make_task(self.tasks, "stray")
            self.assertEqual(self.main("--bank", "--check"), smoke.EXIT_INVALID)

    def test_bank_flag_needs_the_variable(self):
        with patch.dict(os.environ, {smoke.BANK_VARIABLE: ""}):
            self.assertEqual(self.main("--bank", "--check"), smoke.EXIT_INVALID)

    def test_variable_alone_keeps_the_public_set(self):
        broken = self.tmp / "broken"
        broken.mkdir()
        with patch.dict(os.environ, {smoke.BANK_VARIABLE: str(broken)}):
            self.assertEqual(self.main("--check"), 0)
            self.assertEqual(self.main("--bank", "--check"), smoke.EXIT_INVALID)


if __name__ == "__main__":
    unittest.main()
