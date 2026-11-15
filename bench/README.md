# Demerzel benchmark suite

Historical receipts retain their original commit IDs and measurements.
Use the [commit map](../docs/history-commit-map.tsv) to find the corresponding current commit.
See [Repository history](../docs/history.md) for details.

Demerzel treats performance claims as product features. The scripts in this
directory produce every performance number that Demerzel publishes. The raw
data for each number goes into `results/`. The "Benchmark program" section of
[`docs/plan.md`](../docs/plan.md) describes the full program.

## Rules

1. **Reproducible.** The suite lives in this repository, next to the code. You
   can run the same scripts and compare your numbers with ours.
2. **Pinned.** Each report records the exact version of each competitor.
3. **Disclosed.** Each report records the machine data: CPU, RAM, kernel, and
   filesystem.
4. **Statistical.** Timing benchmarks use warmup runs and 50 or more measured
   runs. Each report keeps the confidence intervals that hyperfine calculates.
   The cache-hit measurement is the exception: it runs two live sessions and
   keeps no interval. Its claim is the per-call ratio, not a timing.
5. **Comparable.** When a report compares Demerzel with another agent, it uses
   the same metric as that agent, on the same machine and OS. On Linux the
   memory metric is PSS from `/proc/<pid>/smaps_rollup`, the metric the
   comparison targets publish. On macOS it is physical footprint, the metric
   Apple's tools report. Numbers never compare across operating systems.
   [`docs/plan.md`](../docs/plan.md) names the comparison targets.
6. **Raw data included.** We commit the raw JSON output next to each published
   summary.

## Layers

| Layer | Measures | Method | Status |
|---|---|---|---|
| 1 | Boot time, memory footprint, binary size | hyperfine; on Linux, PSS and USS from `smaps_rollup` and cgroup-v2 `memory.peak`; cargo-bloat | The boot runner, the PTY probe, the PSS sampler, and the idle-session driver are here. |
| 2 | Harness overhead per turn and per tool call; render throughput | A mock OpenAI-compatible server replays recorded transcripts for the turn benches. The render bench drives the transcript and the renderer in process. criterion runs micro-benchmarks. iai-callgrind gates CI on instruction counts. | The criterion turn benchmark and the render benchmark are here. iai-callgrind arrives with CI in M5. |
| 3 | Agentic capability of the harness plus a model | Terminal-Bench 3.0 as the headline suite; Terminal-Bench 2.1 as the longitudinal track; fixed-model A/B benchmarks against other harnesses; the Coding Agent Index as the external target | The hermetic smoke runner is here. It checks the harness against a scripted model and measures no capability. The capability suites come with M3/M4. |
| 4 | Loop efficiency | Prompt-cache hit rate on replayed sessions; tokens per completed task; memory recall precision | The cache-hit measurement is here. The rest comes with M4. |

## The dogfood driver

`bench/dogfood/run.sh` is not a benchmark. It drives the release binary
under tmux against the configured provider and writes the 120x40 screen
as text after each step: a bash call with its question, the tail and
the expand key on a tall output, a declined call. Read the captures in
`target/dogfood/` to check the screen without a terminal in front of
you. It needs tmux and a logged-in provider, and it spends a few
hundred tokens per prompt.

Review `target/dogfood/` before you share a capture. A capture can
contain prompt text, tool output, file content, and model text.

## How to run

```bash
cargo build --release

# Machine spec. Each published number references one of these files.
bench/machine.py

# Layer 1: boot time, pure binary start.
bench/boot/run.sh

# Layer 1: time to interactive, warm and cold.
bench/boot/pty_probe.py
bench/boot/pty_probe.py --cold

# Layer 1: memory of an idle REPL session, over 10 launches.
bench/memory/idle.py

# Layer 1: native memory counters for running agent processes.
bench/memory/memory.py <pid> [pid...]

# Layer 2: harness-added latency per turn, against the mock server.
bench/turn/run.sh

# Layer 2: frame cost and streamed-text throughput of the full screen.
bench/render/run.sh

# The harness end to end, against a scripted model. See "The smoke runner".
python3 bench/eval/smoke.py

# Layer 4: prompt-cache hit rate on a replayed session, live provider.
# Needs credentials and network access, and it costs tokens.
bench/cache/replay.py --provider anthropic
bench/cache/replay.py --provider openai-codex
```

The suite runs on Linux and macOS, the two reference platforms
([`docs/bench-platforms.md`](../docs/bench-platforms.md)). Result files
carry an OS-arch slug, for example `boot-2026-08-23-abc1234-macos-aarch64.json`.
The memory probes run on both platforms. Linux reports PSS and USS.
macOS reports current physical footprint and lifetime peak physical footprint.
The old `memory/pss.py` command forwards to `memory/memory.py`.
Per-file cold mode (`--cold`) runs on Linux.
macOS uses explicit `--cold-system` mode, which purges the system disk cache and requires administrator authorization.
The [cold-boot decision](../docs/bench-platforms.md#macos-cold-boot-decision) records the scope and verification.

The boot benchmark needs `hyperfine`. On Arch, install it with
`pacman -S hyperfine`. On macOS, install it with
`brew install hyperfine`. The probes need Python 3.13 or newer. On
macOS, install Python from Homebrew or python.org. `bench/harness.py`
holds the shared probe machinery. `platform_linux.py` and
`platform_macos.py` hold the native probes.

Method notes:

- The probe gives each run a private data home and HOME with a synthetic
  auth store. A run needs no login and makes no network call.
- The probe records its own spawn floor, measured through the identical
  PTY path with `printf`. The floor sits inside every sample.
- Cold mode, on Linux, evicts the binary from the page cache with `posix_fadvise`
  and verifies the eviction with `mincore`. Shared libraries and
  metadata caches stay warm, so the cold number tracks the binary read
  and moves with binary size. A true cold start needs a fresh boot.
- The memory driver waits for the first prompt, settles for one second,
  and samples each of 10 launches by default.
  On Linux, anonymous memory measures the harness's heap and stacks.
  PSS and USS also count clean mapped binary text, which changes with page-cache state.
  On macOS, `proc_pid_rusage` supplies current and lifetime peak physical footprint.
  The JSON fields end in `_kb`, but all values use KiB (1,024 bytes).
  Footprint and PSS do not compare across platforms.
  The peak belongs to one process, from launch through the sample.
  It is not a concurrent-session peak.
  The PID command samples processes in sequence and does not add their peaks.
  Step 9 delays HTTP client construction until the first model turn.
  The idle fixture now measures startup without an HTTP client.
- The turn suite reports `transport_floor` next to the turn benches.
  Harness-added latency per turn is `turn_text` minus
  `transport_floor`. `turn_text_compat` runs the same turn through a
  named `openai-compat` instance on the Chat Completions route.
  `turn_text_responses` runs it through a named `openai-responses`
  instance on the Responses API route. Session files land under the
  cargo target directory, so appends hit a real filesystem.
- The render suite draws into a headless frame buffer. The layout and
  the buffer diff are the live interface's work; the terminal write is
  not. The [fixture guide](../docs/render-bench.md) explains the M2.5 workloads and comparison limits.
  The three core benchmarks are:
  1. `frame` is one redraw with no new content, the floor of a frame.
  2. `ingest` pushes 64 KiB of text in 16-byte deltas with no frames.
  3. `stream` pushes the same text with a frame every 64 deltas. It is
     frame-bound, so its throughput is frames per second.
- The cache-hit measurement replays the user side of one session: a
  fixed list of user turns. It runs them through the real engine
  against one live provider, as two fresh sessions one after the
  other. The model side is live, so the two sessions differ in what
  the model reads. The report gives `cache_read_tokens /
  input_tokens` per model call, per turn, and per session.
  `input_tokens` counts the cached parts for both providers. The
  engine advertises the full tool set in `ask` mode with no approver,
  so only the read tools run. Sessions land in a private temporary
  data root that goes away with the run. Credentials resolve the way
  a session resolves them. What to compare across milestones:
  1. The per-call ratio. A warm call that reads its predecessor's
     whole prompt shows a stable prefix.
  2. The first call of the second session. It shares its whole prefix
     with the first session's first call. A non-zero rate shows the
     prefix holds across sessions; a zero can also mean the prefix is
     under the provider's minimum cacheable length.
  Session totals move with the model's tool choices, so they do not
  compare.

Run the probe checks with `python3 -m unittest discover -s bench -p 'test_*.py'`.
On macOS, they check the native structure against the installed SDK.
They also check current footprint growth and lifetime peak retention after memory release.

## The smoke runner

`bench/eval/smoke.py` checks the harness end to end with no model and
no network. Each task is a fixture repository, a scripted model
transcript, and an oracle. The runner serves the transcript from
`demerzel-mock` and runs the one-shot CLI against it. Then the runner
checks two things:

1. The mock used every reply of the transcript and refused no request.
2. The oracle accepts the final state of the files.

The mock serves the replies in request order. Thus a prompt change does
not break a task. A harness change breaks a task when it changes the
final state, leaves a reply unused, or makes a request that the mock
refuses. CI runs the public set in the `gates` job, after the release
build.

The smoke run does not prove these things:

- The content of a `read` result or of a model request. The mock
  checks only one part of each request: one tool result for each call
  id of the previous reply.
- The `tool_calls` that the harness sends back in its assistant
  message. The mock does not check them.
- The failure of a call that must fail. Such a call proves only that
  it changed nothing. A call to a missing tool passes the same way.

The runner needs Python 3.11 or newer. Run it after a release build:

```bash
cargo build --release --locked
python3 bench/eval/smoke.py                # every public smoke task
python3 bench/eval/smoke.py create-file    # one task, by name
python3 bench/eval/smoke.py --check        # validate every task, run none
```

`--binary` and `--mock` select other binaries. The defaults are
`target/release/demerzel` and `target/release/demerzel-mock` in this
repository. `--tasks DIR` selects another task directory. `--check`
validates every task of the directory, also when you give task names.

The runner prints one line for each task, with the result and the
duration. A total line with the wall time follows. A failed task names
the stage that failed and shows the tails of the related output. Each
tail holds 20 lines at most, each cut at 300 characters. A failure at
an agent stage shows up to three tails: the agent's stderr, the agent's
stdout, and the mock's stderr. The stages are:

- `setup`: the runner did not make the scratch directory or the
  config file.
- `mock start`: the mock did not start, or its first line is not a
  loopback base URL.
- `agent exit`: the one-shot run did not start, or it exited with a
  status other than 0.
- `timeout`: the one-shot run did not end in 30 seconds.
- `mock count`: the mock reports an unused reply or a refused request,
  or its count line is missing or malformed.
- `oracle`: the oracle did not start, or it exited with a status other
  than 0.
- `oracle timeout`: the oracle did not end in 30 seconds.

| Exit status | Meaning |
|---|---|
| 0 | Every selected task passes. |
| 1 | One or more tasks fail. |
| 2 | The arguments are wrong, the tasks do not validate, or a binary is missing. |
| 129 | SIGHUP stopped the run. |
| 130 | Ctrl-C (SIGINT) stopped the run. |
| 143 | SIGTERM stopped the run. |

The public smoke set is in `bench/eval/tasks/`. Each oracle compares
the final files with the task's `expected/` directory. This table gives
what each final state proves:

| Task | The final state proves |
|---|---|
| `edit-greeting` | `edit` changed line 2 of an executable script and kept its mode. `bash` ran the script; `greet.out` holds its output and exit status 0. |
| `create-file` | `create` made a file in a new directory. A second `create` of the same path did not change the file. |
| `parallel-edits` | Two parallel `edit` calls in one reply each changed their own file. The two parallel `read` calls before them leave no trace in the files. |
| `recover-from-failures` | `check.before` holds the output and exit status 1 of the first check. The `edit` with a stale tag changed nothing. The `edit` with the correct tag wrote `version = 2`. `check.after` holds the output and exit status 0 of the second check. |
| `project-config` | The run used the mock. The project config file names another provider, model, and endpoint. |

If the harness accepts the stale tag, `version.txt` holds `version = 3`,
and the next `edit` fails on its own tag. Each scripted `bash` call
appends to its evidence file, so a call that runs twice changes the
final state. A failed `read` changes no file, so only the mock's check
of the tool-result ids covers `read`.

`bench/test_eval_smoke.py` checks the runner with stand-in binaries.
The probe-check command above runs it.

### The task layout

The public smoke set and the private bank use one layout:

```text
<root>/index.json               the task list
<root>/<name>/task.json         the prompt and the oracle
<root>/<name>/transcript.json   the scripted model replies
<root>/<name>/repo/             the fixture files
```

A task directory can hold more files next to `repo/`, for example the
expected final state. The runner copies only `repo/` and runs the agent
in the copy. The `create` and `edit` tools cannot write outside the
copy. The `read` and `bash` tools can reach any path, also the task
directory, so a task must not depend on the secrecy of these files.

### The formats

The runner's validator owns the three formats. The mock's parser only
maps each reply to the wire. Each format has `schema: 1`. Schema 1 grows
by optional fields. A reader refuses a field that it does not know. The
number changes only when a field changes its meaning or becomes
required.

`--check` and each run refuse these:

- an unknown field or a missing field;
- a `schema` other than 1;
- a key that occurs twice in one object;
- `NaN`, `Infinity`, `-Infinity`, or a number out of range, such as
  `1e400`;
- nesting deeper than 100 levels;
- two calls with one `id` in one reply, because the harness joins each
  tool result to its call by `id`;
- a string with a NUL character, or a string that is not valid UTF-8,
  such as a lone surrogate;
- a listed task without a directory;
- a task directory that the index does not list.

A directory with a name that starts with `.` is not a task.

`index.json` lists the tasks in run order:

```json
{"schema": 1, "tasks": [
  {"name": "create-file", "source_step": "synthetic", "summary": "Create a new file."}
]}
```

- `name`: the task directory. It holds letters, digits, `.`, `_`, and
  `-`. The first character is a letter or a digit.
- `source_step`: the origin of the task. A public task gives
  `synthetic`. A bank task gives `m<milestone>-<step>`: the milestone
  and the plan step whose dogfood found the failure. Examples: `m3-7`,
  `m3-8a`, and `m2.5-8f`.
- `summary`: one line about what the task checks.

`task.json` is the scenario document of a task. Later steps extend it
with optional fields; a task gets no second document.

```json
{"schema": 1, "prompt": "Add docs/usage.txt.",
 "oracle": ["sh", "-c", "diff -r . \"$DEMERZEL_EVAL_TASK_DIR/expected\""]}
```

- `prompt`: the prompt of the one-shot run.
- `oracle`: a command, as a list of arguments. It runs in the final
  state of the copy. Exit status 0 passes. The runner starts the
  command without a shell. Use `sh -c` for a command that needs one.
  `DEMERZEL_EVAL_TASK_DIR` names the task directory, so the oracle can
  read files outside `repo/`. The oracle gets this variable; the agent
  does not.

`transcript.json` is the input of `demerzel-mock`:

```json
{"schema": 1, "replies": [
  {"tool_calls": [{"id": "call_1", "name": "create",
                   "input": {"path": "docs/usage.txt", "content": "Usage.\n"}}]},
  {"text": "Done."}
]}
```

Each reply has `text` or `tool_calls`, not both. Two or more calls in
one reply are parallel calls. The mock serves each reply once, in
request order. The mock refuses these requests with an error and
counts them:

- a request after the last reply;
- a request after a `tool_calls` reply that does not carry one tool
  result for each call id of that reply;
- a request to any other path.

The runner accepts one shape only: replies with tool calls, then one
text reply as the last reply. This shape is an early check. The proof
that the run used every reply is the mock's count. The mock serves
until its stdin closes. Then it prints the count of served replies,
unused replies, and refused requests.

Each `input` must match the input schema of its tool. The `edit` tool
takes the snapshot tag of the file. The tag depends only on the bytes
of the file, so a transcript can give it.

### Isolation

Each task runs in a fresh scratch directory:

1. The runner copies `repo/` to the scratch directory.
2. `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`, and
   `XDG_CACHE_HOME` point to empty scratch directories. The user's
   config, profile, instruction files, logins, and sessions stay out.
3. The run keeps only these variables of the environment: `PATH`,
   `LANG`, `LC_ALL`, `LC_CTYPE`, and `TMPDIR`. Each `DEMERZEL_`
   variable of the parent and each provider key stays out.
4. A user config file in the scratch config directory defines the
   `smoke` instance, with `kind = "openai-compat"` and the mock's
   URL. The mock needs no key, so the instance sends none.
5. The runner runs `demerzel --auto -- <prompt>` in the copy, with
   `DEMERZEL_PROVIDER=smoke`, `DEMERZEL_MODEL=scripted`, and
   `DEMERZEL_BASE_URL` set to the mock. The session file goes to the
   scratch data directory.
6. The runner closes the mock's stdin, reads the count, and stops the
   mock. Then it runs the oracle and removes the scratch directory.

A variable outranks a project config file. Thus a fixture's
`.demerzel/config.toml` cannot move the run to another instance, model,
or endpoint. The [trust rules](../docs/config.md#trust-rules) also stop
a project file from defining an instance. A project role can add
fallback targets; see [Roles](../docs/config.md#roles). A fallback runs
only after a failed model call, and the task then fails:

- The mock counts its refusal of the failed call.
- A shipped adapter finds no credentials in the scratch homes.
- A fallback to the `smoke` instance with another model goes back to
  the mock, which refuses it or serves it a reply out of order.

Each child process runs in its own process group. A stop sends SIGINT
to the agent, or SIGTERM to the mock or the oracle. After 3 seconds it
sends SIGKILL. Then it kills the rest of the group, so a background
child cannot survive. The runner holds SIGINT, SIGTERM, and SIGHUP
while it stops a child or removes a scratch directory. The mock also
stops when the runner dies, because its stdin closes.

### The private bank

The bank holds reproductions of dogfood failures. It is a directory
outside this repository, in the same layout, with its own
`index.json`. `DEMERZEL_EVAL_BANK` names the bank directory. Only the
runner reads the variable. The `demerzel` binary does not read it.

Validate and run the bank with `--bank`:

```bash
export DEMERZEL_EVAL_BANK=/path/to/bank
python3 bench/eval/smoke.py --bank --check
python3 bench/eval/smoke.py --bank
```

Without `--bank`, the runner uses the public set, also when the
variable is set. A bank task can fail until its fix lands, so the bank
never replaces the public set by accident. `--bank` with an unset or
empty variable is an error. `--bank` and `--tasks` exclude each other.

Keep private data out of the public set. Each public task is a
synthetic fixture.
