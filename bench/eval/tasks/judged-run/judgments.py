"""The judgment records of the judged-run task.

The oracle runs this script in the final state of the copy. It reads
the one session file under the scratch data directory and checks:

- one `judgment` record for each judge request that the transcript
  expects, with these events, calls, and checks, and no error;
- each record has the scripted model and answers;
- each `tool_before` record pairs with the `approval` record of its call.
"""

import glob
import json
import os
import sys
from pathlib import Path

# (event, call id, tool, checks) of each judge request. The approval
# check asks two questions; each other check asks the question of its
# own name.
EXPECTED = sorted([
    ("tool_before", "call_test", "bash", ("approval",)),
    ("tool_after", "call_test", "bash", ("outcome",)),
    ("tool_before", "call_edit", "edit", ("approval",)),
    ("tool_before", "call_lint", "bash", ("approval",)),
    ("tool_before", "call_lint_again", "bash", ("approval",)),
    ("tool_after", "call_lint_again", "bash", ("progress",)),
    ("turn_end", None, None, ("claim",)),
], key=repr)
QUESTIONS = {
    "approval": ("approval.destructive", "approval.in_scope"),
    "outcome": ("outcome",),
    "progress": ("progress",),
    "claim": ("claim",),
}


def fail(reason: str) -> None:
    print(f"judgments: {reason}")
    sys.exit(1)


task = Path(os.environ["DEMERZEL_EVAL_TASK_DIR"])
judge = json.loads((task / "transcript.json").read_text())["judge"]
pattern = os.path.join(os.environ["XDG_DATA_HOME"], "demerzel", "projects", "*", "sessions", "*.jsonl")
sessions = glob.glob(pattern)
if len(sessions) != 1:
    fail(f"expected one session file, found {len(sessions)}")
records = [json.loads(line) for line in Path(sessions[0]).read_text().splitlines()]
judgments = [record for record in records if record.get("type") == "judgment"]
approvals = {record.get("tool_use_id") for record in records if record.get("type") == "approval"}

if len(judgments) != judge["requests"]:
    fail(f"{len(judgments)} judgment records; the transcript expects {judge['requests']}")
seen = sorted(
    ((r["event"], r.get("call_id"), r.get("tool"), tuple(sorted(r["checks"]))) for r in judgments),
    key=repr,
)
if seen != EXPECTED:
    fail(f"the records ask {seen}; expected {EXPECTED}")
for record in judgments:
    where = f"{record['event']} {record.get('call_id')}"
    if "error" in record:
        fail(f"{where}: the record has an error: {record['error']}")
    if record["instance"] != "judge" or record.get("model") != judge["model"]:
        fail(f"{where}: instance {record['instance']!r} and model {record.get('model')!r}")
    if set(record["checks"].values()) != {"log"}:
        fail(f"{where}: a check is not at log: {record['checks']}")
    asked = sorted(question for check in record["checks"] for question in QUESTIONS[check])
    expected = {
        question: {"type": "yes_no", "yes": judge["answers"][question]["noul"]} for question in asked
    }
    if record.get("answers") != expected:
        fail(f"{where}: answers {record.get('answers')}; expected {expected}")
    if record["event"] == "tool_before" and record["call_id"] not in approvals:
        fail(f"{where}: no approval record has this call id")
print(f"judgments: {len(judgments)} records match")
