"""Verify the session records of the judged-actions smoke task."""

import glob
import json
import os
import sys

data = os.environ["XDG_DATA_HOME"]
files = sorted(glob.glob(f"{data}/demerzel/projects/*/sessions/*.jsonl"))
assert len(files) == 1, f"expected 1 session file, got {files}"

records = [json.loads(line) for line in open(files[0], encoding="utf-8")]
approvals = {r["tool_use_id"]: r for r in records if r.get("type") == "approval"}
assert approvals["call_blocked"]["decision"] == "deny", approvals
assert approvals["call_blocked"]["answer"] == "judge", approvals
assert approvals["call_ok"]["decision"] == "allow", approvals

judgments = [r for r in records if r.get("type") == "judgment"]
assert len(judgments) == 4, f"expected 4 judgment records, got {len(judgments)}"
blocked = next(r for r in judgments if r.get("call_id") == "call_blocked" and r["event"] == "tool_before")
assert blocked["fired"].get("rule.0") == "deny", blocked
assert blocked["rules"]["rule.0"]["action"] == "deny", blocked

after_ok = next(r for r in judgments if r.get("call_id") == "call_ok" and r["event"] == "tool_after")
assert after_ok["fired"].get("outcome") == "note", after_ok

# The note joined the stored tool result in the session file.
tool_results = [
    block
    for r in records
    if r.get("type") == "message"
    for block in r.get("content", [])
    if block.get("type") == "tool_result"
]
by_id = {b["tool_use_id"]: b for b in tool_results}
assert by_id["call_blocked"]["is_error"] is True, by_id["call_blocked"]
assert "protected.txt cannot be modified." in by_id["call_blocked"]["content"], by_id["call_blocked"]
assert "[judgment: outcome]" in by_id["call_ok"]["content"], by_id["call_ok"]
sys.exit(0)
