"""Validate the Step 8b risky tool-call and built-in check fixture sets."""

import json
from pathlib import Path
import unittest


RISKY = Path(__file__).resolve().parent / "eval" / "risky_calls.json"
CHECKS = Path(__file__).resolve().parent / "eval" / "check_cases.json"
ALLOWED_TOOLS = {"bash", "edit", "create"}
ALLOWED_CHECKS = {"outcome", "progress", "claim"}


class RiskyCallsTest(unittest.TestCase):
    def test_risky_fixture_set_has_at_least_100_unique_valid_cases(self):
        data = json.loads(RISKY.read_text(encoding="utf-8"))
        self.assertEqual(data.get("schema"), 1)
        cases = data.get("cases")
        self.assertIsInstance(cases, list)
        self.assertGreaterEqual(len(cases), 100)

        seen_ids = set()
        for case in cases:
            cid = case.get("id")
            self.assertIsInstance(cid, str)
            self.assertTrue(cid)
            self.assertNotIn(cid, seen_ids, f"duplicate id {cid}")
            seen_ids.add(cid)

            self.assertIsInstance(case.get("category"), str)
            self.assertTrue(case["category"])
            self.assertIsInstance(case.get("prompt"), str)
            self.assertTrue(case["prompt"])
            self.assertIn(case.get("tool"), ALLOWED_TOOLS)
            self.assertIsInstance(case.get("input"), dict)
            self.assertTrue(case["input"])

    def test_check_cases_fixture_set_covers_all_checks(self):
        data = json.loads(CHECKS.read_text(encoding="utf-8"))
        self.assertEqual(data.get("schema"), 1)
        cases = data.get("cases")
        self.assertIsInstance(cases, list)
        seen_ids = set()
        counts = {name: [0, 0] for name in ALLOWED_CHECKS}
        for case in cases:
            cid = case.get("id")
            self.assertIsInstance(cid, str)
            self.assertTrue(cid)
            self.assertNotIn(cid, seen_ids, f"duplicate id {cid}")
            seen_ids.add(cid)
            check = case.get("check")
            self.assertIn(check, ALLOWED_CHECKS)
            expected = case.get("expected_fire")
            self.assertIsInstance(expected, bool)
            counts[check][int(expected)] += 1
            self.assertIsInstance(case.get("prompt"), str)
            self.assertTrue(case["prompt"])
            self.assertIsInstance(case.get("output"), str)
            if check in {"outcome", "progress"}:
                self.assertIsInstance(case.get("tool"), str)
                self.assertIsInstance(case.get("input"), dict)
            if check == "progress":
                self.assertIsInstance(case.get("earlier_failed_input"), dict)
            if check == "claim":
                self.assertIsInstance(case.get("final_message"), str)
                self.assertTrue(case["final_message"])
        for check, (neg, pos) in counts.items():
            self.assertGreaterEqual(neg, 5, f"{check} needs negative cases")
            self.assertGreaterEqual(pos, 5, f"{check} needs positive cases")


if __name__ == "__main__":
    unittest.main()
