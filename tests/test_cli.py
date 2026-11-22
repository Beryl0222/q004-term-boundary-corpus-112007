"""核对命令行只读查询与状态持久化。"""

from pathlib import Path
import json
import tempfile
import unittest

from termcorpus.cli import main


DATA = Path(__file__).resolve().parents[1] / "data" / "records.json"


class CliTest(unittest.TestCase):
    def _run(self, *args):
        from contextlib import redirect_stdout
        import io

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([str(DATA), *args])
        return code, json.loads(buffer.getvalue())

    def test_check_reports_bad_projection(self):
        code, data = self._run("check")
        self.assertEqual(code, 0)
        problems = {row["annotation"]: row["problems"] for row in data["annotation_problems"]}
        self.assertIn("B2:n2", problems)

    def test_plan_flags_leaks(self):
        code, data = self._run("plan", "--training", "D1", "D4", "D7")
        self.assertEqual(code, 1)
        self.assertEqual(data["training"], ["D7"])
        self.assertEqual(data["checks"], ["check-storage-07", "check-control-02"])

    def test_history_explains_other_manuscript_boundary(self):
        code, data = self._run("history", "相变存储器")
        self.assertEqual(code, 0)
        by_marking = {row["marking"]: row for row in data["occurrences"]}
        self.assertIn("A1", by_marking)
        self.assertIn("B1", by_marking)
        self.assertNotEqual(
            by_marking["A1"]["raw_span"], by_marking["B1"]["annotation"]
        )

    def test_release_refuses_leaking_training_set(self):
        code, data = self._run(
            "release",
            "--training",
            "D1",
            "--id",
            "R9",
            "--date",
            "2026-11-22",
        )
        self.assertEqual(code, 1)
        self.assertFalse(data["samples"] if "samples" in data else [])
        self.assertIn("leaked", data)

    def test_release_records_confirmed_only(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            self._run(
                "review",
                "confirm",
                "A1:t3",
                "--state",
                str(state),
                "--save",
                str(state),
            )
            code, data = self._run(
                "release",
                "--training",
                "D7",
                "--id",
                "R2",
                "--date",
                "2026-11-22",
                "--state",
                str(state),
            )
            self.assertEqual(code, 0)
            self.assertEqual(data["checks"], ["check-storage-07", "check-control-02"])
            self.assertEqual(data["training"], ["D7"])
            # D7 的标注都还没确认，不能作为已标注样本导出。
            self.assertEqual(data["samples"], [])

    def test_review_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            code, data = self._run(
                "review",
                "confirm",
                "A1:t3",
                "--state",
                str(state),
                "--save",
                str(state),
            )
            self.assertEqual(code, 0)
            self.assertTrue(state.exists())
            saved = json.loads(state.read_text(encoding="utf-8"))
            record = next(
                row for row in saved["records"] if row["annotation"] == "A1:t3"
            )
            self.assertEqual(record["status"], "confirmed")
            code, preview = self._run(
                "review",
                "preview",
                "A1:m1",
                "--state",
                str(state),
            )
            self.assertEqual(code, 0)
            self.assertTrue(
                any(item["will_reopen"] for item in preview["impact"])
            )


if __name__ == "__main__":
    unittest.main()
