"""核对泄漏分组、发布隔离、导出与交付影响。"""

from pathlib import Path
import copy
import json
import unittest

from termcorpus.grouping import build_grouping, leak_evidence, path_between
from termcorpus.records import load_packet, packet_from_dict
from termcorpus.release import (
    ReleaseError,
    audit_provenance,
    export_labeled,
    leak_impact,
    plan_release,
    release_record,
)
from termcorpus.review import ReviewState, adjudicate


DATA = Path(__file__).resolve().parents[1] / "data" / "records.json"


class GroupingTest(unittest.TestCase):
    def setUp(self):
        self.packet = load_packet(DATA)
        self.grouping = build_grouping(self.packet)

    def test_same_manual_and_its_qa_share_a_group(self):
        self.assertEqual(
            self.grouping.document_to_group["D1"],
            self.grouping.document_to_group["D2"],
        )
        self.assertEqual(
            self.grouping.member_to_group["QX"],
            self.grouping.member_to_group["D2"],
        )

    def test_shared_term_alone_does_not_group(self):
        # D3、D7 都出现"相变存储器"，但不是片段、转述或同组问答。
        self.assertNotEqual(
            self.grouping.document_to_group["D3"],
            self.grouping.document_to_group["D7"],
        )
        self.assertNotEqual(
            self.grouping.document_to_group["D3"],
            self.grouping.document_to_group["D1"],
        )

    def test_transcription_and_pointer_join_checks(self):
        # D4 是转述且指向旧机问答，经 D5 与 D2 的校验组相连。
        group = self.grouping.group_for("D4")
        self.assertIn("check-storage-07", group.pins)
        self.assertIn("check-control-02", group.pins)
        kinds = {edge.kind for edge in group.edges}
        self.assertIn("paraphrase", kinds)
        self.assertIn("answer_pointer", kinds)

    def test_mentions_alone_is_not_evidence(self):
        data = copy.deepcopy(json.loads(DATA.read_text(encoding="utf-8")))
        # 去掉两条实质证据，只保留 mentions 著录。
        data["catalogue"][3]["raw"] = "D4 另有一条与 D2、D5 无关的记录。"
        packet = packet_from_dict(data)
        grouping = build_grouping(packet)
        self.assertNotEqual(
            grouping.document_to_group["D4"], grouping.document_to_group["D2"]
        )

    def test_leak_evidence_names_the_basis(self):
        findings = leak_evidence(self.grouping, ["D4"], self.packet)
        pins = {finding["pin"] for finding in findings}
        self.assertEqual(pins, {"check-storage-07", "check-control-02"})
        for finding in findings:
            self.assertTrue(finding["basis"])
            self.assertTrue(finding["path"])

    def test_path_reports_edges(self):
        path = path_between(self.grouping, "D1", "D5")
        self.assertTrue(path)
        self.assertEqual(path[0].a, "D1")


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.packet = load_packet(DATA)
        self.grouping = build_grouping(self.packet)

    def test_plan_removes_leaking_material_and_keeps_coverage_note(self):
        plan = plan_release(self.packet, ["D1", "D4", "D7"], self.grouping)
        self.assertEqual(plan.training, ["D7"])
        self.assertEqual(sorted(plan.removed), ["D1", "D4"])
        coverage = {item["area"]: item for item in plan.missing_coverage}
        self.assertIn("存储", coverage)
        self.assertEqual(coverage["存储"]["status"], "reduced")
        self.assertIn("check-storage-07", coverage["存储"]["pins"])
        # 校验编号保持原样，不被重排。
        self.assertEqual(
            plan.checks, ["check-storage-07", "check-control-02"]
        )

    def test_plan_rejects_reselected_check_ids(self):
        with self.assertRaises(ReleaseError):
            plan_release(
                self.packet,
                ["D7"],
                self.grouping,
                required_checks=["check-storage-07"],
            )

    def test_training_material_cannot_be_a_check_record(self):
        plan = plan_release(self.packet, ["D2", "D7"], self.grouping)
        self.assertIn("D2", plan.removed)
        self.assertTrue(
            any(item.document == "D2" for item in plan.leaked)
        )

    def test_impact_names_delivered_versions(self):
        impact = leak_impact(self.packet, ["D1"], self.grouping)
        self.assertTrue(impact.releases)
        release = impact.releases[0]
        self.assertEqual(release["release"], "R1")
        self.assertIn("check-storage-07", release["pins"])
        self.assertIn("通知", impact.message)

    def test_no_impact_when_material_never_delivered(self):
        impact = leak_impact(self.packet, ["D7"], self.grouping)
        self.assertFalse(impact.releases)

    def test_export_requires_confirmation(self):
        state = ReviewState()
        state.ensure(self.packet)
        result = export_labeled(
            self.packet, state, doc_ids=["D7"], grouping=self.grouping
        )
        self.assertEqual(result.samples, [])
        self.assertTrue(result.skipped)

    def test_exported_spans_are_utf8_bytes_and_nested(self):
        state = ReviewState()
        state.ensure(self.packet)
        for annotation_id in ("A1:t3", "A1:m2"):
            adjudicate(state, self.packet, annotation_id, action="confirm")
        result = export_labeled(
            self.packet,
            state,
            doc_ids=["D1"],
            grouping=self.grouping,
            include_pinned=True,
        )
        self.assertEqual(len(result.samples), 1)
        sample = result.samples[0]
        text = sample["text"]
        annotations = [
            annotation
            for source in sample["sources"]
            for annotation in source["annotations"]
        ]
        self.assertTrue(annotations)
        for annotation in annotations:
            start, end = annotation["byte_span"]
            self.assertEqual(
                text.encode("utf-8")[start:end].decode("utf-8"),
                annotation["snapshot"],
            )
        spans = [tuple(item["byte_span"]) for item in annotations]
        for start, end in spans:
            for other_start, other_end in spans:
                if (start, end) == (other_start, other_end):
                    continue
                self.assertFalse(
                    start < other_start < end < other_end,
                    "导出不得含交叉区间",
                )

    def test_pinned_group_is_not_exported_by_default(self):
        state = ReviewState()
        state.ensure(self.packet)
        for annotation_id in self.packet.annotations:
            adjudicate(state, self.packet, annotation_id, action="confirm")
        result = export_labeled(
            self.packet, state, doc_ids=["D1"], grouping=self.grouping
        )
        self.assertEqual(result.samples, [])
        self.assertIn("校验", result.skipped[0]["reason"])

    def test_release_record_keeps_checks_and_reports_leaks(self):
        state = ReviewState()
        state.ensure(self.packet)
        plan = plan_release(self.packet, ["D1", "D7"], self.grouping)
        record = release_record(
            self.packet,
            plan,
            state,
            release_id="R2",
            date="2026-11-22",
            grouping=self.grouping,
        )
        self.assertEqual(record["checks"], ["check-storage-07", "check-control-02"])
        self.assertEqual(record["training"], ["D7"])
        self.assertTrue(record["leaks"])
        self.assertEqual(record["removed"], ["D1"])

    def test_provenance_lists_source_and_offsets(self):
        rows = audit_provenance(self.packet, self.grouping)
        by_id = {row["annotation"]: row for row in rows}
        self.assertEqual(by_id["B1:t1"]["offset_unit"], "utf16")
        self.assertEqual(by_id["A1:t3"]["offset_unit"], "unicode_char")
        self.assertEqual(by_id["A1:t3"]["group"], by_id["B1:t1"]["group"])
        self.assertFalse(by_id["A1:t3"]["pinned"])
        self.assertTrue(by_id["A1:t3"]["pinned_group"])


if __name__ == "__main__":
    unittest.main()
