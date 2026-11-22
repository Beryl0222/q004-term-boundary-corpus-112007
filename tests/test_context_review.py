"""核对语境判定与本地裁定不重置整句。"""

from pathlib import Path
import unittest

from termcorpus.context import (
    annotation_context,
    classify,
    document_term_occurrences,
    models_in,
)
from termcorpus.records import load_packet
from termcorpus.review import (
    ReviewState,
    adjudicate,
    affected_annotations,
    boundary_view,
    literal_boundary_history,
    preview_group_impact,
    preview_impact,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "records.json"


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.packet = load_packet(DATA)

    def test_negation_scope_does_not_swallow_positive_use(self):
        context = classify(
            "答：不是磁带，XC芯片使用Ａ-1相变存储器。",
            15,
            20,
            "相变存储器",
        )
        self.assertEqual(context.label, "肯定使用")
        self.assertEqual(context.sense, "器件实体")
        self.assertTrue(any("同句另有否定" in note for note in context.notes))

    def test_negated_term_is_not_promoted_to_definition(self):
        item = next(
            occurrence
            for occurrence in document_term_occurrences(
                self.packet, self.packet.documents["D3"]
            )
            if occurrence.literal == "相变存储器"
        )
        self.assertEqual(item.context.label, "术语提及")
        self.assertEqual(item.context.sense, "术语提及")
        self.assertTrue(any("该部件" in note for note in item.context.notes))

    def test_comparison_mention_is_distinct_from_use(self):
        item = next(
            occurrence
            for occurrence in document_term_occurrences(
                self.packet, self.packet.documents["D7"]
            )
            if occurrence.literal == "相变存储器"
        )
        self.assertEqual(item.context.label, "比较提及")

    def test_variant_spelling_is_reported_not_merged(self):
        items = document_term_occurrences(self.packet, self.packet.documents["D4"])
        variant = next(item for item in items if item.literal == "相变储存")
        self.assertEqual(variant.variant_of, "相变存储器")
        self.assertIsNone(
            next(
                (
                    item
                    for item in items
                    if item.literal == "相变储存" and item.variant_of is None
                ),
                None,
            )
        )

    def test_fullwidth_and_halfwidth_models_are_same_identity(self):
        self.assertEqual(models_in("使用Ａ-1"), models_in("使用A-1"))

    def test_annotation_context_uses_raw_positions(self):
        annotation = self.packet.annotations["A1:t3"]
        context = annotation_context(self.packet, annotation)
        self.assertEqual(context.label, "肯定使用")


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.packet = load_packet(DATA)
        self.state = ReviewState()
        self.state.ensure(self.packet)

    def _confirm(self, *annotation_ids):
        for annotation_id in annotation_ids:
            adjudicate(self.state, self.packet, annotation_id, action="confirm")

    def test_boundary_view_shows_both_sources_and_cross_word_entities(self):
        view = boundary_view(self.packet, "D1")
        self.assertEqual({source["source"] for source in view["sources"]}, {"marking_a", "marking_b"})
        cross = {item["entity"] for item in view["cross_word_entities"]}
        self.assertIn("A1:m1", cross)
        self.assertIn("B1:n1", cross)
        boundaries = [item for item in view["comparisons"] if item["kind"] == "boundary"]
        self.assertTrue(
            any(
                item["a"]["literal"] == "相变存储器"
                and item["b"]["literal"] == "相变"
                for item in boundaries
            )
        )

    def test_local_adjudication_only_reopens_affected_confirmations(self):
        self._confirm("A1:t0", "A1:t1", "A1:t3", "A1:m2")
        result = adjudicate(
            self.state,
            self.packet,
            "A1:m1",
            action="set_boundary",
            span=[8, 13],
        )
        touched = {item["annotation"] for item in result["impact"]}
        self.assertIn("A1:t3", touched)
        self.assertIn("A1:t1", touched)
        self.assertNotIn("A1:t0", touched)
        # 证据未变的确认保持确认。
        self.assertEqual(self.state.get("A1:t0").status, "confirmed")
        self.assertEqual(self.state.get("A1:t1").status, "confirmed")
        # 嵌套关系变化的标注重新待确认。
        self.assertEqual(self.state.get("A1:t3").status, "pending")

    def test_unchanged_confirmation_is_kept_when_evidence_is_stable(self):
        self._confirm("A1:t1", "A1:t3")
        adjudicate(self.state, self.packet, "A1:m2", action="confirm")
        self.assertEqual(self.state.get("A1:t1").status, "confirmed")
        self.assertEqual(self.state.get("A1:t3").status, "confirmed")

    def test_preview_does_not_write(self):
        self._confirm("A1:t1")
        before = self.state.get("A1:t1").status
        preview = preview_impact(self.state, self.packet, "A1:m1")
        self.assertEqual(self.state.get("A1:t1").status, before)
        self.assertTrue(preview["impact"])
        self.assertTrue(all(item["record"] == "D1" for item in preview["impact"]))

    def test_editing_other_manuscript_does_not_reopen_confirmed(self):
        import copy
        import json

        from termcorpus.records import packet_from_dict

        self._confirm("A1:t1", "A1:t3")
        evidence = self.state.get("A1:t1").evidence
        data = copy.deepcopy(json.loads(DATA.read_text(encoding="utf-8")))
        data["catalogue"][2]["raw"] = "相变存储器是材料课用语，这台冷柜不使用别的部件。"
        other = packet_from_dict(data)
        state = ReviewState.from_dict(self.state.as_dict())
        state.ensure(other)
        self.assertEqual(state.get("A1:t1").status, "confirmed")
        self.assertEqual(state.get("A1:t1").evidence, evidence)

    def test_editing_same_manuscript_reopens_on_adjudication(self):
        self._confirm("A1:t3")
        result = adjudicate(
            self.state, self.packet, "A1:m1", action="set_boundary", span=[8, 12]
        )
        touched = {item["annotation"]: item for item in result["impact"]}
        self.assertTrue(touched["A1:t3"]["evidence_changed"])

    def test_group_impact_preview_lists_samples(self):
        from termcorpus.grouping import build_grouping

        grouping = build_grouping(self.packet)
        preview = preview_group_impact(self.state, self.packet, "G1", grouping)
        self.assertEqual(
            set(preview["documents"]), {"D1", "D2", "D4", "D5"}
        )
        self.assertTrue(any(row["pinned"] for row in preview["samples"]))

    def test_same_literal_has_different_boundaries_in_other_manuscript(self):
        rows = literal_boundary_history(self.packet, "相变存储器")
        records = {row["record"] for row in rows}
        self.assertIn("D1", records)
        markings = {row["marking"] for row in rows if row["record"] == "D1"}
        self.assertEqual(markings, {"A1", "B1"})
        self.assertTrue(all(row["sentence"] for row in rows))

    def test_affected_annotation_reasons_are_explicit(self):
        affected = {item["annotation"]: item["reasons"] for item in affected_annotations(self.packet, "A1:m1")}
        self.assertIn("嵌套实体", affected["A1:m2"])
        self.assertIn("同句谓词归属", affected["A1:t1"])

    def test_reject_keeps_annotation_out_of_exports(self):
        result = adjudicate(self.state, self.packet, "A1:m2", action="reject")
        self.assertEqual(result["status"], "adjudicated")
        self.assertNotEqual(self.state.get("A1:m2").status, "confirmed")


if __name__ == "__main__":
    unittest.main()
