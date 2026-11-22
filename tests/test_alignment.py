"""核对位置口径、排版差异对齐与原始字符位置保留。"""

from pathlib import Path
import unicodedata
import unittest

from termcorpus.records import load_packet, packet_from_dict
from termcorpus.unicode_align import (
    OffsetError,
    align_display_to_raw,
    canonical_text,
    char_span_to_utf8,
    convert_span,
    utf16_length,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "records.json"


class OffsetTest(unittest.TestCase):
    def test_source_b_offsets_are_utf16_units(self):
        packet = load_packet(DATA)
        marking = next(item for item in packet.markings if item.id == "B2")
        token = next(item for item in marking.annotations if item.literal == "B-2")
        self.assertEqual(token.source_span, (10, 13))
        # 展示文本第 2 个 UTF-16 单元开始于 emoji 之后。
        self.assertEqual(token.display_span, (9, 12))

    def test_utf16_surrogate_pair_cannot_split(self):
        with self.assertRaises(OffsetError):
            convert_span("🎛Café", 1, 3, "utf16")
        self.assertEqual(utf16_length("🎛Café"), 6)

    def test_utf8_bytes_for_downstream(self):
        text = "芯片✅使用Ａ-1相变存储器。"
        self.assertEqual(char_span_to_utf8(text, 0, 2), (0, 6))
        self.assertEqual(text.encode("utf-8")[0:6].decode("utf-8"), "芯片")

    def test_combining_sequence_is_one_cluster(self):
        composed = unicodedata.normalize("NFC", "Café")
        self.assertNotEqual(composed, "Café")
        alignment = align_display_to_raw("Café", composed)
        self.assertEqual(alignment.unmatched_display, [])
        self.assertEqual(alignment.unmatched_raw, [])
        self.assertEqual(canonical_text("Café"), canonical_text(composed))


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.packet = load_packet(DATA)

    def test_fullwidth_model_aligns_without_losing_positions(self):
        marking = next(item for item in self.packet.markings if item.id == "A1")
        token = next(item for item in marking.annotations if item.literal == "Ａ-1")
        self.assertEqual(token.raw_span, (5, 8))
        self.assertEqual(marking.alignment.unmatched_display, [])
        self.assertEqual(marking.alignment.unmatched_raw, [])

    def test_line_break_and_spacing_are_layout_only(self):
        alignment = align_display_to_raw("问：答案\n答：是。", "问：答案 答：是。")
        self.assertEqual(alignment.unmatched_display, [])
        self.assertEqual(alignment.unmatched_raw, [])

    def test_semantic_difference_is_not_silently_aligned(self):
        alignment = align_display_to_raw("使用存储器", "使用储存器")
        self.assertTrue(alignment.unmatched_display or alignment.unmatched_raw)

    def test_barrier_in_entity_projection_is_flagged(self):
        packet = self.packet
        marking = next(item for item in packet.markings if item.id == "B2")
        entity = next(item for item in marking.annotations if item.kind == "entity")
        joined = " ".join(entity.problems)
        self.assertIn("标点", joined)
        self.assertIn("无法唯一定位", joined)

    def test_repeated_span_projection_is_not_promoted(self):
        data = {
            "catalogue": [
                {"id": "D9", "origin": "重复稿", "area": "存储", "raw": "B-2 与 B-2 同型。"}
            ],
            "Q": [],
            "marking_a": [
                {
                    "id": "A9",
                    "record": "D9",
                    # 展示文本只截了一段，重复片段无法从字面唯一确定位置。
                    "display": "B-2",
                    "tokens": [],
                    "marks": [["m9", 0, 3, "设备"]],
                    "relations": [],
                }
            ],
            "marking_b": [],
            "pin": [],
            "release_manifest": [],
        }
        packet = packet_from_dict(data)
        entity = next(iter(packet.annotations.values()))
        self.assertTrue(any("无法唯一定位" in note for note in entity.problems))


if __name__ == "__main__":
    unittest.main()
