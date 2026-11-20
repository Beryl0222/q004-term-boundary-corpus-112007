"""核对随档资料可以读取，以及原始记录引用完整。"""

from pathlib import Path
import tempfile
import unittest

from inspect_records import load_records


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self.data = load_records(Path(__file__).resolve().parents[1] / "data" / "records.json")

    def test_record_references(self):
        for source, field, target in [('marking_a', 'record', 'catalogue'), ('marking_b', 'record', 'catalogue'), ('pin', 'record', 'catalogue')]:
            group, _, key = target.partition(".")
            keys = {row[key or "id"] for row in self.data[group]}
            for row in self.data[source]:
                refs = row.get(field)
                if refs is None:
                    continue
                refs = refs if isinstance(refs, list) else [refs]
                with self.subTest(kind=source, record=row["id"], field=field):
                    self.assertLessEqual(set(refs), keys)

    def test_duplicate_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"组":[{"id":"1"},{"id":"1"}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "编号重复"):
                load_records(path)

    def test_source_packet(self):
        for source, encoding in (("marking_a", None), ("marking_b", "utf-16-le")):
            for item in self.data[source]:
                for token, left, right, _ in item["tokens"]:
                    text = item["display"]
                    actual = text[left:right] if encoding is None else text.encode(encoding)[2 * left:2 * right].decode(encoding)
                    self.assertEqual(actual, token)
                for _, left, right, _ in item["marks"]:
                    self.assertLess(left, right)
                    self.assertLessEqual(right, len(item["display"]) if encoding is None else len(item["display"].encode(encoding)) // 2)
        known = {pin["public_id"] for pin in self.data["pin"]}
        self.assertEqual(known, set(self.data["release_manifest"][0]["checks"]))


if __name__ == "__main__":
    unittest.main()
