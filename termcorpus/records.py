"""读取资料包：文献、问答、两家分词与带位置的标注、校验 pin 与发布记录。

所有进入工具的区间先转换到 Unicode 字符区间（Python 字符串下标），同时保留
来源声明的位置口径。下游需要的 UTF-8 字节区间在导出阶段再计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import unicodedata

from .unicode_align import (
    DisplayRawAlignment,
    OffsetError,
    align_display_to_raw,
    canonical_text,
    convert_span,
)


class PacketError(ValueError):
    """资料包不满足约定。"""


SOURCE_UNITS = {
    "marking_a": "unicode_char",
    "marking_b": "utf16",
}

def _is_content(char: str) -> bool:
    return unicodedata.category(char)[0] in {"L", "N", "M"}


def content_text(text: str) -> str:
    """只保留文字、数字与组合标记，用于比较内容而不是排版。

    以组合字符簇为单位做 NFKC，`e`+`◌́` 与 `é` 得到同一个内容串。
    """
    from .unicode_align import clusters

    return "".join(
        char
        for start, end in clusters(text)
        for char in unicodedata.normalize("NFKC", text[start:end])
        if _is_content(char)
    )


def find_literal(text: str, literal: str) -> list[tuple[int, int]]:
    """在 text 中查找字面，容许排版差异，返回原文字符区间。"""
    needle = content_text(literal)
    if not needle:
        return []
    # 以组合字符簇为单位做规范合成，再把内容字符映射回原文字符位置。
    # 逐字符合成会把 e + ◌́ 留在两个簇里，找不到 Café。
    from .unicode_align import clusters

    normalized = []
    positions = []
    for start, end in clusters(text):
        for piece in unicodedata.normalize("NFKC", text[start:end]):
            if _is_content(piece):
                normalized.append(piece)
                positions.append(start)
                positions.append(end)
    haystack = "".join(normalized)
    spans = []
    cursor = haystack.find(needle)
    while cursor != -1:
        first = cursor
        last = cursor + len(needle) - 1
        spans.append((positions[2 * first], positions[2 * last + 1]))
        cursor = haystack.find(needle, cursor + 1)
    return _dedupe_spans(spans)


def _dedupe_spans(spans):
    result = []
    for span in spans:
        if span not in result:
            result.append(span)
    return result


@dataclass(frozen=True)
class Document:
    id: str
    origin: str
    area: str
    raw: str
    mentions: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.raw


@dataclass(frozen=True)
class QAPair:
    id: str
    question: str
    answer: str
    record: str

    @property
    def text(self) -> str:
        return f"{self.question}\n{self.answer}"


@dataclass(frozen=True)
class Pin:
    id: str
    public_id: str
    record: str


@dataclass(frozen=True)
class ReleaseManifest:
    id: str
    date: str
    training: tuple[str, ...]
    checks: tuple[str, ...]

    def as_dict(self):
        return {
            "id": self.id,
            "date": self.date,
            "training": list(self.training),
            "checks": list(self.checks),
        }


@dataclass
class Annotation:
    id: str
    marking_id: str
    source: str
    record: str
    kind: str  # token / entity
    literal: str
    display_span: tuple[int, int]  # 已转换为 Unicode 字符区间
    source_span: tuple[int, int]  # 合作方声明的原位置口径
    raw_span: tuple[int, int] | None
    pos: str | None = None
    label: str | None = None
    alignment_notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def display_literal(self) -> str:
        return self.literal

    def summary(self):
        return {
            "id": self.id,
            "marking": self.marking_id,
            "source": self.source,
            "record": self.record,
            "kind": self.kind,
            "literal": self.literal,
            "pos": self.pos,
            "label": self.label,
            "display_span": list(self.display_span),
            "source_span": list(self.source_span),
            "raw_span": list(self.raw_span) if self.raw_span else None,
            "alignment_notes": list(self.alignment_notes),
            "problems": list(self.problems),
        }


def _is_boundary_punctuation(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def _check_entity_projection(marking, annotation, raw_text):
    """实体投影的边界问题：标点被吞、重复片段无法唯一定位。"""
    literal = annotation.literal
    stripped = literal.strip()
    if literal != stripped:
        annotation.problems.append("实体边界含空白，投影需校对")
    if stripped and (
        _is_boundary_punctuation(stripped[0]) or _is_boundary_punctuation(stripped[-1])
    ):
        annotation.problems.append("实体边界含标点，投影可能只覆盖实体一部分")
    if annotation.raw_span is None:
        return
    trimmed = "".join(
        char
        for char in stripped
        if not _is_boundary_punctuation(char)
    )
    if not trimmed:
        return
    found = find_literal(raw_text, trimmed)
    if len(found) > 1:
        annotation.problems.append(
            f"重复片段“{trimmed}”出现 {len(found)} 次，当前投影无法唯一定位"
        )
    elif len(found) == 1 and annotation.raw_span != found[0]:
        if not (
            annotation.raw_span[0] <= found[0][0]
            and found[0][1] <= annotation.raw_span[1]
        ):
            annotation.problems.append("实体投影与字面出现位置不完全一致")


@dataclass
class Relation:
    predicate: str
    mark_ref: str
    target_id: str | None
    predicate_span: tuple[int, int] | None
    occurrences: int
    notes: list[str] = field(default_factory=list)

    def summary(self):
        return {
            "predicate": self.predicate,
            "mark_ref": self.mark_ref,
            "target": self.target_id,
            "predicate_span": list(self.predicate_span) if self.predicate_span else None,
            "occurrences": self.occurrences,
            "notes": list(self.notes),
        }


@dataclass
class Marking:
    id: str
    source: str
    record: str
    display: str
    offset_unit: str
    annotations: list[Annotation] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    alignment: DisplayRawAlignment | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def tokens(self):
        return [item for item in self.annotations if item.kind == "token"]

    @property
    def entities(self):
        return [item for item in self.annotations if item.kind == "entity"]

    def annotation(self, annotation_id):
        for item in self.annotations:
            if item.id == annotation_id:
                return item
        raise KeyError(annotation_id)


@dataclass
class Packet:
    records: dict
    documents: dict[str, Document]
    qa: dict[str, QAPair]
    markings: list[Marking]
    pins: dict[str, Pin]
    releases: dict[str, ReleaseManifest]
    fingerprint: str
    path: str | None = None

    @property
    def annotations(self):
        return {
            annotation.id: annotation
            for marking in self.markings
            for annotation in marking.annotations
        }

    def marking_for(self, record: str):
        return [marking for marking in self.markings if marking.record == record]


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not data:
        raise PacketError("资料应包含非空的记录类别")
    for kind, rows in data.items():
        if not isinstance(rows, list):
            raise PacketError(f"{kind} 应为记录数组")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise PacketError(f"{kind} 中存在没有编号的记录")
            if row["id"] in ids:
                raise PacketError(f"{kind} 中编号重复 {row['id']}")
            ids.add(row["id"])
    return data


def _require(groups, kind):
    if kind not in groups:
        raise PacketError(f"缺少记录类别 {kind}")
    return groups[kind]


def _build_packet(data, path=None) -> Packet:
    documents = {}
    for row in _require(data, "catalogue"):
        for key in ("origin", "area", "raw"):
            if not isinstance(row.get(key), str):
                raise PacketError(f"catalogue {row['id']} 缺少 {key}")
        mentions = row.get("mentions", [])
        if not isinstance(mentions, list) or not all(isinstance(item, str) for item in mentions):
            raise PacketError(f"catalogue {row['id']} 的 mentions 应为编号数组")
        documents[row["id"]] = Document(
            id=row["id"],
            origin=row["origin"],
            area=row["area"],
            raw=row["raw"],
            mentions=tuple(mentions),
        )
    for document in documents.values():
        unknown = set(document.mentions) - set(documents)
        if unknown:
            raise PacketError(f"{document.id} 引用了不存在的文献 {sorted(unknown)}")

    qa = {}
    for row in _require(data, "Q"):
        record = row.get("record")
        if record not in documents:
            raise PacketError(f"问答 {row['id']} 引用了不存在的文献 {record}")
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise PacketError(f"问答 {row['id']} 缺少 question/answer")
        qa[row["id"]] = QAPair(row["id"], row["question"], row["answer"], record)

    pins = {}
    public_ids = set()
    for row in _require(data, "pin"):
        record = row.get("record")
        if record not in documents:
            raise PacketError(f"pin {row['id']} 引用了不存在的文献 {record}")
        public_id = row.get("public_id")
        if not isinstance(public_id, str) or not public_id:
            raise PacketError(f"pin {row['id']} 缺少 public_id")
        if public_id in public_ids:
            raise PacketError(f"校验编号重复 {public_id}")
        public_ids.add(public_id)
        pins[row["id"]] = Pin(row["id"], public_id, record)

    releases = {}
    for row in _require(data, "release_manifest"):
        training = tuple(row.get("training", []))
        checks = tuple(row.get("checks", []))
        if not set(training) <= set(documents):
            raise PacketError(f"发布 {row['id']} 的训练编号不完整")
        if not set(checks) <= public_ids:
            raise PacketError(f"发布 {row['id']} 的校验编号不是已有 pin")
        releases[row["id"]] = ReleaseManifest(row["id"], str(row.get("date", "")), training, checks)

    markings = []
    for source in ("marking_a", "marking_b"):
        if source not in data:
            continue
        unit = SOURCE_UNITS[source]
        for row in data[source]:
            record = row.get("record")
            if record not in documents:
                raise PacketError(f"{source} {row['id']} 引用了不存在的文献 {record}")
            display = row.get("display")
            if not isinstance(display, str):
                raise PacketError(f"{source} {row['id']} 缺少 display")
            marking = Marking(
                id=row["id"],
                source=source,
                record=record,
                display=display,
                offset_unit=unit,
            )
            raw_text = documents[record].raw
            alignment = align_display_to_raw(display, raw_text)
            marking.alignment = alignment
            if alignment.unmatched_display or alignment.unmatched_raw:
                marking.problems.append(
                    "展示文本与原样文本存在无法按排版差异消除的差异"
                )
            if canonical_text(display) != canonical_text(raw_text):
                marking.problems.append("展示文本与原样文本内容不一致")

            tokens = row.get("tokens", [])
            marks = row.get("marks", [])
            if not isinstance(tokens, list) or not isinstance(marks, list):
                raise PacketError(f"{source} {row['id']} 的 tokens/marks 应为数组")

            for index, token in enumerate(tokens):
                if not isinstance(token, list) or len(token) != 4:
                    raise PacketError(f"{source} {row['id']} 第 {index} 个 token 格式错误")
                literal, start, end, pos = token
                annotation = _make_annotation(
                    marking=marking,
                    annotation_id=f"{marking.id}:t{index}",
                    kind="token",
                    literal=literal,
                    start=start,
                    end=end,
                    pos=pos,
                    label=None,
                    raw_text=raw_text,
                )
                marking.annotations.append(annotation)

            for index, mark in enumerate(marks):
                if not isinstance(mark, list) or len(mark) != 4:
                    raise PacketError(f"{source} {row['id']} 第 {index} 个 mark 格式错误")
                mark_id, start, end, label = mark
                annotation = _make_annotation(
                    marking=marking,
                    annotation_id=f"{marking.id}:{mark_id}",
                    kind="entity",
                    literal=None,
                    start=start,
                    end=end,
                    pos=None,
                    label=label,
                    raw_text=raw_text,
                )
                _check_entity_projection(marking, annotation, raw_text)
                marking.annotations.append(annotation)

            by_id = {item.id.split(":", 1)[1]: item for item in marking.entities}
            for relation_row in row.get("relations", []):
                if not isinstance(relation_row, list) or len(relation_row) != 2:
                    raise PacketError(f"{source} {row['id']} relation 格式错误")
                predicate, mark_ref = relation_row
                target = by_id.get(mark_ref)
                if target is None:
                    marking.problems.append(f"关系引用了不存在的标注 {mark_ref}")
                    continue
                occurrences = find_literal(raw_text, predicate)
                span = occurrences[0] if len(occurrences) == 1 else None
                notes = []
                if len(occurrences) > 1:
                    notes.append("同句谓词字面出现多次，无法唯一定位")
                elif not occurrences:
                    notes.append("谓词字面未在原稿中出现")
                if not any(
                    item.kind == "token" and item.literal == predicate for item in marking.tokens
                ):
                    notes.append("谓词未在本方分词中，归属需校对确认")
                marking.relations.append(
                    Relation(
                        predicate=predicate,
                        mark_ref=mark_ref,
                        target_id=target.id,
                        predicate_span=span,
                        occurrences=len(occurrences),
                        notes=notes,
                    )
                )
            markings.append(marking)

    fingerprint = hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return Packet(
        records=data,
        documents=documents,
        qa=qa,
        markings=markings,
        pins=pins,
        releases=releases,
        fingerprint=fingerprint,
        path=str(path) if path else None,
    )


def _make_annotation(
    marking: Marking,
    annotation_id: str,
    kind: str,
    literal,
    start,
    end,
    pos,
    label,
    raw_text=None,
):
    note = []
    if literal is not None and not isinstance(literal, str):
        marking.problems.append(f"{annotation_id} 的字面不是字符串")
        literal = ""
    if not isinstance(start, int) or not isinstance(end, int):
        marking.problems.append(f"{annotation_id} 的位置不是整数")
        start, end = 0, 0
    try:
        char_start, char_end = convert_span(marking.display, start, end, marking.offset_unit)
    except OffsetError as error:
        marking.problems.append(f"{annotation_id}: {error}")
        return Annotation(
            id=annotation_id,
            marking_id=marking.id,
            source=marking.source,
            record=marking.record,
            kind=kind,
            literal=literal or "",
            display_span=(start, end),
            source_span=(start, end),
            raw_span=None,
            pos=pos,
            label=label,
            problems=[str(error)],
        )
    if literal is None:
        literal = marking.display[char_start:char_end]
    actual = marking.display[char_start:char_end]
    if actual != literal:
        note.append(f"来源区间切出 {actual!r} 与字面 {literal!r} 不一致")
    mapping = marking.alignment.map_span(char_start, char_end)
    raw_start, raw_end, ambiguous, mapping_notes = mapping
    note.extend(mapping_notes)
    # 原样文本上的区间应切出与字面内容一致的字串；不一致说明标注本身
    # 或展示文本有问题，需要校对。
    if raw_text is not None and content_text(raw_text[raw_start:raw_end]) != content_text(literal):
        note.append("映射到原样文本后与字面内容不一致")
    return Annotation(
        id=annotation_id,
        marking_id=marking.id,
        source=marking.source,
        record=marking.record,
        kind=kind,
        literal=literal,
        display_span=(char_start, char_end),
        source_span=(start, end),
        raw_span=(raw_start, raw_end),
        pos=pos,
        label=label,
        alignment_notes=mapping_notes,
        problems=note,
    )


def load_packet(path) -> Packet:
    path = Path(path)
    data = _read_json(path)
    return _build_packet(data, path)


def packet_from_dict(data, path=None) -> Packet:
    return _build_packet(data, path)


def packet_annotation(packet: Packet, annotation_id: str) -> Annotation:
    try:
        return packet.annotations[annotation_id]
    except KeyError as error:
        raise KeyError(f"找不到标注 {annotation_id}") from error


def validate_pins(packet: Packet):
    """检查 pin 仍指向可读记录，且发布清单没有增删或重排校验编号。"""
    problems = []
    for release in packet.releases.values():
        if len(set(release.checks)) != len(release.checks):
            problems.append(f"发布 {release.id} 的校验编号重复")
        missing = [item for item in release.checks if item not in {
            pin.public_id for pin in packet.pins.values()
        }]
        if missing:
            problems.append(
                f"发布 {release.id} 引用了已不存在的校验编号 {missing}"
            )
    return problems
