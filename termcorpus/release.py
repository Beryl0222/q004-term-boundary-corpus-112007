"""发布隔离、标签导出与交付影响。

校验 pin 的条目编号和内容属于固定校验集：训练集不得包含同组材料，也不能
靠重排或替换校验样本来"解决"泄漏。确认后的标注才可导出；导出的偏移口径
是下游使用的 UTF-8 字节区间，区间可有嵌套、禁止交叉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from .context import annotation_context
from .grouping import Grouping, build_grouping, leak_evidence
from .records import Packet
from .review import ReviewState
from .unicode_align import char_span_to_utf8

EXPORT_VERSION = 1


class ReleaseError(ValueError):
    """发布请求违反校验隔离或导出规则。"""


@dataclass
class LeakFinding:
    document: str
    pin: str
    check_record: str
    path: list[dict]
    basis: list[str]
    other_pins: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "document": self.document,
            "pin": self.pin,
            "check_record": self.check_record,
            "basis": list(self.basis),
            "path": list(self.path),
            "other_pins": list(self.other_pins),
        }


@dataclass
class ReleasePlan:
    requested: list[str]
    training: list[str]
    checks: list[str] = field(default_factory=list)
    leaked: list[LeakFinding] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    missing_coverage: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self):
        return not self.leaked

    def as_dict(self):
        return {
            "ok": self.ok,
            "requested": list(self.requested),
            "training": list(self.training),
            "checks": list(self.checks),
            "removed": list(self.removed),
            "excluded": list(self.excluded),
            "leaked": [item.as_dict() for item in self.leaked],
            "missing_coverage": list(self.missing_coverage),
            "warnings": list(self.warnings),
        }


def _pins_for_group(grouping, packet, group_id):
    return sorted(
        pin.public_id for pin in packet.pins.values() if grouping.document_to_group.get(pin.record) == group_id
    )


def plan_release(
    packet: Packet,
    training: list[str],
    grouping: Grouping | None = None,
    required_checks: list[str] | None = None,
    previous_training: list[str] | None = None,
):
    """计算一次发布：清理泄漏材料，保留固定校验集编号，说明缺失覆盖。

    不重排、不替换校验样本。如果清理后某个领域没有训练材料，把缺失的
    领域及其原本覆盖的材料列给负责人决定补什么资料。
    """
    grouping = grouping or build_grouping(packet)
    known = set(packet.documents)
    unknown = [doc for doc in training if doc not in known]
    if unknown:
        raise ReleaseError(f"训练清单包含不存在的文献 {sorted(unknown)}")
    checks = list(required_checks) if required_checks is not None else [
        pin.public_id for pin in packet.pins.values()
    ]
    known_checks = {pin.public_id for pin in packet.pins.values()}
    if set(checks) != known_checks or len(checks) != len(known_checks):
        raise ReleaseError("校验编号必须与已有 pin 完全一致，不能重排、增删或替换")
    if required_checks is not None and list(required_checks) != [
        pin.public_id for pin in packet.pins.values()
    ]:
        raise ReleaseError("不能改变已有固定校验集的条目编号或顺序")
    check_records = {
        pin.record for pin in packet.pins.values() if pin.public_id in checks
    }

    requested = list(dict.fromkeys(training))
    pinned_groups = {
        grouping.document_to_group[record]
        for record in check_records
        if record in grouping.document_to_group
    }
    leaked = []
    kept = []
    for doc_id in requested:
        group_id = grouping.document_to_group.get(doc_id)
        if doc_id in check_records:
            leaked.append(
                LeakFinding(
                    document=doc_id,
                    pin=next(
                        pin.public_id
                        for pin in packet.pins.values()
                        if pin.record == doc_id
                    ),
                    check_record=doc_id,
                    path=[],
                    basis=["训练材料本身是固定校验集记录"],
                )
            )
            continue
        if group_id in pinned_groups:
            findings = leak_evidence(grouping, [doc_id], packet)
            group_pins = _pins_for_group(grouping, packet, group_id)
            if len(group_pins) > 1:
                # 拼接引用把一篇材料连到多个校验组：每份依据都列出其余
                # 被牵连的 pin，负责人要能一次看到全部影响。
                extra = [pin for pin in group_pins]
            else:
                extra = []
            for finding in findings:
                leaked.append(
                    LeakFinding(
                        document=finding["document"],
                        pin=finding["pin"],
                        check_record=finding["check_record"],
                        path=finding["path"],
                        basis=finding.get("basis", []),
                        other_pins=[
                            pin for pin in extra if pin != finding["pin"]
                        ],
                    )
                )
            if not findings:
                leaked.append(
                    LeakFinding(
                        document=doc_id,
                        pin=", ".join(sorted(group_pins)),
                        check_record=", ".join(
                            sorted(
                                {
                                    packet.pins[pin_id].record
                                    for pin_id in group_pins
                                }
                            )
                        ),
                        path=[],
                        basis=[f"与固定校验集同属 {group_id}，但未找到可引用的证据路径"],
                        other_pins=sorted(group_pins),
                    )
                )
            continue
        kept.append(doc_id)

    removed = [doc for doc in requested if doc not in kept]
    plan = ReleasePlan(
        requested=requested,
        training=kept,
        checks=checks,
        leaked=leaked,
        removed=removed,
        excluded=[doc for doc in requested if doc in check_records],
    )

    # 说明清理造成的领域覆盖变化：哪些请求中的领域完全失去材料、哪些
    # 只是被削减。一个材料若因拼接引用牵连了多个校验组，泄漏依据会一并
    # 列在 missing_coverage 的 basis/pins 中。
    area_by_doc = {doc: packet.documents[doc].area for doc in packet.documents}
    kept_areas = {area_by_doc[doc] for doc in kept}
    baseline = list(requested) if not previous_training else list(previous_training)
    baseline_areas = {area_by_doc[doc] for doc in baseline if doc in area_by_doc}
    for area in sorted(baseline_areas):
        area_docs = sorted(doc for doc, value in area_by_doc.items() if value == area)
        area_requested = [doc for doc in requested if area_by_doc[doc] == area]
        area_kept = [doc for doc in kept if area_by_doc[doc] == area]
        if area in kept_areas and len(area_kept) >= len(area_requested):
            continue
        leaked_here = [item for item in plan.leaked if item.document in area_requested]
        by_pin = {}
        for item in leaked_here:
            entry = by_pin.setdefault(item.pin, {"documents": [], "basis": []})
            if item.document not in entry["documents"]:
                entry["documents"].append(item.document)
            for basis in item.basis:
                if basis not in entry["basis"]:
                    entry["basis"].append(basis)
        # 同一篇材料牵连的其余 pin 只作为"一并影响"提示，其具体依据
        # 由那篇材料自己的 finding 列出，避免把一条路径的理由张冠李戴。
        chained = {}
        for item in leaked_here:
            for other in item.other_pins:
                if other and other != item.pin:
                    chained.setdefault(other, set()).add(item.document)
        for pin, documents in chained.items():
            entry = by_pin.setdefault(pin, {"documents": [], "basis": []})
            for document in sorted(documents):
                if document not in entry["documents"]:
                    entry["documents"].append(document)
            note = "同一篇材料因拼接引用与该校验编号相连，一并受影响"
            if note not in entry["basis"]:
                entry["basis"].append(note)
        # 同一领域里还能补的材料：包内未被请求、且不在任何校验组内。
        pinned_groups_in_packet = {
            grouping.document_to_group[pin.record]
            for pin in packet.pins.values()
            if pin.record in grouping.document_to_group
        }
        alternatives = sorted(
            doc
            for doc, value in area_by_doc.items()
            if value == area
            and doc not in requested
            and grouping.document_to_group.get(doc) not in pinned_groups_in_packet
        )
        plan.missing_coverage.append(
            {
                "area": area,
                "documents_in_packet": area_docs,
                "requested": area_requested,
                "kept": area_kept,
                "leaked": [doc for doc in area_requested if doc not in kept],
                "pins": sorted(by_pin),
                "leak_by_pin": [
                    {
                        "pin": pin,
                        "documents": by_pin[pin]["documents"],
                        "basis": by_pin[pin]["basis"],
                    }
                    for pin in sorted(by_pin)
                ],
                "alternatives_in_packet": alternatives,
                "status": "missing" if not area_kept else "reduced",
                "reason": (
                    f"材料与固定校验集 {', '.join(sorted(by_pin))} 同组，"
                    "清理后该领域没有训练材料"
                    if by_pin and not area_kept
                    else (
                        f"该领域清理后只剩 {', '.join(area_kept)}，"
                        f"失去 {', '.join(sorted(by_pin))} 对应材料的覆盖"
                        if by_pin
                        else "该领域原有材料全部不可用"
                    )
                ),
                "request": (
                    f"请负责人决定是否为「{area}」补充训练材料"
                    + (
                        f"；包内可用候选：{', '.join(alternatives)}"
                        if alternatives
                        else "；包内没有可补材料，需要另行收集"
                    )
                ),
            }
        )

    if previous_training is not None:
        if [doc for doc in previous_training if doc in removed]:
            plan.warnings.append(
                "上一版训练清单中有材料因泄漏被清理："
                + ", ".join(doc for doc in previous_training if doc in removed)
            )
    return plan


@dataclass
class Impact:
    releases: list[dict]
    message: str

    def as_dict(self):
        return {"releases": list(self.releases), "message": self.message}


def delivered_releases(packet: Packet):
    return sorted(packet.releases.values(), key=lambda item: (item.date, item.id))


def leak_impact(packet: Packet, doc_ids: list[str], grouping: Grouping | None = None):
    """某个候选训练材料被确认泄漏答案后，对已交付版本的影响。"""
    grouping = grouping or build_grouping(packet)
    findings = leak_evidence(grouping, doc_ids, packet)
    affected = []
    for release in delivered_releases(packet):
        hit_docs = sorted({item["document"] for item in findings if item["document"] in release.training})
        if not hit_docs:
            continue
        pins = sorted({item["pin"] for item in findings if item["document"] in release.training})
        affected.append(
            {
                "release": release.id,
                "date": release.date,
                "training_hit": hit_docs,
                "pins": pins,
                "notify": True,
            }
        )
    message = (
        "需要通知使用方并发布修正版："
        + ", ".join(f"{row['release']}（{row['date']}）" for row in affected)
        if affected
        else "没有已交付版本包含这些材料，无需通知使用方"
    )
    return Impact(affected, message)


@dataclass
class ExportResult:
    samples: list[dict]
    skipped: list[dict]
    problems: list[str]

    def as_dict(self):
        return {
            "version": EXPORT_VERSION,
            "samples": list(self.samples),
            "skipped": list(self.skipped),
            "problems": list(self.problems),
        }


def _downstream_spans(packet, annotations):
    """把原文字符区间变为下游 UTF-8 字节区间，检查嵌套无交叉。"""
    spans = []
    raw = None
    for annotation in annotations:
        text = packet.documents[annotation.record].raw
        if raw is None:
            raw = text
        elif raw != text:
            return None, ["同一样本包含多份原稿，无法共用一套字节偏移"]
        start, end = annotation.raw_span
        byte_start, byte_end = char_span_to_utf8(text, start, end)
        spans.append((byte_start, byte_end, annotation))
    problems = []
    for index, (left_start, left_end, left) in enumerate(spans):
        for right_start, right_end, right in spans[index + 1 :]:
            if left_start < right_start < left_end < right_end or (
                right_start < left_start < right_end < left_end
            ):
                problems.append(
                    f"标注 {left.id} 与 {right.id} 的字节区间交叉，下游不允许"
                )
    return spans, problems


def export_labeled(
    packet: Packet,
    state: ReviewState,
    doc_ids: list[str] | None = None,
    grouping: Grouping | None = None,
    include_pinned: bool = False,
):
    """只导出已确认标注。默认不导出固定校验集所在分组。"""
    grouping = grouping or build_grouping(packet)
    pinned_groups = {
        grouping.document_to_group[pin.record]
        for pin in packet.pins.values()
        if pin.record in grouping.document_to_group
    }
    selected = doc_ids if doc_ids is not None else sorted(packet.documents)
    samples = []
    skipped = []
    for doc_id in selected:
        if doc_id not in packet.documents:
            skipped.append({"document": doc_id, "reason": "没有该文献"})
            continue
        group_id = grouping.document_to_group.get(doc_id)
        if not include_pinned and group_id in pinned_groups:
            skipped.append(
                {
                    "document": doc_id,
                    "reason": "与固定校验集同组，不导出",
                    "group": group_id,
                }
            )
            continue
        marking_annotations = []
        for marking in packet.marking_for(doc_id):
            for annotation in marking.annotations:
                record = state.records.get(annotation.id)
                if record is None or record.status != "confirmed":
                    continue
                marking_annotations.append((marking, annotation))
        confirmed_ids = {annotation.id for _, annotation in marking_annotations}
        if not marking_annotations:
            skipped.append({"document": doc_id, "reason": "没有已确认标注"})
            continue
        raw = packet.documents[doc_id].raw
        # 下游禁止交叉区间：按起点排序后贪心保留，遇到交叉的标注改为
        # 待校对并在导出结果中列出，而不是整段静默丢弃。
        ordered = sorted(
            marking_annotations,
            key=lambda pair: (
                pair[1].raw_span[0] if pair[1].raw_span else 0,
                -(pair[1].raw_span[1] if pair[1].raw_span else 0),
            ),
        )
        kept = []
        dropped = []
        kept_spans = []
        for marking, annotation in ordered:
            if annotation.raw_span is None:
                dropped.append((annotation, "没有原样文本位置"))
                continue
            start, end = annotation.raw_span
            if any(
                start < kept_end and kept_start < end and not (
                    (kept_start <= start and end <= kept_end)
                    or (start <= kept_start and kept_end <= end)
                )
                for kept_start, kept_end in kept_spans
            ):
                dropped.append((annotation, "与另一标注交叉，下游不允许"))
                continue
            kept.append((marking, annotation))
            kept_spans.append((start, end))
        kept_annotations = [annotation for _, annotation in kept]
        spans, problems = _downstream_spans(packet, kept_annotations)
        if problems:
            skipped.append({"document": doc_id, "reason": "字节区间校验失败", "problems": problems})
            continue
        byte_by_id = {
            annotation.id: (start, end) for start, end, annotation in spans
        }
        sample = {
            "document": doc_id,
            "origin": packet.documents[doc_id].origin,
            "area": packet.documents[doc_id].area,
            "text": raw,
            "text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "offset_unit": "utf-8-byte",
            "nested_allowed": True,
            "crossing_allowed": False,
            "sources": [],
            "dropped": [
                {"annotation": annotation.id, "reason": reason}
                for annotation, reason in dropped
            ],
        }
        by_source = {}
        for marking, annotation in kept:
            entry = by_source.setdefault(
                marking.source,
                {
                    "marking": marking.id,
                    "offset_unit": marking.offset_unit,
                    "annotations": [],
                },
            )
            context = annotation_context(packet, annotation)
            start, end = byte_by_id[annotation.id]
            entry["annotations"].append(
                {
                    "id": annotation.id,
                    "kind": annotation.kind,
                    "byte_span": [start, end],
                    "raw_char_span": list(annotation.raw_span),
                    "literal": annotation.literal,
                    "pos": annotation.pos,
                    "label": annotation.label,
                    "snapshot": annotation.literal,
                    "sense": context.sense,
                    "context_label": context.label,
                    "provenance": {
                        "source": marking.source,
                        "offset_unit": marking.offset_unit,
                        "source_span": list(annotation.source_span),
                        "display_span": list(annotation.display_span),
                        "record": annotation.record,
                    },
                }
            )
        sample["sources"] = list(by_source.values())
        samples.append(sample)
    return ExportResult(samples, skipped, [])


def release_record(
    packet: Packet,
    plan: ReleasePlan,
    state: ReviewState,
    release_id: str,
    date: str,
    grouping: Grouping | None = None,
):
    """生成可写入发布清单的记录，保留已有校验编号。"""
    grouping = grouping or build_grouping(packet)
    exported = export_labeled(packet, state, doc_ids=plan.training, grouping=grouping)
    return {
        "id": release_id,
        "date": date,
        "training": list(plan.training),
        "checks": list(plan.checks),
        "samples": exported.samples,
        "skipped": exported.skipped,
        "leaks": [item.as_dict() for item in plan.leaked],
        "removed": list(plan.removed),
        "missing_coverage": list(plan.missing_coverage),
    }


def audit_provenance(packet: Packet, grouping: Grouping | None = None):
    """发布记录中每段标注的来由：来源、偏移口径、对齐差异。"""
    grouping = grouping or build_grouping(packet)
    rows = []
    for annotation in packet.annotations.values():
        marking = next(
            item for item in packet.markings if item.id == annotation.marking_id
        )
        group_id = grouping.document_to_group.get(annotation.record)
        pinned_records = [pin.record for pin in packet.pins.values()]
        pinned_groups = {
            grouping.document_to_group.get(record)
            for record in pinned_records
            if record in grouping.document_to_group
        }
        rows.append(
            {
                "annotation": annotation.id,
                "record": annotation.record,
                "origin": packet.documents[annotation.record].origin,
                "source": marking.source,
                "offset_unit": marking.offset_unit,
                "source_span": list(annotation.source_span),
                "display_span": list(annotation.display_span),
                "raw_span": list(annotation.raw_span) if annotation.raw_span else None,
                "literal": annotation.literal,
                "alignment_notes": list(annotation.alignment_notes),
                "problems": list(annotation.problems),
                "group": group_id,
                "pinned": annotation.record in pinned_records,
                "pinned_group": group_id in pinned_groups,
            }
        )
    return rows
