"""校对视图与局部裁定。

校对需要在不丢原文字符位置的前提下看到两套边界、跨词实体和词义差异；
裁定是局部的：改变一个实体边界或谓词归属后，只把受影响的待确认标注重新
打开。已确认且证据未变的标注不会因为整句重算而被重置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from .context import annotation_context, classify, models_in
from .records import Annotation, Marking, Packet, content_text


@dataclass
class AnnotationRecord:
    """校对状态中的一条记录。"""

    annotation_id: str
    status: str  # pending / confirmed / adjudicated
    evidence: str
    decision: str | None = None
    note: str | None = None
    span: tuple[int, int] | None = None
    label: str | None = None
    pos: str | None = None
    sense: str | None = None

    @property
    def confirmed(self):
        return self.status == "confirmed"

    def as_dict(self):
        result = {
            "annotation": self.annotation_id,
            "status": self.status,
            "evidence": self.evidence,
        }
        for key in ("decision", "note", "label", "pos", "sense"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.span is not None:
            result["span"] = list(self.span)
        return result


def _sentence_text(packet: Packet, annotation: Annotation) -> str:
    if annotation.raw_span is None:
        return ""
    raw = packet.documents[annotation.record].raw
    start, end = annotation.raw_span
    from .context import sentence_span

    left, right = sentence_span(raw, start)
    return raw[left:right]


def _enclosing(packet: Packet, marking: Marking, annotation: Annotation, overrides=None):
    overrides = overrides or {}
    span = _span_for(annotation, overrides)
    if span is None:
        return []
    start, end = span
    result = []
    for other in marking.entities:
        if other.id == annotation.id:
            continue
        other_span = _span_for(other, overrides)
        if other_span is None:
            continue
        other_start, other_end = other_span
        if other_start <= start and end <= other_end and other_span != span:
            result.append((other.id, other_span))
    return sorted(result)


def _span_for(annotation: Annotation, overrides):
    override = overrides.get(annotation.id)
    if override and override.get("span") is not None:
        return tuple(override["span"])
    return annotation.raw_span


def _literal_for(annotation: Annotation, overrides):
    override = overrides.get(annotation.id)
    if override and override.get("literal") is not None:
        return override["literal"]
    return annotation.literal


def _relations(packet: Packet, marking: Marking, annotation: Annotation, overrides=None):
    result = []
    for relation in marking.relations:
        if relation.target_id != annotation.id:
            continue
        result.append(
            {
                "predicate": relation.predicate,
                "span": relation.predicate_span,
                "occurrences": relation.occurrences,
            }
        )
    return sorted(result, key=lambda item: (item["predicate"], item["span"] or (0, 0)))


def evidence_fingerprint(packet: Packet, annotation: Annotation, overrides=None) -> str:
    """一条标注的证据指纹：内容、位置、嵌套关系与同句谓词。

    overrides 是其他已裁定标注的新边界/标签；计算嵌套关系时必须一并
    代入，否则改了一个实体的边界后，受影响标注的证据看起来"没变"。

    指纹只覆盖该标注所在的原稿，不覆盖整个资料包：别的原稿被编辑不应
    让本稿已确认的标注失效。
    """
    marking = next(
        item for item in packet.markings if item.id == annotation.marking_id
    )
    overrides = overrides or {}
    span = _span_for(annotation, overrides)
    payload = {
        "record": annotation.record,
        "record_raw_sha256": hashlib.sha256(
            packet.documents[annotation.record].raw.encode("utf-8")
        ).hexdigest(),
        "source": annotation.source,
        "kind": annotation.kind,
        "literal": _literal_for(annotation, overrides),
        "span": span,
        "pos": annotation.pos,
        "label": annotation.label,
        "sentence": _sentence_text(packet, annotation),
        "enclosing": _enclosing(packet, marking, annotation, overrides),
        "relations": _relations(packet, marking, annotation, overrides),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class ReviewState:
    records: dict[str, AnnotationRecord] = field(default_factory=dict)
    adjudications: list[dict] = field(default_factory=list)

    def ensure(self, packet: Packet):
        """为所有标注建立缺省 pending 状态，保留已有确认。"""
        for annotation in packet.annotations.values():
            fingerprint = evidence_fingerprint(packet, annotation)
            existing = self.records.get(annotation.id)
            if existing is None:
                self.records[annotation.id] = AnnotationRecord(
                    annotation_id=annotation.id,
                    status="pending",
                    evidence=fingerprint,
                )

    def get(self, annotation_id):
        return self.records.get(annotation_id)

    def as_dict(self):
        return {
            "records": [
                record.as_dict()
                for record in sorted(
                    self.records.values(), key=lambda item: item.annotation_id
                )
            ],
            "adjudications": list(self.adjudications),
        }

    @classmethod
    def from_dict(cls, data):
        state = cls()
        for row in data.get("records", []):
            span = tuple(row["span"]) if row.get("span") else None
            state.records[row["annotation"]] = AnnotationRecord(
                annotation_id=row["annotation"],
                status=row.get("status", "pending"),
                evidence=row.get("evidence", ""),
                decision=row.get("decision"),
                note=row.get("note"),
                span=span,
                label=row.get("label"),
                pos=row.get("pos"),
                sense=row.get("sense"),
            )
        state.adjudications = list(data.get("adjudications", []))
        return state


def _same_clause(packet, left: Annotation, right: Annotation) -> bool:
    from .context import clause_span

    if left.raw_span is None or right.raw_span is None or left.record != right.record:
        return False
    raw = packet.documents[left.record].raw
    a = clause_span(raw, left.raw_span[0])
    b = clause_span(raw, right.raw_span[0])
    return a == b


def affected_annotations(packet: Packet, annotation_id: str, overrides=None):
    """列出局部裁定可能影响的标注及原因。

    范围是同一标记者：重叠/嵌套的实体、同小句里作为谓词的词条、以及以它
    为目标的谓词关系。跨来源的同位置标注只作比较信息，不进入状态重置。

    给出 overrides 时，按裁定后的新边界计算一次；调用方再把裁定前后的
    受影响集合取并集，这样"原本嵌套、改完不再嵌套"的标注也会被看到。
    """
    overrides = overrides or {}
    annotation = packet.annotations[annotation_id]
    marking = next(
        item for item in packet.markings if item.id == annotation.marking_id
    )
    predicate_literals = {
        relation.predicate
        for relation in marking.relations
        if relation.target_id == annotation_id
    }
    annotation_span = _span_for(annotation, overrides)
    affected = []
    for other in marking.annotations:
        if other.id == annotation_id:
            continue
        reasons = []
        other_span = _span_for(other, overrides)
        if (
            annotation_span
            and other_span
            and annotation_span[0] < other_span[1]
            and other_span[0] < annotation_span[1]
        ):
            overlap = (
                annotation_span[0] <= other_span[0]
                and other_span[1] <= annotation_span[1]
            ) or (
                other_span[0] <= annotation_span[0]
                and annotation_span[1] <= other_span[1]
            )
            reasons.append("嵌套实体" if overlap else "区间重叠")
        if other.literal in predicate_literals or (
            other.kind == "token"
            and other.pos == "动词"
            and _same_clause(packet, annotation, other)
        ):
            reasons.append("同句谓词归属")
        if reasons:
            affected.append({"annotation": other.id, "reasons": sorted(set(reasons))})
    return affected


def _apply_overrides(annotation: Annotation, override):
    if not override:
        return annotation
    return Annotation(
        id=annotation.id,
        marking_id=annotation.marking_id,
        source=annotation.source,
        record=annotation.record,
        kind=annotation.kind,
        literal=annotation.literal,
        display_span=annotation.display_span,
        source_span=annotation.source_span,
        raw_span=tuple(override["span"])
        if override.get("span") is not None
        else annotation.raw_span,
        pos=override.get("pos", annotation.pos),
        label=override.get("label", annotation.label),
        alignment_notes=list(annotation.alignment_notes),
        problems=list(annotation.problems),
    )


def evidence_after(packet: Packet, annotation: Annotation, overrides) -> str:
    """按当前裁定（含其他标注的新边界）计算证据指纹。"""
    override = overrides.get(annotation.id)
    effective = _apply_overrides(annotation, override)
    if override and override.get("span") is not None:
        # 只有真的改了边界，字面才应按新区间取原文；否则保留来源字面，
        # 免得组合字符/全角写法造成"证据变了"的假象。
        effective = _apply_boundary_literal(packet, effective)
    return evidence_fingerprint(packet, effective, overrides=overrides)


def _apply_boundary_literal(packet: Packet, annotation: Annotation):
    """改边界后的字面按原文导出，供证据与显示使用。"""
    if annotation.raw_span is None:
        return annotation
    raw = packet.documents[annotation.record].raw
    start, end = annotation.raw_span
    if raw[start:end] == annotation.literal:
        return annotation
    return Annotation(
        id=annotation.id,
        marking_id=annotation.marking_id,
        source=annotation.source,
        record=annotation.record,
        kind=annotation.kind,
        literal=raw[start:end],
        display_span=annotation.display_span,
        source_span=annotation.source_span,
        raw_span=annotation.raw_span,
        pos=annotation.pos,
        label=annotation.label,
        alignment_notes=list(annotation.alignment_notes),
        problems=list(annotation.problems),
    )


def adjudicate(
    state: ReviewState,
    packet: Packet,
    annotation_id: str,
    action: str = "confirm",
    span=None,
    label=None,
    pos=None,
    sense=None,
    note=None,
    reviewer="校对",
):
    """记录一次局部裁定，返回受影响的标注及其新旧状态。

    - 已确认的受影响标注：证据未变则保持确认；证据变化则退回 pending。
    - 未确认的受影响标注：保持 pending，但会出现在返回值里供校对查看。
    """
    if annotation_id not in packet.annotations:
        raise KeyError(f"找不到标注 {annotation_id}")
    if action not in {"confirm", "reject", "set_boundary", "set_label", "set_sense"}:
        raise ValueError(f"未知裁定动作 {action}")
    if action == "set_boundary" and span is None:
        raise ValueError("改边界需要给出新的原文字符区间")
    if action == "set_label" and not (label or pos):
        raise ValueError("改标签需要给出 label 或 pos")
    if action == "set_sense" and sense is None:
        raise ValueError("改词义需要给出 sense")
    if span is not None:
        raw = packet.documents[packet.annotations[annotation_id].record].raw
        if not 0 <= span[0] < span[1] <= len(raw):
            raise ValueError(f"新边界 {tuple(span)} 不在原文范围内")

    before = {
        annotation_id: state.records.get(annotation_id).status
        if state.records.get(annotation_id)
        else "pending"
    }
    overrides = {}
    for row in state.adjudications:
        if row.get("action") not in {"set_boundary", "set_label", "set_sense"}:
            continue
        entry = overrides.setdefault(row["annotation"], {"annotation": row["annotation"]})
        if "span" in row:
            entry["span"] = row["span"]
        if "label" in row:
            entry["label"] = row["label"]
        if "pos" in row:
            entry["pos"] = row["pos"]
        if "sense" in row:
            entry["sense"] = row["sense"]
    old_evidence = {
        other_id: state.records[other_id].evidence
        for other_id in state.records
    }

    # 记录裁定本身；同一标注的后一次边界/标签裁定覆盖前一次。
    state.adjudications = [
        row
        for row in state.adjudications
        if not (
            row["annotation"] == annotation_id
            and row.get("action") in {"set_boundary", "set_label", "set_sense"}
            and action in {"set_boundary", "set_label", "set_sense"}
        )
    ]
    record = {
        "annotation": annotation_id,
        "action": action,
        "reviewer": reviewer,
        "note": note,
    }
    if span is not None:
        record["span"] = list(span)
    if label is not None:
        record["label"] = label
    if pos is not None:
        record["pos"] = pos
    if sense is not None:
        record["sense"] = sense
    state.adjudications.append(record)

    if action == "set_boundary":
        overrides[annotation_id] = {
            "annotation": annotation_id,
            "span": list(span),
        }
    if action == "set_label":
        overrides.setdefault(annotation_id, {"annotation": annotation_id})
        overrides[annotation_id]["label"] = label
    if action == "set_sense":
        overrides.setdefault(annotation_id, {"annotation": annotation_id})
        overrides[annotation_id]["sense"] = sense

    before_impact = {
        item["annotation"]: item for item in affected_annotations(packet, annotation_id)
    }
    after_impact = {
        item["annotation"]: item for item in affected_annotations(packet, annotation_id, overrides)
    }
    impacted = []
    for other_id in sorted(set(before_impact) | set(after_impact)):
        reasons = sorted(
            set(before_impact.get(other_id, {}).get("reasons", []))
            | set(after_impact.get(other_id, {}).get("reasons", []))
        )
        impacted.append({"annotation": other_id, "reasons": reasons})
    impact = []
    for item in impacted:
        other = packet.annotations[item["annotation"]]
        new_evidence = evidence_after(packet, other, overrides)
        current = state.records.get(other.id)
        old_status = current.status if current else "pending"
        if current and current.status == "confirmed" and new_evidence == current.evidence:
            new_status = "confirmed"
        elif current and current.status == "confirmed":
            new_status = "pending"
        else:
            new_status = old_status if old_status != "confirmed" else "pending"
        if current is None:
            state.records[other.id] = AnnotationRecord(
                annotation_id=other.id,
                status=new_status,
                evidence=new_evidence,
            )
        else:
            current.evidence = new_evidence
            current.status = new_status
        impact.append(
            {
                "annotation": other.id,
                "reasons": item["reasons"],
                "previous": old_status,
                "status": new_status,
                "evidence_changed": old_evidence.get(other.id) != new_evidence,
            }
        )

    target = state.records.get(annotation_id)
    if target is None:
        target = AnnotationRecord(
            annotation_id=annotation_id,
            status="pending",
            evidence=evidence_after(packet, packet.annotations[annotation_id], overrides),
        )
        state.records[annotation_id] = target
    target.evidence = evidence_after(packet, packet.annotations[annotation_id], overrides)
    if action == "confirm":
        target.status = "confirmed"
        target.decision = "accept"
        target.note = note
    elif action == "reject":
        target.status = "adjudicated"
        target.decision = "reject"
        target.note = note
    else:
        target.status = "adjudicated"
        target.decision = action
        target.note = note
        if span is not None:
            target.span = tuple(span)
        if label is not None:
            target.label = label
        if pos is not None:
            target.pos = pos
        if sense is not None:
            target.sense = sense
    return {
        "annotation": annotation_id,
        "action": action,
        "status": target.status,
        "impact": impact,
    }


def reset_evidence(state: ReviewState, packet: Packet):
    """按当前资料重新计算证据；仅供参考，不改变已确认状态。"""
    for annotation in packet.annotations.values():
        record = state.records.get(annotation.id)
        if record is None:
            state.records[annotation.id] = AnnotationRecord(
                annotation_id=annotation.id,
                status="pending",
                evidence=evidence_fingerprint(packet, annotation),
            )
        else:
            record.evidence = evidence_fingerprint(packet, annotation)


def boundary_view(packet: Packet, record_id: str):
    """同一记录的两种分词边界、跨词实体和词义差异。"""
    markings = packet.marking_for(record_id)
    if not markings:
        return {"record": record_id, "sources": [], "notes": ["没有该记录的分词资料"]}
    document = packet.documents[record_id]
    sources = []
    for marking in markings:
        source = {
            "marking": marking.id,
            "source": marking.source,
            "offset_unit": marking.offset_unit,
            "problems": list(marking.problems),
            "tokens": [],
            "entities": [],
            "relations": [],
        }
        for annotation in marking.annotations:
            if annotation.raw_span is None:
                continue
            context = annotation_context(packet, annotation)
            row = {
                "id": annotation.id,
                "literal": annotation.literal,
                "raw_span": list(annotation.raw_span),
                "source_span": list(annotation.source_span),
                "display_span": list(annotation.display_span),
                "pos": annotation.pos,
                "label": annotation.label,
                "context": context.as_dict(),
                "problems": list(annotation.problems),
            }
            (source["tokens"] if annotation.kind == "token" else source["entities"]).append(row)
        for relation in marking.relations:
            source["relations"].append(relation.summary())
        sources.append(source)

    # 跨词实体：实体区间跨过本来源的分词边界。
    cross_word = []
    comparisons = []
    for source in sources:
        for entity in source["entities"]:
            start, end = entity["raw_span"]
            crossed = [
                token
                for token in source["tokens"]
                if token["raw_span"][0] < end
                and start < token["raw_span"][1]
                and (
                    token["raw_span"][0] > start
                    or token["raw_span"][1] < end
                )
            ]
            if crossed:
                cross_word.append(
                    {
                        "marking": source["marking"],
                        "entity": entity["id"],
                        "entity_literal": entity["literal"],
                        "crossed_tokens": [token["literal"] for token in crossed],
                    }
                )
    if len(sources) > 1:
        comparisons = _compare_sources(packet, sources)

    return {
        "record": record_id,
        "raw": document.raw,
        "origin": document.origin,
        "area": document.area,
        "models": models_in(document.raw),
        "sources": sources,
        "cross_word_entities": cross_word,
        "comparisons": comparisons,
    }


def preview_impact(state: ReviewState, packet: Packet, annotation_id: str, grouping=None):
    """只读预览：裁定这条标注会影响哪些样本与状态，不写入任何裁定。"""
    if annotation_id not in packet.annotations:
        raise KeyError(f"找不到标注 {annotation_id}")
    annotation = packet.annotations[annotation_id]
    impacted = affected_annotations(packet, annotation_id)
    samples = sorted(
        {
            annotation.record,
            *(
                packet.annotations[item["annotation"]].record
                for item in impacted
            ),
        }
    )
    group = None
    if grouping is not None:
        group = grouping.document_to_group.get(annotation.record)
    rows = []
    for item in impacted:
        other = packet.annotations[item["annotation"]]
        current = state.records.get(other.id)
        rows.append(
            {
                "annotation": other.id,
                "record": other.record,
                "reasons": item["reasons"],
                "status": current.status if current else "pending",
                "will_reopen": bool(current and current.status == "confirmed"),
                "note": (
                    "证据未变则保持确认"
                    if current and current.status == "confirmed"
                    else "仍是待确认，会出现在裁定结果里"
                ),
            }
        )
    return {
        "annotation": annotation_id,
        "record": annotation.record,
        "group": group,
        "samples": samples,
        "impact": rows,
    }


def preview_group_impact(state: ReviewState, packet: Packet, group_id: str, grouping):
    """只读预览：某个泄漏组的裁定会牵动哪些样本。

    组内包含固定校验集记录时，任何训练用途的导出都会被牵连；这里把组内
    文档、pin、待确认数量与已确认数量一并列出，避免编辑只看到单条标注。
    """
    group = next((item for item in grouping.groups if item.id == group_id), None)
    if group is None:
        raise KeyError(f"找不到分组 {group_id}")
    rows = []
    for doc_id in group.documents:
        annotations = [
            annotation
            for marking in packet.marking_for(doc_id)
            for annotation in marking.annotations
        ]
        statuses = [
            state.records[annotation.id].status
            if annotation.id in state.records
            else "pending"
            for annotation in annotations
        ]
        rows.append(
            {
                "document": doc_id,
                "origin": packet.documents[doc_id].origin,
                "area": packet.documents[doc_id].area,
                "pinned": any(pin.record == doc_id for pin in packet.pins.values()),
                "annotations": len(annotations),
                "pending": statuses.count("pending"),
                "confirmed": statuses.count("confirmed"),
                "adjudicated": statuses.count("adjudicated"),
            }
        )
    return {
        "group": group_id,
        "documents": group.documents,
        "members": group.members,
        "pins": group.pins,
        "areas": group.areas,
        "samples": rows,
    }


def _compare_sources(packet: Packet, sources):
    """比较两套边界：文字相同但区间不同、实体嵌套不同、词义不同。"""
    entries = []
    for source in sources:
        for kind in ("tokens", "entities"):
            for item in source[kind]:
                entries.append((source["marking"], source["source"], kind, item))
    comparisons = []
    for index, (mark_a, source_a, kind_a, item_a) in enumerate(entries):
        for mark_b, source_b, kind_b, item_b in entries[index + 1 :]:
            if source_a == source_b:
                continue
            overlap = (
                item_a["raw_span"][0] < item_b["raw_span"][1]
                and item_b["raw_span"][0] < item_a["raw_span"][1]
            )
            if not overlap:
                # 词义差异只在同一区间或同一字面重叠时才值得比较；
                # 否则每个词条都会和对方的每个词条配成一对。
                continue
            same = item_a["raw_span"] == item_b["raw_span"]
            same_content = content_text(item_a["literal"]) == content_text(
                item_b["literal"]
            )
            if same and same_content and (
                item_a["context"]["sense"] == item_b["context"]["sense"]
            ):
                continue
            if item_a["context"]["sense"] != item_b["context"]["sense"]:
                kind = "sense"
                detail = {
                    "same_span": same,
                    "same_content": same_content,
                    "a_sense": item_a["context"]["sense"],
                    "b_sense": item_b["context"]["sense"],
                    "a_label": item_a["context"]["label"],
                    "b_label": item_b["context"]["label"],
                }
            elif same and same_content:
                continue
            else:
                kind = "boundary"
                detail = {
                    "same_span": same,
                    "same_content": same_content,
                    "nested": (
                        item_a["raw_span"][0] <= item_b["raw_span"][0]
                        and item_b["raw_span"][1] <= item_a["raw_span"][1]
                    )
                    or (
                        item_b["raw_span"][0] <= item_a["raw_span"][0]
                        and item_a["raw_span"][1] <= item_b["raw_span"][1]
                    ),
                }
            comparisons.append(
                {
                    "kind": kind,
                    "a": {
                        "marking": mark_a,
                        "id": item_a["id"],
                        "literal": item_a["literal"],
                        "span": item_a["raw_span"],
                        "context": item_a["context"]["label"],
                    },
                    "b": {
                        "marking": mark_b,
                        "id": item_b["id"],
                        "literal": item_b["literal"],
                        "span": item_b["raw_span"],
                        "context": item_b["context"]["label"],
                    },
                    "detail": detail,
                }
            )
    # 去重：同一对条目可能在 token/entity 两轮里重复。
    deduped = []
    seen = set()
    for item in comparisons:
        key = (
            item["kind"],
            item["a"]["id"],
            item["b"]["id"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def explain_boundary_history(rows):
    """解释同一字面在不同原稿/来源中的边界差异。"""
    differences = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["source"] == right["source"] and left["record"] == right["record"]:
                continue
            if (
                left["raw_span"] == right["raw_span"]
                and left["record"] == right["record"]
                and left["sense"] == right["sense"]
            ):
                continue
            left_text = left["sentence"]
            right_text = right["sentence"]
            differences.append(
                {
                    "a": {
                        "record": left["record"],
                        "marking": left["marking"],
                        "source": left["source"],
                        "span": left["raw_span"],
                        "literal": left["literal"],
                        "sense": left["sense"],
                    },
                    "b": {
                        "record": right["record"],
                        "marking": right["marking"],
                        "source": right["source"],
                        "span": right["raw_span"],
                        "literal": right["literal"],
                        "sense": right["sense"],
                    },
                    "same_record": left["record"] == right["record"],
                    "span_differs": left["raw_span"] != right["raw_span"],
                    "sense_differs": left["sense"] != right["sense"],
                    "a_sentence": left_text,
                    "b_sentence": right_text,
                    "explanation": _explain_pair(left, right),
                }
            )
    return differences


def _explain_pair(left, right):
    reasons = []
    if left["source"] and right["source"] and left["source"] != right["source"]:
        reasons.append(
            f"{left['source']}（{left['offset_unit']}）与"
            f"{right['source']}（{right['offset_unit']}）是两个合作方的不同切分"
        )
    elif not left["source"] or not right["source"]:
        reasons.append("其中一份原稿还没有对应标注，边界仅由原稿语境推定")
    if left["record"] != right["record"]:
        reasons.append(
            f"分别出自 {left['origin']} 与 {right['origin']}，原稿语境不同"
        )
    if left["sense"] != right["sense"]:
        reasons.append(
            f"词义不同：{left['sense']} 对 {right['sense']}，不能按字面合并"
        )
    if left["record"] == right["record"] and left["raw_span"] != right["raw_span"]:
        reasons.append(f"同一原稿内边界不同：{left['raw_span']} 对 {right['raw_span']}")
    elif left["record"] != right["record"] and not reasons:
        reasons.append("不同原稿中的字符位置不可直接比较，需按各自原稿确认边界")
    if left["annotated"] != right["annotated"]:
        reasons.append("一份是合作方已标注的切分，另一份只有原稿字面")
    return reasons


def literal_boundary_history(packet: Packet, literal: str):
    """同一字面在不同原稿中的边界与词义。

    除了标注里出现的字面，也扫描未标注原稿中的同一字面：这样校对能
    看到"同一字面在另一份原稿里"为何边界或词义不同，而不是只看已标过
    的那几处。`annotated` 为 False 的条目表示该稿件还没有对应标注。
    """
    from .records import find_literal

    target = content_text(literal)
    results = []
    covered = set()
    for marking in packet.markings:
        for annotation in marking.annotations:
            if content_text(annotation.literal) != target:
                continue
            if annotation.raw_span is None:
                continue
            raw = packet.documents[annotation.record].raw
            context = classify(
                raw, annotation.raw_span[0], annotation.raw_span[1], annotation.literal
            )
            covered.add((annotation.record, annotation.raw_span))
            results.append(
                {
                    "record": annotation.record,
                    "origin": packet.documents[annotation.record].origin,
                    "raw": raw,
                    "marking": marking.id,
                    "source": marking.source,
                    "offset_unit": marking.offset_unit,
                    "annotation": annotation.id,
                    "kind": annotation.kind,
                    "raw_span": list(annotation.raw_span),
                    "sentence_span": list(context.sentence),
                    "literal": annotation.literal,
                    "sentence": raw[context.sentence[0] : context.sentence[1]],
                    "sense": context.sense,
                    "label": context.label,
                    "pos": annotation.pos,
                    "entity_label": annotation.label,
                    "annotated": True,
                    "surface_differs": annotation.literal != literal,
                }
            )
    for doc_id, document in packet.documents.items():
        for start, end in find_literal(document.raw, literal):
            if (doc_id, (start, end)) in covered:
                continue
            context = classify(document.raw, start, end, document.raw[start:end])
            results.append(
                {
                    "record": doc_id,
                    "origin": document.origin,
                    "raw": document.raw,
                    "marking": None,
                    "source": None,
                    "offset_unit": "unicode_char",
                    "annotation": None,
                    "kind": None,
                    "raw_span": [start, end],
                    "sentence_span": list(context.sentence),
                    "literal": document.raw[start:end],
                    "sentence": document.raw[context.sentence[0] : context.sentence[1]],
                    "sense": context.sense,
                    "label": context.label,
                    "pos": None,
                    "entity_label": None,
                    "annotated": False,
                    "surface_differs": document.raw[start:end] != literal,
                }
            )
    results.sort(key=lambda row: (row["record"], row["raw_span"][0], row["source"] or ""))
    return results
