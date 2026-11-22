"""把互相能提示答案的材料并成泄漏组。

分组的判据是可核对的内容证据，不是"都出现过同一个术语"。字面共现（例如
两篇材料都提到"相变存储器"）不构成证据；同文献片段、同义转述、同一技术
问答、以及正文里指向答案出处的引用才构成证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .records import Packet, content_text, find_literal
from .context import POINTER_CUES

MIN_SHARED_FRAGMENT = 6
PARAPHRASE_MIN_SIMILARITY = 0.6


def longest_common_substring(left: str, right: str):
    """返回 (长度, 字串)。数据量小，直接动态规划。"""
    best_length = 0
    best_end = 0
    previous = [0] * (len(right) + 1)
    for i, char in enumerate(left, start=1):
        current = [0] * (len(right) + 1)
        for j, other in enumerate(right, start=1):
            if char == other:
                current[j] = previous[j - 1] + 1
                if current[j] > best_length:
                    best_length = current[j]
                    best_end = i
        previous = current
    return best_length, left[best_end - best_length : best_end]


def _shared_fragment_evidence(left: str, right: str):
    left_text = content_text(left)
    right_text = content_text(right)
    length, fragment = longest_common_substring(left_text, right_text)
    if length < MIN_SHARED_FRAGMENT:
        return None
    if not fragment:
        return None
    coverage = length / max(1, min(len(left_text), len(right_text)))
    # 一篇材料的大部分内容都出现在另一篇里，才像"同一文献的片段"；
    # 过短的共同句也可能只是两边都转述了同一事实，留给转述判据。
    if coverage < 0.5 and length < 12:
        return None
    return {
        "kind": "verbatim_fragment",
        "fragment": fragment,
        "length": length,
        "coverage": round(coverage, 3),
        "description": (
            f"两篇材料共有 {length} 个连续内容字符“{fragment}”，"
            "属于同一文献的片段"
        ),
    }


def _term_and_model_profile(packet: Packet, doc_id: str):
    from .context import document_term_occurrences, models_in

    document = packet.documents[doc_id]
    occurrences = document_term_occurrences(packet, document)
    terms = []
    for item in occurrences:
        cleaned = content_text(item.literal)
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return terms, set(models_in(document.raw))


def _term_overlap(left_terms, right_terms):
    best = None
    for left in left_terms:
        for right in right_terms:
            if left == right:
                score = 1.0
            else:
                from .context import _edit_distance

                distance = _edit_distance(left, right, 2)
                if distance > 2:
                    continue
                score = 1 - distance / max(len(left), len(right))
            if score >= PARAPHRASE_MIN_SIMILARITY:
                candidate = (score, left, right)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best


def _paraphrase_evidence(packet: Packet, left_id: str, right_id: str):
    left_terms, left_models = _term_and_model_profile(packet, left_id)
    right_terms, right_models = _term_and_model_profile(packet, right_id)
    models = left_models & right_models
    if not models:
        return None
    overlap = _term_overlap(left_terms, right_terms)
    if not overlap:
        return None
    score, left_term, right_term = overlap
    left_text = content_text(packet.documents[left_id].raw)
    right_text = content_text(packet.documents[right_id].raw)
    length, fragment = longest_common_substring(left_text, right_text)
    return {
        "kind": "paraphrase",
        "models": sorted(models),
        "terms": [left_term, right_term],
        "similarity": round(score, 3),
        "shared_fragment": fragment,
        "shared_fragment_length": length,
        "description": (
            f"同一型号 {'/'.join(sorted(models))} 与近乎相同的术语"
            f"“{left_term}”/“{right_term}”同时出现，构成转述"
        ),
    }


def _pointer_evidence(packet: Packet, doc_id: str):
    """找出正文里指向别篇答案出处的引用。

    只有"见/参见 + 出处名称"这种能让读者直接翻到答案的引用才算证据；
    mentions 只是著录，单独不足以分组。出处名称允许轻微的排版或写法
    差异（例如"旧机型问答"对"旧机问答"）。
    """
    from .context import _edit_distance

    document = packet.documents[doc_id]
    text = document.raw
    evidence = []
    for cue in POINTER_CUES:
        for start, end in find_literal(text, cue):
            sentence_start = max(text.rfind(mark, 0, start) for mark in "。！？!?\n") + 1
            sentence_end = len(text)
            for mark in "。！？!?\n":
                found = text.find(mark, end)
                if found != -1:
                    sentence_end = min(sentence_end, found)
            clause = text[sentence_start:sentence_end]
            phrase = content_text(text[end:sentence_end])
            if len(phrase) < 2:
                continue
            for other in packet.documents.values():
                if other.id == doc_id:
                    continue
                origin = content_text(other.origin)
                question = content_text(packet.qa[other.id].question) if other.id in packet.qa else ""
                matches = []
                for target, label in ((origin, "出处"), (question, "问句")):
                    if not target:
                        continue
                    distance = _edit_distance(phrase, target, 2)
                    if distance > 2:
                        continue
                    prefix = 0
                    while (
                        prefix < min(len(phrase), len(target))
                        and phrase[prefix] == target[prefix]
                    ):
                        prefix += 1
                    if prefix < 2:
                        continue
                    matches.append((distance, label, target))
                if not matches:
                    continue
                distance, label, target = min(matches)
                evidence.append(
                    {
                        "kind": "answer_pointer",
                        "from": doc_id,
                        "to": other.id,
                        "cue": cue,
                        "clause": clause,
                        "target_origin": other.origin,
                        "target_question": packet.qa[other.id].question
                        if other.id in packet.qa
                        else None,
                        "matched": label,
                        "expression": text[end:sentence_end],
                        "distance": distance,
                        "mentions": other.id in document.mentions,
                        "description": (
                            f"正文“{clause.strip()}”引用“{other.origin}”，"
                            f"其{label}与所指出处的写法只差 {distance} 个字，"
                            "读者可据此直接找到答案"
                        ),
                    }
                )
    return evidence


@dataclass
class Edge:
    a: str
    b: str
    kind: str
    evidence: dict

    def as_dict(self):
        return {"a": self.a, "b": self.b, "kind": self.kind, "evidence": self.evidence}


@dataclass
class Group:
    id: str
    members: list[str]
    documents: list[str]
    pins: list[str]
    areas: list[str]
    edges: list[Edge] = field(default_factory=list)

    def as_dict(self):
        return {
            "id": self.id,
            "members": list(self.members),
            "documents": list(self.documents),
            "pins": list(self.pins),
            "areas": list(self.areas),
            "evidence": [edge.as_dict() for edge in self.edges],
        }


@dataclass
class Grouping:
    groups: list[Group]
    edges: list[Edge]
    document_to_group: dict[str, str]
    member_to_group: dict[str, str] = field(default_factory=dict)

    def group_for(self, doc_id: str) -> Group:
        group_id = self.member_to_group.get(doc_id) or self.document_to_group.get(doc_id)
        return next(group for group in self.groups if group.id == group_id)

    def as_dict(self):
        return {
            "groups": [group.as_dict() for group in self.groups],
            "document_to_group": dict(self.document_to_group),
            "member_to_group": dict(self.member_to_group),
            "edges": [edge.as_dict() for edge in self.edges],
        }


def build_grouping(packet: Packet) -> Grouping:
    edges: list[Edge] = []
    node_ids = list(packet.documents) + list(packet.qa)

    def add(a, b, kind, evidence):
        if any(
            {edge.a, edge.b} == {a, b} and edge.kind == kind for edge in edges
        ):
            return
        edges.append(Edge(a, b, kind, evidence))

    # 同一技术问答的 question/answer 与所引用文献同组发表。
    for qa in packet.qa.values():
        add(
            qa.id,
            qa.record,
            "qa_pair",
            {
                "kind": "qa_pair",
                "description": "question 与 answer 为同组发表，answer 可直接提示该文献内容",
                "record": qa.record,
            },
        )

    document_ids = list(packet.documents)
    for index, left_id in enumerate(document_ids):
        for right_id in document_ids[index + 1 :]:
            left = packet.documents[left_id]
            right = packet.documents[right_id]
            verbatim = _shared_fragment_evidence(left.raw, right.raw)
            if verbatim:
                add(left_id, right_id, "verbatim_fragment", verbatim)
                continue
            paraphrase = _paraphrase_evidence(packet, left_id, right_id)
            if paraphrase:
                add(left_id, right_id, "paraphrase", paraphrase)

            # 正文指向答案出处的引用。mentions 只是著录信息，仅作旁证。
            for pointer in _pointer_evidence(packet, left_id):
                if pointer["to"] == right_id:
                    pointer["mentioned"] = right_id in left.mentions
                    add(left_id, right_id, "answer_pointer", pointer)
            for pointer in _pointer_evidence(packet, right_id):
                if pointer["to"] == left_id:
                    pointer["mentioned"] = left_id in right.mentions
                    add(left_id, right_id, "answer_pointer", pointer)

    parent = {node: node for node in node_ids}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in edges:
        union(edge.a, edge.b)

    components = {}
    for node in node_ids:
        components.setdefault(find(node), []).append(node)

    groups = []
    document_to_group = {}
    member_to_group = {}
    for number, members in enumerate(
        sorted(components.values(), key=lambda items: (-len(items), items)),
        start=1,
    ):
        docs = [item for item in members if item in packet.documents]
        qa_members = [item for item in members if item in packet.qa]
        pins = sorted(
            pin.public_id for pin in packet.pins.values() if pin.record in docs
        )
        areas = sorted({packet.documents[doc].area for doc in docs})
        group_edges = [
            edge for edge in edges if edge.a in members and edge.b in members
        ]
        group = Group(
            id=f"G{number}",
            members=sorted(members),
            documents=sorted(docs),
            pins=pins,
            areas=areas,
            edges=group_edges,
        )
        groups.append(group)
        for doc in docs:
            document_to_group[doc] = group.id
        for member in members:
            member_to_group[member] = group.id
    return Grouping(groups, edges, document_to_group, member_to_group)


def path_between(grouping: Grouping, start: str, end: str):
    """在组内找一条证据最短路径，用于说明具体泄漏依据。"""
    adjacency = {}
    for edge in grouping.edges:
        adjacency.setdefault(edge.a, []).append((edge.b, edge))
        adjacency.setdefault(edge.b, []).append((edge.a, edge))
    queue = [(start, [])]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        if node == end:
            return path
        for neighbour, edge in adjacency.get(node, []):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append((neighbour, path + [edge]))
    return None


def leak_evidence(grouping: Grouping, training_docs, packet: Packet):
    """返回每条训练材料到校验 pin 的泄漏依据。"""
    findings = []
    pin_by_record = {}
    for pin in packet.pins.values():
        pin_by_record.setdefault(pin.record, []).append(pin)
    for doc_id in training_docs:
        group = grouping.group_for(doc_id) if doc_id in grouping.document_to_group else None
        if group is None:
            continue
        for pin in packet.pins.values():
            if pin.public_id not in group.pins:
                continue
            if pin.record == doc_id:
                findings.append(
                    {
                        "document": doc_id,
                        "pin": pin.public_id,
                        "check_record": pin.record,
                        "path": [],
                        "reason": "训练材料本身就是校验记录",
                    }
                )
                continue
            path = path_between(grouping, doc_id, pin.record)
            if path is None:
                continue
            findings.append(
                {
                    "document": doc_id,
                    "pin": pin.public_id,
                    "check_record": pin.record,
                    "check_area": packet.documents[pin.record].area,
                    "path": [edge.as_dict() for edge in path],
                    "basis": [edge.evidence.get("description", edge.kind) for edge in path],
                }
            )
    return findings
