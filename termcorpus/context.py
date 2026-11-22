"""语境与词义判定。

只对齐字符串会把否定回答中的术语用法和正面定义混为一谈：本模块按小句
确定否定辖域，并把"术语提及""比较项"与"器件实体"分开。判定是建议而不
是裁定，校对可以推翻；同字面不同词义不会被自动合并。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from .records import Annotation, Document, Packet, content_text

SENTENCE_DELIMITERS = "。！？!?\n"
CLAUSE_DELIMITERS = "，,；;：:、"

NEGATION_CUES = (
    "不是",
    "并非",
    "不属于",
    "不使用",
    "不采用",
    "不涉及",
    "不含",
    "不用",
    "没有",
    "未使用",
    "未采用",
    "无",
    "非",
)
CONTRAST_CUES = ("而是", "但是", "但", "然而", "不过", "实际", "改为", "却", "而")
DEFINITION_CUES = ("称为", "定义为", "指的是", "是指", "即")
USE_CUES = ("使用", "采用", "选用", "配置", "搭载", "装用", "用")
META_CUES = ("用语", "术语", "词汇", "一词", "概念", "称呼", "说法", "字面")
COMPARISON_CUES = ("比较", "对比", "相比", "区分", "不同于", "相对于")
ANAPHORA_CUES = ("该部件", "该器件", "该元件", "此部件", "此器件", "它", "其")
POINTER_CUES = ("见", "参见", "详见", "另见", "问答", "FAQ", "旧机型问答")

MODEL_PATTERN = re.compile(r"[A-Za-z][-－]?\d+", re.IGNORECASE)

class ContextError(ValueError):
    pass


def _split_spans(text: str, delimiters: str):
    spans = []
    start = 0
    for index, char in enumerate(text):
        if char in delimiters:
            spans.append((start, index + 1))
            start = index + 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def sentence_span(text: str, position: int):
    for start, end in _split_spans(text, SENTENCE_DELIMITERS):
        if start <= position < end:
            return start, end
        if position == len(text) and end == len(text):
            return start, end
    return 0, len(text)


def clause_span(text: str, position: int):
    start, end = sentence_span(text, position)
    for left, right in _split_spans(text[start:end], CLAUSE_DELIMITERS):
        if start + left <= position < start + right:
            return start + left, start + right
    return start, end


def _cue_positions(text: str, start: int, end: int, cues):
    found = []
    for cue in cues:
        cursor = text.find(cue, start, end)
        while cursor != -1:
            found.append((cursor, cursor + len(cue), cue))
            cursor = text.find(cue, cursor + 1, end)
    return sorted(found)


@dataclass
class Context:
    label: str  # 定义 / 肯定使用 / 否定使用 / 术语提及 / 比较提及 / 其它
    sense: str  # 器件实体 / 术语提及 / 比较项 / 未定
    sentence: tuple[int, int]
    clause: tuple[int, int]
    cues: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "label": self.label,
            "sense": self.sense,
            "sentence": list(self.sentence),
            "clause": list(self.clause),
            "cues": list(self.cues),
            "notes": list(self.notes),
        }


def classify(text: str, start: int, end: int, literal: str) -> Context:
    """对 text 中 [start, end) 的字面做语境判定。"""
    if not 0 <= start < end <= len(text):
        raise ContextError(f"区间越界 {start}-{end}")
    sentence = sentence_span(text, start)
    clause = clause_span(text, start)
    note = []
    cues = []

    meta = _cue_positions(text, clause[0], clause[1], META_CUES)
    comparison = _cue_positions(text, clause[0], clause[1], COMPARISON_CUES)
    negations = _cue_positions(text, clause[0], start, NEGATION_CUES)
    uses = _cue_positions(text, clause[0], clause[1], USE_CUES)
    definitions = _cue_positions(text, clause[0], clause[1], DEFINITION_CUES)

    # 否定辖域：同句否定标志若只作用于前一小句（如"不是磁带，XC……"），
    # 不能波及本术语。同小句内若出现转折，也不自动判为否定。
    scoped_negation = []
    for cue_start, cue_end, cue in negations:
        between = text[cue_end:start]
        if any(marker in between for marker in CONTRAST_CUES):
            note.append(f"否定标志“{cue}”后有转折，不直接否定本术语")
            continue
        scoped_negation.append((cue_start, cue_end, cue))
    for _, _, cue in negations:
        cues.append({"cue": cue, "type": "negation", "scoped": bool(scoped_negation)})

    # 同句其他小句中的否定：只记录，不改变本术语所在小句的判定。
    sentence_negations = [
        (s, e, c)
        for s, e, c in _cue_positions(text, sentence[0], sentence[1], NEGATION_CUES)
        if not (s <= start < e) and not (clause[0] <= s < clause[1])
    ]

    if meta:
        label, sense = "术语提及", "术语提及"
        cues.append({"cue": meta[0][2], "type": "meta"})
        note.append("术语在这里被当作字面或行话提及，不是器件定义")
    elif comparison:
        label, sense = "比较提及", "比较项"
        cues.append({"cue": comparison[0][2], "type": "comparison"})
        note.append("术语处于比较结构，未断言设备使用该部件")
    elif scoped_negation:
        label, sense = "否定使用", "器件实体"
        cues.append({"cue": scoped_negation[0][2], "type": "negation-scope"})
    elif definitions:
        label, sense = "定义", "器件实体"
        cues.append({"cue": definitions[0][2], "type": "definition"})
    elif uses and any(cue_end <= start for _, cue_end, _ in uses):
        label, sense = "肯定使用", "器件实体"
        cues.append({"cue": uses[0][2], "type": "use"})
    else:
        label, sense = "其它", "未定"
        note.append("没有明确的使用、否定或定义线索，留给校对")

    if sentence_negations and label in {"肯定使用", "定义"}:
        cue = sentence_negations[0][2]
        note.append(f"同句另有否定“{cue}”，其辖域不在本术语所在小句")

    if label == "术语提及" or sense == "术语提及":
        tail = text[clause[1] : sentence[1]]
        anaphora = [cue for cue in ANAPHORA_CUES if cue in tail]
        if anaphora and any(cue in tail for cue in NEGATION_CUES):
            note.append(
                f"后句以“{anaphora[0]}”加否定回指，可能指向本术语；否定用法不能记到术语字面本身"
            )
            cues.append({"cue": anaphora[0], "type": "anaphora"})

    return Context(label, sense, sentence, clause, cues, note)


def term_literals(packet: Packet):
    """从已标注内容得到候选术语字面。

    实体标注给出的组件/设备/材料是主要来源；名词词条只有在不被更长的
    实体字面包含时才单独收入，避免把"存储器"同时当成"相变存储器"。
    这里得到的是候选，词义仍由语境判定，同字面不会因为字符串相同就
    自动合并。
    """
    entity_terms = []
    noun_terms = []
    for marking in packet.markings:
        for annotation in marking.annotations:
            literal = annotation.literal
            cleaned = content_text(literal)
            if len(cleaned) < 2 or not any("一" <= char <= "鿿" for char in cleaned):
                continue
            if annotation.kind == "entity" and annotation.label in {
                "组件",
                "设备",
                "材料",
                "术语",
            }:
                entity_terms.append(literal)
            elif annotation.kind == "token" and annotation.pos == "名词":
                noun_terms.append(literal)
    entity_cleaned = [content_text(item) for item in entity_terms]
    terms = list(dict.fromkeys(entity_terms))
    for literal in noun_terms:
        cleaned = content_text(literal)
        if any(cleaned in other for other in entity_cleaned):
            continue
        if literal not in terms:
            terms.append(literal)
    return sorted(terms, key=lambda item: (-len(content_text(item)), item))


def _edit_distance(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char != other),
                )
            )
        previous = current
    return previous[-1]


def _lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for char in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, start=1):
            if char == other:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def variant_candidates(text: str, terms, min_length=3, threshold=0.6, known_spans=()):
    """在同一文档中寻找与已知术语近似、但边界不同的字面。

    只报告候选：跨字形的近似不能证明两处材料互为转述，也不能据此合并
    词义。与被认可术语重叠的窗口视为边界差异，不当作新术语。
    """
    from .records import find_literal

    candidates = []
    for term in terms:
        needle = content_text(term)
        if len(needle) < min_length:
            continue
        for width in range(max(min_length, len(needle) - 1), len(needle) + 2):
            for start in range(0, len(text) - width + 1):
                end = start + width
                if any(start < right and left < end for left, right in known_spans):
                    continue
                window = text[start:end]
                if any(char in SENTENCE_DELIMITERS + CLAUSE_DELIMITERS for char in window):
                    continue
                cleaned = content_text(window)
                if len(cleaned) < min_length or cleaned == needle:
                    continue
                # 只考虑与术语长度相近的窗口：多出一个字就可能只是把相邻
                # 词吞进来，而不是同一术语的另一写法。
                if not len(needle) - 1 <= len(cleaned) <= len(needle) + 1:
                    continue
                if not any("一" <= char <= "鿿" for char in cleaned):
                    continue
                distance = _edit_distance(cleaned, needle, 2)
                if distance > 2:
                    continue
                score = 1 - distance / max(len(cleaned), len(needle))
                if score < threshold:
                    continue
                prefix = 0
                while (
                    prefix < min(len(cleaned), len(needle))
                    and cleaned[prefix] == needle[prefix]
                ):
                    prefix += 1
                suffix = 0
                while (
                    suffix < min(len(cleaned), len(needle)) - prefix
                    and cleaned[-1 - suffix] == needle[-1 - suffix]
                ):
                    suffix += 1
                # 至少有一端与术语相同，才像同一术语的另一种切法；否则
                # 只是碰巧共享字符，不能自动对照。
                if prefix < 2 and suffix < 2:
                    continue
                candidates.append(
                    {
                        "term": term,
                        "literal": window,
                        "span": (start, end),
                        "similarity": round(score, 3),
                    }
                )
    unique = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item["span"][1] - item["span"][0]), item["span"]),
    ):
        if any(
            existing["span"][0] <= candidate["span"][0]
            and candidate["span"][1] <= existing["span"][1]
            and existing["term"] == candidate["term"]
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


@dataclass
class TermOccurrence:
    doc_id: str
    literal: str
    span: tuple[int, int]
    context: Context
    variant_of: str | None = None
    similarity: float | None = None

    def as_dict(self):
        return {
            "doc": self.doc_id,
            "literal": self.literal,
            "span": list(self.span),
            "variant_of": self.variant_of,
            "similarity": self.similarity,
            "context": self.context.as_dict(),
        }


def document_term_occurrences(packet: Packet, document: Document):
    from .records import find_literal

    terms = term_literals(packet)
    results = []
    covered = []
    for term in terms:
        for start, end in find_literal(document.raw, term):
            if any(start < right and left < end for left, right in covered):
                # 被更长术语覆盖的同位置字面仍单独保留，词义判定会区分。
                pass
            results.append(
                TermOccurrence(
                    doc_id=document.id,
                    literal=document.raw[start:end],
                    span=(start, end),
                    context=classify(document.raw, start, end, document.raw[start:end]),
                )
            )
            covered.append((start, end))
    for candidate in variant_candidates(document.raw, terms, known_spans=covered):
        start, end = candidate["span"]
        # 与已认可字面在内容上等价的窗口只是边界差异；它会在跨稿比较
        # 中作为另一套边界显示，不应重复成一枚新术语。
        if content_text(document.raw[start:end]) == content_text(candidate["term"]):
            continue
        results.append(
            TermOccurrence(
                doc_id=document.id,
                literal=document.raw[start:end],
                span=(start, end),
                context=classify(document.raw, start, end, document.raw[start:end]),
                variant_of=candidate["term"],
                similarity=candidate["similarity"],
            )
        )
    results.sort(key=lambda item: (item.span[0], -(item.span[1] - item.span[0])))
    deduped = []
    for item in results:
        if any(
            existing.span == item.span and existing.literal == item.literal
            for existing in deduped
        ):
            continue
        deduped.append(item)
    return deduped


def annotation_context(packet: Packet, annotation: Annotation) -> Context:
    raw = packet.documents[annotation.record].raw
    if annotation.raw_span is None:
        return Context("无法定位", "未定", (0, 0), (0, 0), [], ["映射到原样文本失败"])
    start, end = annotation.raw_span
    return classify(raw, start, end, annotation.literal)


def models_in(text: str):
    normalized = unicodedata.normalize("NFKC", text)
    return sorted(
        {match.group(0).upper().replace("－", "-") for match in MODEL_PATTERN.finditer(normalized)}
    )
