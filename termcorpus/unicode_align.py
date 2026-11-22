"""显示文本与原样文本之间的可解释对齐。

合作方在 display 上标注，原文以 raw 保存；二者的差异只允许是排版差异：
换行等空白、全角型号（Ａ -> A）以及组合字符（e + ◌́ -> é）。任何改变
字面意义的差异都不会在这里被"猜"掉，而是留给上层标为待校对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata


class OffsetError(ValueError):
    """偏移不在合法字符边界上。"""


@dataclass(frozen=True)
class CanonicalUnit:
    """一个非空白规范字符，以及它在原字符串中的字符区间。"""

    char: str
    start: int
    end: int


def _is_mark(char: str) -> bool:
    return unicodedata.combining(char) != 0 or unicodedata.category(char) in {"Mn", "Me"}


def clusters(text: str):
    """按基字符 + 后继组合字符切分，返回字符区间。"""
    bounds = []
    start = 0
    for index, char in enumerate(text):
        if index > start and not _is_mark(char):
            bounds.append((start, index))
            start = index
    if start < len(text):
        bounds.append((start, len(text)))
    return bounds


def canonical_units(text: str):
    """返回用于比较的规范字符序列。

    NFKC 负责全角字形和兼容字符，组合字符由它一并合成。空白不参与
    匹配，因此换行和空格的增删不会造成未匹配。
    """
    units = []
    for start, end in clusters(text):
        piece = unicodedata.normalize("NFKC", text[start:end])
        for char in piece:
            if char.isspace():
                continue
            units.append(CanonicalUnit(char, start, end))
    return units


@dataclass
class DisplayRawAlignment:
    display: str
    raw: str
    # 每个 display 字符对应的 raw 字符区间；None 表示该字符不在 raw 中。
    char_to_raw: list[tuple[int, int] | None] = field(default_factory=list)
    unmatched_display: list[int] = field(default_factory=list)
    unmatched_raw: list[int] = field(default_factory=list)
    canonical_display: str = ""
    canonical_raw: str = ""
    # 规范字符的数量不同，例如 display 为 Ａ、raw 为 A 时长度相同；
    # 长度不同并不代表失败。
    length_changed: bool = False

    @property
    def exact(self) -> bool:
        return (
            not self.unmatched_display
            and not self.unmatched_raw
            and self.display == self.raw
        )

    def _position_near(self, index: int) -> int:
        """为没有规范字符的字符（通常是空白）选一个插入点。"""
        for cursor in range(index, len(self.display)):
            span = self.char_to_raw[cursor]
            if span is not None:
                return span[0]
        for cursor in range(index - 1, -1, -1):
            span = self.char_to_raw[cursor]
            if span is not None:
                return span[1]
        return 0

    def char_boundary_is_cluster_edge(self, index: int) -> bool:
        if index <= 0 or index >= len(self.display):
            return True
        # display 的组合字符必须与基字符处在同一侧。
        return not _is_mark(self.display[index])

    def map_span(self, left: int, right: int):
        """把 display 的字符区间映射为 raw 字符区间。

        返回 (raw_left, raw_right, ambiguous, notes)。ambiguous 表示边界
        落在组合字符簇内部或有非空白字符无法在 raw 中定位，此时调用方
        应把该标注交给校对，而不是自动采信。
        """
        if not 0 <= left <= right <= len(self.display):
            raise OffsetError(f"display 区间越界 {left}-{right}")
        notes = []
        ambiguous = False
        if not self.char_boundary_is_cluster_edge(left) or not self.char_boundary_is_cluster_edge(right):
            ambiguous = True
            notes.append("边界落在组合字符簇内部")
        spans = []
        for index in range(left, right):
            span = self.char_to_raw[index]
            if span is None:
                if not self.display[index].isspace():
                    ambiguous = True
                    notes.append(f"display 第 {index} 个字符在 raw 中无对应")
                continue
            spans.append(span)
        if spans:
            return min(s for s, _ in spans), max(e for _, e in spans), ambiguous, notes
        position = self._position_near(left)
        return position, position, ambiguous, notes


def _global_align(a: list[CanonicalUnit], b: list[CanonicalUnit]):
    """Needleman-Wunsch：规范字符相同代价 0，替换 2，增删 1。"""
    rows, cols = len(a), len(b)
    infinity = 10**9
    cost = [[infinity] * (cols + 1) for _ in range(rows + 1)]
    move = [[""] * (cols + 1) for _ in range(rows + 1)]
    cost[0][0] = 0
    for i in range(1, rows + 1):
        cost[i][0] = i
        move[i][0] = "up"
    for j in range(1, cols + 1):
        cost[0][j] = j
        move[0][j] = "left"
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            same = a[i - 1].char == b[j - 1].char
            diagonal = cost[i - 1][j - 1] + (0 if same else 2)
            up = cost[i - 1][j] + 1
            left = cost[i][j - 1] + 1
            best = min(diagonal, up, left)
            cost[i][j] = best
            # 优先匹配/替换，其次删除 display 字符，最后插入 raw 字符。
            if same and diagonal == best:
                move[i][j] = "match"
            elif diagonal == best:
                move[i][j] = "sub"
            elif up == best:
                move[i][j] = "up"
            else:
                move[i][j] = "left"
    pairs = []
    unmatched_a = []
    unmatched_b = []
    i, j = rows, cols
    while i > 0 or j > 0:
        step = move[i][j]
        if step in {"match", "sub"}:
            if step == "match":
                pairs.append((i - 1, j - 1))
            else:
                unmatched_a.append(i - 1)
                unmatched_b.append(j - 1)
            i -= 1
            j -= 1
        elif step == "up":
            unmatched_a.append(i - 1)
            i -= 1
        else:
            unmatched_b.append(j - 1)
            j -= 1
    pairs.reverse()
    unmatched_a.reverse()
    unmatched_b.reverse()
    return pairs, unmatched_a, unmatched_b


def align_display_to_raw(display: str, raw: str) -> DisplayRawAlignment:
    """对齐合作方展示文本与原样文本，保留两侧字符位置。"""
    display_units = canonical_units(display)
    raw_units = canonical_units(raw)
    pairs, unmatched_display, unmatched_raw = _global_align(display_units, raw_units)
    char_to_raw: list[tuple[int, int] | None] = [None] * len(display)
    for display_index, raw_index in pairs:
        unit = display_units[display_index]
        raw_span = (raw_units[raw_index].start, raw_units[raw_index].end)
        for char_index in range(unit.start, unit.end):
            char_to_raw[char_index] = raw_span
    return DisplayRawAlignment(
        display=display,
        raw=raw,
        char_to_raw=char_to_raw,
        unmatched_display=[display_units[i].start for i in unmatched_display],
        unmatched_raw=[raw_units[i].start for i in unmatched_raw],
        canonical_display="".join(unit.char for unit in display_units),
        canonical_raw="".join(unit.char for unit in raw_units),
        length_changed=len(display_units) != len(raw_units),
    )


def canonical_text(text: str) -> str:
    """仅消除排版差异后的文本，用于比较字面。"""
    return "".join(unit.char for unit in canonical_units(text))


def layout_text(text: str) -> str:
    """消除排版差异（空白、全角字形、组合字符表示），保留标点等字符。"""
    return "".join(unit.char for unit in canonical_units(text))


@dataclass(frozen=True)
class LayoutProjection:
    text: str
    spans: tuple[tuple[int, int], ...]


def layout_projection(text: str) -> LayoutProjection:
    """返回去除空白后的规范文本，以及每个规范字符的原字符区间。"""
    chars = []
    spans = []
    for start, end in clusters(text):
        piece = unicodedata.normalize("NFKC", text[start:end])
        for char in piece:
            if char.isspace():
                continue
            chars.append(char)
            spans.append((start, end))
    return LayoutProjection("".join(chars), tuple(spans))


def char_span_to_utf8(text: str, start: int, end: int) -> tuple[int, int]:
    if not 0 <= start <= end <= len(text):
        raise OffsetError(f"字符区间越界 {start}-{end}")
    return (
        len(text[:start].encode("utf-8")),
        len(text[:end].encode("utf-8")),
    )


def char_to_utf16_offsets(text: str) -> list[int]:
    """返回每个字符边界对应的 UTF-16 编码单元偏移。"""
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode("utf-16-le")) // 2)
    return offsets


def utf16_span_to_char(text: str, start: int, end: int) -> tuple[int, int]:
    offsets = char_to_utf16_offsets(text)
    try:
        left = offsets.index(start)
        right = offsets.index(end)
    except ValueError as error:
        raise OffsetError(
            f"UTF-16 区间 {start}-{end} 未落在字符边界上"
        ) from error
    return left, right


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def convert_span(text: str, start: int, end: int, unit: str) -> tuple[int, int]:
    """把来源的偏移口径转换成 Unicode 字符区间。"""
    if unit == "unicode_char":
        if not 0 <= start <= end <= len(text):
            raise OffsetError(f"字符区间越界 {start}-{end}")
        return start, end
    if unit == "utf16":
        if not 0 <= start <= end <= utf16_length(text):
            raise OffsetError(f"UTF-16 区间越界 {start}-{end}")
        return utf16_span_to_char(text, start, end)
    raise OffsetError(f"未知偏移口径 {unit}")
