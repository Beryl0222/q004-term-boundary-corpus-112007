"""术语语料的边界对齐、校对与发布隔离工具。"""

from .records import load_packet, packet_from_dict
from .unicode_align import align_display_to_raw, char_span_to_utf8

__all__ = [
    "load_packet",
    "packet_from_dict",
    "align_display_to_raw",
    "char_span_to_utf8",
]
