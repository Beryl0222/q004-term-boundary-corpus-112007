"""命令行入口：检查、对齐、语境、分组、校对、发布与查询。

所有命令只读资料包；需要写文件时显式给出 --output。命令输出 JSON，便于
校对界面和发布脚本使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .context import document_term_occurrences, term_literals
from .grouping import build_grouping, leak_evidence
from .records import load_packet, validate_pins
from .release import (
    ReleaseError,
    audit_provenance,
    export_labeled,
    leak_impact,
    plan_release,
    release_record,
)
from .review import (
    ReviewState,
    adjudicate,
    boundary_view,
    explain_boundary_history,
    literal_boundary_history,
    preview_group_impact,
    preview_impact,
)


def _emit(data, output=None):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _load_state(path):
    if not path or not Path(path).exists():
        return None
    return ReviewState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_check(args):
    packet = load_packet(args.packet)
    problems = validate_pins(packet)
    marking_problems = [
        {"marking": marking.id, "problems": marking.problems}
        for marking in packet.markings
        if marking.problems
    ]
    annotation_problems = [
        {"annotation": annotation.id, "problems": annotation.problems}
        for annotation in packet.annotations.values()
        if annotation.problems
    ]
    data = {
        "documents": len(packet.documents),
        "qa": len(packet.qa),
        "markings": len(packet.markings),
        "annotations": len(packet.annotations),
        "pins": [pin.public_id for pin in packet.pins.values()],
        "releases": [release.id for release in packet.releases.values()],
        "problems": problems,
        "marking_problems": marking_problems,
        "annotation_problems": annotation_problems,
    }
    _emit(data, args.output)
    return 1 if problems else 0


def cmd_align(args):
    packet = load_packet(args.packet)
    data = []
    for marking in packet.markings:
        if args.record and marking.record != args.record:
            continue
        data.append(
            {
                "marking": marking.id,
                "record": marking.record,
                "source": marking.source,
                "offset_unit": marking.offset_unit,
                "display": marking.display,
                "raw": packet.documents[marking.record].raw,
                "canonical_display": marking.alignment.canonical_display,
                "canonical_raw": marking.alignment.canonical_raw,
                "unmatched_display": marking.alignment.unmatched_display,
                "unmatched_raw": marking.alignment.unmatched_raw,
                "problems": list(marking.problems),
                "annotations": [
                    annotation.summary() for annotation in marking.annotations
                ],
            }
        )
    _emit({"markings": data}, args.output)


def cmd_terms(args):
    packet = load_packet(args.packet)
    if args.record:
        document = packet.documents[args.record]
        data = [item.as_dict() for item in document_term_occurrences(packet, document)]
    else:
        data = {
            doc_id: [item.as_dict() for item in document_term_occurrences(packet, document)]
            for doc_id, document in packet.documents.items()
        }
    _emit({"terms": term_literals(packet), "occurrences": data}, args.output)


def cmd_boundaries(args):
    packet = load_packet(args.packet)
    _emit(boundary_view(packet, args.record), args.output)


def cmd_groups(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    if args.preview:
        state = _load_state(args.state) or ReviewState()
        state.ensure(packet)
        _emit(
            preview_group_impact(state, packet, args.preview, grouping),
            args.output,
        )
        return 0
    _emit(grouping.as_dict(), args.output)


def cmd_leaks(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    training = args.training or [
        doc for doc in packet.documents if doc not in {
            pin.record for pin in packet.pins.values()
        }
    ]
    findings = leak_evidence(grouping, training, packet)
    _emit({"training": training, "findings": findings}, args.output)


def cmd_plan(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    plan = plan_release(
        packet,
        args.training,
        grouping,
        previous_training=args.previous,
    )
    _emit(plan.as_dict(), args.output)
    return 0 if plan.ok else 1


def cmd_impact(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    impact = leak_impact(packet, args.training, grouping)
    _emit(impact.as_dict(), args.output)


def cmd_release(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    plan = plan_release(packet, args.training, grouping)
    if plan.leaked:
        # 训练集必须清理泄漏材料；不允许带着泄漏直接发布。
        _emit(plan.as_dict(), args.output)
        print(
            "发布被拒绝：训练清单含泄漏材料，请按 plan 的 removed 清理后重试",
            file=sys.stderr,
        )
        return 1
    state = _load_state(args.state) or ReviewState()
    state.ensure(packet)
    record = release_record(
        packet,
        plan,
        state,
        release_id=args.id,
        date=args.date,
        grouping=grouping,
    )
    _emit(record, args.output)
    return 0


def cmd_export(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    state = _load_state(args.state) or ReviewState()
    state.ensure(packet)
    result = export_labeled(
        packet,
        state,
        doc_ids=args.document or None,
        grouping=grouping,
        include_pinned=args.include_pinned,
    )
    _emit(result.as_dict(), args.output)


def cmd_review(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    state = _load_state(args.state) or ReviewState()
    state.ensure(packet)
    if args.action == "preview":
        data = preview_impact(state, packet, args.annotation, grouping)
    elif args.action == "list":
        data = state.as_dict()
    else:
        data = adjudicate(
            state,
            packet,
            args.annotation,
            action=args.action,
            span=args.span,
            label=args.label,
            pos=args.pos,
            sense=args.sense,
            note=args.note,
            reviewer=args.reviewer,
        )
        if args.save:
            Path(args.save).write_text(
                json.dumps(state.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    _emit(data, args.output)


def cmd_history(args):
    packet = load_packet(args.packet)
    rows = literal_boundary_history(packet, args.literal)
    _emit(
        {
            "literal": args.literal,
            "occurrences": rows,
            "differences": explain_boundary_history(rows),
        },
        args.output,
    )


def cmd_provenance(args):
    packet = load_packet(args.packet)
    grouping = build_grouping(packet)
    rows = audit_provenance(packet, grouping)
    if args.annotation:
        rows = [row for row in rows if row["annotation"] == args.annotation]
    if args.record:
        rows = [row for row in rows if row["record"] == args.record]
    _emit({"provenance": rows}, args.output)


def build_parser():
    parser = argparse.ArgumentParser(description="术语语料的边界、校对与发布工具")
    parser.add_argument("packet", type=Path, help="资料包 JSON 路径")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="检查资料包与 pin")
    check.add_argument("--output", type=Path)
    check.set_defaults(func=cmd_check)

    align = sub.add_parser("align", help="显示两家分词与原稿的对齐")
    align.add_argument("--record")
    align.add_argument("--output", type=Path)
    align.set_defaults(func=cmd_align)

    terms = sub.add_parser("terms", help="按语境列出术语出现")
    terms.add_argument("--record")
    terms.add_argument("--output", type=Path)
    terms.set_defaults(func=cmd_terms)

    boundaries = sub.add_parser("boundaries", help="两种分词边界与跨词实体")
    boundaries.add_argument("--record", required=True)
    boundaries.add_argument("--output", type=Path)
    boundaries.set_defaults(func=cmd_boundaries)

    groups = sub.add_parser("groups", help="泄漏分组与证据")
    groups.add_argument("--preview", help="预览某组的裁定会影响哪些样本")
    groups.add_argument("--state", type=Path)
    groups.add_argument("--output", type=Path)
    groups.set_defaults(func=cmd_groups)

    leaks = sub.add_parser("leaks", help="训练材料到校验 pin 的泄漏依据")
    leaks.add_argument("--training", nargs="+")
    leaks.add_argument("--output", type=Path)
    leaks.set_defaults(func=cmd_leaks)

    plan = sub.add_parser("plan", help="计算隔离后的训练集与覆盖缺失")
    plan.add_argument("--training", nargs="+", required=True)
    plan.add_argument("--previous", nargs="*")
    plan.add_argument("--output", type=Path)
    plan.set_defaults(func=cmd_plan)

    impact = sub.add_parser("impact", help="已交付版本的影响")
    impact.add_argument("--training", nargs="+", required=True)
    impact.add_argument("--output", type=Path)
    impact.set_defaults(func=cmd_impact)

    release = sub.add_parser("release", help="生成发布记录")
    release.add_argument("--training", nargs="+", required=True)
    release.add_argument("--id", required=True)
    release.add_argument("--date", required=True)
    release.add_argument("--state", type=Path)
    release.add_argument("--output", type=Path)
    release.set_defaults(func=cmd_release)

    export = sub.add_parser("export", help="导出已确认标注")
    export.add_argument("--document", nargs="*")
    export.add_argument("--state", type=Path)
    export.add_argument("--include-pinned", action="store_true")
    export.add_argument("--output", type=Path)
    export.set_defaults(func=cmd_export)

    review = sub.add_parser("review", help="校对状态与局部裁定")
    review.add_argument("action", choices=["list", "preview", "confirm", "reject", "set_boundary", "set_label", "set_sense"])
    review.add_argument("annotation", nargs="?")
    review.add_argument("--span", nargs=2, type=int)
    review.add_argument("--label")
    review.add_argument("--pos")
    review.add_argument("--sense")
    review.add_argument("--note")
    review.add_argument("--reviewer", default="校对")
    review.add_argument("--state", type=Path)
    review.add_argument("--save", type=Path)
    review.add_argument("--output", type=Path)
    review.set_defaults(func=cmd_review)

    history = sub.add_parser("history", help="同一字面在不同原稿中的边界")
    history.add_argument("literal")
    history.add_argument("--output", type=Path)
    history.set_defaults(func=cmd_history)

    provenance = sub.add_parser("provenance", help="发布记录中标注的来由")
    provenance.add_argument("--annotation")
    provenance.add_argument("--record")
    provenance.add_argument("--output", type=Path)
    provenance.set_defaults(func=cmd_provenance)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (KeyError, ReleaseError, ValueError, FileNotFoundError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
