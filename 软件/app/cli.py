# -*- coding: utf-8 -*-
"""命令行模式：适合批量分析、集成到其他流程或做自动化测试。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .demo import PALETTES, make_tongue_image
from .features import FeatureError, extract_features
from .imaging import (
    ImageError,
    TongueNotFoundError,
    assess_capture_quality,
    load_image,
    resize_max,
    segment_tongue,
)
from .knowledge import KnowledgeBaseError, evaluate, load_knowledge_base
from .report import build_report
from .utils import parse_roi, setup_logging


def run_cli(args: argparse.Namespace) -> int:
    setup_logging()

    try:
        kb = load_knowledge_base(args.kb)
    except KnowledgeBaseError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    # 1) 取图
    image_path: Optional[str] = None
    if args.image:
        try:
            img = load_image(args.image)
            image_path = args.image
        except ImageError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 2
    else:
        demo_name = args.demo if isinstance(args.demo, str) else "normal"
        if demo_name not in PALETTES:
            demo_name = "normal"
        img = make_tongue_image(demo_name)
        print(f"[提示] 未指定 --image，使用内置示例图：{demo_name}")

    img = resize_max(img)
    roi = parse_roi(args.roi)

    issues = assess_capture_quality(img)
    for item in issues:
        print(f"[拍摄提示] {item}", file=sys.stderr)

    # 2) 分割 + 特征 + 打分
    try:
        seg = segment_tongue(img, roi)
    except TongueNotFoundError as exc:
        print(f"[提示] {exc}", file=sys.stderr)
        return 3
    except ImageError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    try:
        feats = extract_features(img, seg)
    except FeatureError as exc:
        print(f"[提示] {exc}", file=sys.stderr)
        return 3

    matches = evaluate(feats.to_dict(), kb)
    report = build_report(
        feats, matches, kb, seg=seg, image=img, image_path=image_path
    )

    # 3) 输出
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.to_text())

    if not args.no_save:
        paths = report.save(args.out)
        print()
        print(f"[已保存] 文本报告：{paths['txt']}")
        print(f"[已保存] 可打印网页：{paths['html']}")
        print(f"[已保存] 结构化数据：{paths['json']}")
    return 0
