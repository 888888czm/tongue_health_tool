# -*- coding: utf-8 -*-
"""内置示例图生成与自检。

用途：
  * 没有摄像头时演示/验收软件（--demo）
  * 打包后自检（--selftest），确认核心算法在本机可运行

示例图是程序合成的，不含任何真实患者数据。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


# 调色板：RGB 颜色 + 预期征象（仅用于自检，不参与实际分析）
PALETTES: Dict[str, dict] = {
    "normal": {
        "body": (196, 122, 130),
        "coating": None,
        "coating_ratio": 0.0,
        "scalloped": False,
        "cracks": False,
        "petechiae": False,
        "highlights": 0.010,
        "width": 0.200,
        "expect_body": {"pale_red"},
        "expect_top": {"balanced"},
    },
    "pale_scallop": {
        "body": (214, 168, 170),
        "coating": (232, 228, 222),
        "coating_ratio": 0.34,
        "scalloped": True,
        "cracks": False,
        "petechiae": False,
        "highlights": 0.016,
        "width": 0.235,
        "expect_body": {"pale", "pale_white"},
        "expect_top": {"spleen_damp", "phlegm_damp", "qi_blood_deficiency"},
    },
    "heat": {
        "body": (200, 80, 85),
        "coating": (226, 210, 150),
        "coating_ratio": 0.36,
        "scalloped": False,
        "cracks": False,
        "petechiae": False,
        "highlights": 0.0005,
        "width": 0.195,
        "expect_body": {"red", "deep_red"},
        "expect_top": {"internal_heat", "phlegm_damp", "heart_liver_fire"},
    },
    "stasis": {
        "body": (150, 92, 112),
        "coating": None,
        "coating_ratio": 0.0,
        "scalloped": False,
        "cracks": False,
        "petechiae": True,
        "highlights": 0.010,
        "width": 0.200,
        "expect_body": {"purple", "dark_purple"},
        "expect_top": {"blood_stasis"},
    },
    "yin": {
        "body": (200, 85, 92),
        "coating": None,
        "coating_ratio": 0.0,
        "scalloped": False,
        "cracks": True,
        "petechiae": False,
        "highlights": 0.0005,
        "width": 0.150,
        "expect_body": {"red", "deep_red"},
        "expect_top": {"yin_deficiency", "internal_heat", "heart_liver_fire"},
    },
}


def _rgb2bgr(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (int(c[2]), int(c[1]), int(c[0]))


def make_tongue_image(name: str = "normal", size: int = 720, seed: int = 7) -> np.ndarray:
    """生成一张合成的舌象照片（BGR）。"""
    cfg = PALETTES.get(name, PALETTES["normal"])
    rng = np.random.default_rng(seed)
    h = w = int(size)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # --- 皮肤背景（带轻微渐变与噪声）---
    skin = np.array(_rgb2bgr((208, 172, 154)), dtype=np.float32)
    shade = 1.0 - 0.16 * (yy / h) - 0.10 * ((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) / (w * h)
    bg = np.clip(skin[None, None, :] * shade[..., None], 0, 255)
    bg += rng.normal(0, 2.2, (h, w, 1))
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # --- 口腔（暗色）---
    mouth_center = (w // 2, int(h * 0.54))
    mouth_axes = (int(w * 0.295), int(h * 0.345))
    cv2.ellipse(bg, mouth_center, mouth_axes, 0, 0, 360, (46, 34, 40), -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)

    # --- 舌体 ---
    body_bgr = np.array(_rgb2bgr(cfg["body"]), dtype=np.float32)
    cx, cy = w // 2, int(h * 0.545)
    ax, ay = int(w * cfg["width"]), int(h * 0.275)
    tongue_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(tongue_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    # 舌体纹理：中心稍亮、边缘稍暗 + 细噪声
    radial = 1.0 - 0.16 * (((xx - cx) / max(ax, 1)) ** 2 + ((yy - cy) / max(ay, 1)) ** 2)
    texture = np.clip(radial, 0.85, 1.05)[..., None]
    tongue = np.clip(body_bgr[None, None, :] * texture + rng.normal(0, 3.0, (h, w, 1)), 0, 255)
    tongue = tongue.astype(np.uint8)

    # 舌苔（椭圆软边叠加）
    ratio = float(cfg.get("coating_ratio") or 0.0)
    if cfg.get("coating") and ratio > 0:
        coat_mask = np.zeros((h, w), np.uint8)
        cax = int(ax * min(0.92, (ratio ** 0.5) * 1.15))
        cay = int(ay * min(0.92, (ratio ** 0.5) * 1.10))
        cv2.ellipse(coat_mask, (cx, cy - int(ay * 0.06)), (max(4, cax), max(4, cay)), 0, 0, 360, 255, -1)
        coat_mask = cv2.GaussianBlur(coat_mask, (0, 0), max(2.0, min(ax, ay) * 0.06))
        alpha = (coat_mask.astype(np.float32) / 255.0)[..., None] * 0.78
        coat_bgr = np.array(_rgb2bgr(cfg["coating"]), dtype=np.float32)
        coat = np.clip(coat_bgr[None, None, :] + rng.normal(0, 4.0, (h, w, 1)), 0, 255)
        tongue = (tongue.astype(np.float32) * (1 - alpha) + coat * alpha).astype(np.uint8)

    # 裂纹
    if cfg.get("cracks"):
        dark = tuple(int(v * 0.72) for v in body_bgr)
        for i in range(3):
            x0 = cx + int(rng.integers(-ax * 0.35, ax * 0.35))
            pts = []
            for t in np.linspace(-0.55, 0.62, 7):
                px = x0 + int(rng.normal(0, ax * 0.06))
                py = cy + int(ay * t)
                pts.append((px, py))
            cv2.polylines(tongue, [np.array(pts, np.int32)], False, dark, 2, cv2.LINE_AA)

    # 瘀点
    if cfg.get("petechiae"):
        spot = _rgb2bgr((104, 30, 52))
        for _ in range(16):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0.15, 0.72)
            px = int(cx + np.cos(ang) * ax * rad)
            py = int(cy + np.sin(ang) * ay * rad)
            r = int(rng.integers(2, 5))
            cv2.circle(tongue, (px, py), r, spot, -1, cv2.LINE_AA)

    # 合成到背景
    m3 = (tongue_mask > 0)[..., None]
    img = np.where(m3, tongue, bg)

    # 齿痕：沿舌体左右边缘"咬"出缺口（用口腔颜色覆盖）
    if cfg.get("scalloped"):
        mouth_bgr = (46, 34, 40)
        for side in (-1, 1):
            for t in np.linspace(-0.55, 0.55, 5):
                px = int(cx + side * ax * 0.985)
                py = int(cy + ay * t)
                cv2.circle(img, (px, py), int(min(ax, ay) * 0.085), mouth_bgr, -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (0, 0), 1.1)

    # 高光（反光点）
    hi = float(cfg.get("highlights") or 0.0)
    if hi > 0.002:
        n = max(1, int(round(hi * 800)))
        for _ in range(n):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0.0, 0.55)
            px = int(cx + np.cos(ang) * ax * rad)
            py = int(cy + np.sin(ang) * ay * rad * 0.8 - ay * 0.18)
            rw = int(rng.integers(3, 9))
            rh = int(rng.integers(2, 6))
            cv2.ellipse(img, (px, py), (rw, rh), 0, 0, 360, (245, 248, 250), -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (0, 0), 1.2)

    # 整体轻微模糊 + 噪声，模拟真实摄像头
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    img = np.clip(
        img.astype(np.float32) + rng.normal(0, 1.6, img.shape), 0, 255
    ).astype(np.uint8)
    return img


def save_demo_images(out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in PALETTES:
        img = make_tongue_image(name)
        path = os.path.join(out_dir, f"demo_{name}.png")
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def run_selftest(verbose: bool = True) -> int:
    """打包后自检：生成 5 张示例图并跑完整分析流程。"""
    from .features import extract_features
    from .imaging import segment_tongue
    from .knowledge import evaluate, load_knowledge_base

    failures = []
    try:
        kb = load_knowledge_base()
    except Exception as exc:  # noqa: BLE001
        print(f"[自检] 知识库加载失败：{exc}")
        return 2

    for name, cfg in PALETTES.items():
        try:
            img = make_tongue_image(name)
            seg = segment_tongue(img)
            feats = extract_features(img, seg)
            matches = evaluate(feats.to_dict(), kb)
            top = matches[0].name if matches else "无"
            top_id = matches[0].id if matches else ""
            ok_body = feats.body_color in cfg["expect_body"]
            ok_top = top_id in cfg["expect_top"]
            status = "OK " if (ok_body and ok_top) else "WARN"
            if not (ok_body and ok_top):
                failures.append(name)
            if verbose:
                print(
                    f"[自检] {status} {name:12s} 舌质={feats.body_color:10s} "
                    f"舌苔={feats.coating_color:12s} 齿痕={int(feats.scalloped)} "
                    f"裂纹={int(feats.cracks)} 瘀点={int(feats.petechiae)} "
                    f"→ 首位={top}"
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"[自检] FAIL {name}: {exc}")

    if failures:
        print(f"[自检] 有 {len(failures)} 个示例未达到预期，请查看日志。")
        return 1
    print("[自检] 全部通过：核心流程、知识库、规则匹配均正常。")
    return 0
