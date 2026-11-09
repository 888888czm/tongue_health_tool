# -*- coding: utf-8 -*-
"""舌象特征提取：舌质颜色、舌苔、形态、齿痕、裂纹、瘀点、润燥等。

所有阈值都是"可解释的经验阈值"，不是训练出来的模型。报告里会同时给出
客观指标（Lab/HSV 中位数、面积比等），方便医生或用户判断可信度。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .imaging import Segmentation, _kernel, _odd


class FeatureError(Exception):
    pass


# ----------------------------------------------------------------------------
# 颜色分类（基于 CIELAB 8-bit：L 0-255, a/b 以 128 为中心）
# ----------------------------------------------------------------------------

def classify_body_color(L: float, A: float, B: float) -> str:
    """舌质颜色分类。

    经验阈值（受白平衡影响，报告会标注"估算"）：
      紫暗 : a 偏高、b 偏低、整体偏暗
      绛   : a 很高且 L 很低
      红   : a 很高
      淡红 : 正常
      淡白 : a 偏低
      苍白 : a 很低
    """
    if A >= 145 and B <= 133 and L <= 140:
        return "dark_purple" if L <= 105 else "purple"
    if A >= 163:
        return "deep_red" if (A >= 168 and L <= 125) else "red"
    if A >= 149:
        return "pale_red"
    if A >= 140:
        return "pale"
    return "pale_white"


def classify_coating(
    ratio: float, yellow_fraction: float, greasy: bool
) -> str:
    """舌苔分类：无苔/薄白/厚白/白腻/黄/厚黄。"""
    if ratio < 0.05:
        return "none"
    if yellow_fraction >= 0.35:
        return "thick_yellow" if ratio >= 0.30 else "yellow"
    if greasy and ratio >= 0.10:
        return "white_greasy"
    if ratio >= 0.28:
        return "thick_white"
    return "thin_white"


@dataclass
class TongueFeatures:
    body_color: str = "unknown"
    coating_color: str = "unknown"
    coating_greasy: bool = False
    moisture: str = "normal"
    scalloped: bool = False
    scallop_count: int = 0
    cracks: bool = False
    crack_ratio: float = 0.0
    petechiae: bool = False
    petechiae_count: int = 0
    peripheral_red: bool = False
    width_length_ratio: float = 0.0
    circularity: float = 0.0
    solidity: float = 0.0
    area_ratio: float = 0.0
    quality: str = "medium"
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 确保是原生 Python 类型，方便 JSON 序列化
        for key, value in list(d.items()):
            if isinstance(value, (np.bool_,)):
                d[key] = bool(value)
            elif isinstance(value, (np.integer,)):
                d[key] = int(value)
            elif isinstance(value, (np.floating,)):
                d[key] = float(value)
        return d


def _median(arr: np.ndarray, default: float = 0.0) -> float:
    try:
        if arr is None or arr.size == 0:
            return default
        return float(np.median(arr))
    except Exception:
        return default


def extract_features(bgr: np.ndarray, seg: Segmentation) -> TongueFeatures:
    """从图像和分割结果中提取舌象特征。"""
    if bgr is None or bgr.size == 0:
        raise FeatureError("没有可分析的图像")
    roi = seg.roi
    crop = bgr[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    if crop.size == 0:
        raise FeatureError("取景框为空")

    mask = seg.mask
    if mask.shape[:2] != crop.shape[:2]:
        mask = cv2.resize(
            mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    sel = mask > 0
    total = int(np.count_nonzero(sel))
    if total < 200:
        raise FeatureError("舌体区域过小，无法提取特征，请重新拍摄")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    L = lab[:, :, 0].astype(np.int16)
    A = lab[:, :, 1].astype(np.int16)
    B = lab[:, :, 2].astype(np.int16)

    # ---- 高光（反光点）识别：用于排除对舌质颜色的干扰，同时估计润燥 ----
    specular = sel & (V >= 243) & (S <= 40)
    spec_ratio = float(specular.sum()) / total

    body_sel = sel & ~specular
    if int(body_sel.sum()) < max(50, total // 5):
        body_sel = sel

    body_L = _median(L[body_sel], 150.0)
    body_A = _median(A[body_sel], 155.0)
    body_B = _median(B[body_sel], 138.0)
    body_S = _median(S[body_sel], 90.0)

    # ---- 舌苔 ----
    spec_dil = cv2.dilate(
        specular.astype(np.uint8) * 255, _kernel(5), iterations=1
    ) > 0
    white = (
        sel
        & (L > body_L + 12)
        & (S < 110)
        & (A < body_A - 2)
        & ~spec_dil
    )
    yellow = (
        sel
        & (B > body_B + 6)
        & (B > 150)
        & (S > 55)
        & (H >= 8)
        & (H <= 40)
        & (L > body_L + 5)
    )
    coat = white | yellow
    coat_count = int(coat.sum())
    coating_ratio = coat_count / float(total)
    yellow_fraction = (float(yellow.sum()) / coat_count) if coat_count > 0 else 0.0

    greasy = False
    if coat_count >= 200 and coating_ratio >= 0.10:
        texture = float(np.std(L[coat]))
        greasy = texture >= 20.0

    coating_color = classify_coating(coating_ratio, yellow_fraction, greasy)

    # ---- 润燥 ----
    if spec_ratio >= 0.030:
        moisture = "wet"
    elif spec_ratio <= 0.0020:
        moisture = "dry"
    else:
        moisture = "normal"

    # ---- 形态 ----
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise FeatureError("舌体轮廓提取失败，请重新拍摄")
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    circularity = (4.0 * math.pi * area / (peri * peri)) if peri > 0 else 0.0

    ratio = 0.0
    try:
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        mx = max(rw, rh)
        if mx > 1:
            ratio = float(min(rw, rh) / mx)
    except Exception:
        ratio = 0.0

    # ---- 齿痕：轮廓相对凸包的凹陷 ----
    scallop_count = 0
    try:
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is not None and len(hull) >= 4 and len(cnt) >= 4:
            defects = cv2.convexityDefects(cnt, hull.astype(np.int32))
            if defects is not None:
                # OpenCV 4.x 返回 (N,1,4)，OpenCV 5.x 返回 (N,4)，统一处理
                arr = defects.reshape(-1, 4)
                min_depth = 0.025 * min(mask.shape[:2])
                for item in arr:
                    depth = float(item[3]) / 256.0
                    if depth >= min_depth:
                        scallop_count += 1
    except Exception:
        scallop_count = 0
    scalloped = scallop_count >= 2

    # ---- 裂纹：黑帽变换提取细暗线 ----
    crack_ratio = 0.0
    try:
        k = _odd(min(mask.shape[:2]) * 0.05, 5)
        kernel = _kernel(k)
        er = cv2.erode(mask, kernel, iterations=1)
        er_sel = er > 0
        if int(er_sel.sum()) > 100:
            bh = cv2.morphologyEx(
                L.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel
            )
            vals = bh[er_sel]
            thr = max(15.0, float(vals.mean()) + 2.5 * float(vals.std()))
            dark = (bh.astype(np.float32) >= thr) & er_sel
            crack_ratio = float(dark.sum()) / float(er_sel.sum())
    except Exception:
        crack_ratio = 0.0
    cracks = crack_ratio >= 0.018

    # ---- 瘀点：舌面上的小片深红/暗红斑点（排除边缘阴影与舌苔区）----
    petechiae_count = 0
    petechiae = False
    try:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        inner = dist >= max(3.0, float(dist.max()) * 0.15)
        spot = (
            sel
            & inner
            & ~coat
            & (L <= body_L - 25)
            & (A >= body_A + 8)
            & (A <= body_A + 45)
            & (S >= 70)
            & (B <= body_B + 10)
            & ((H <= 12) | (H >= 168))
        )
        spot_u8 = (spot.astype(np.uint8)) * 255
        n, _, stats, _ = cv2.connectedComponentsWithStats(spot_u8, connectivity=8)
        max_area = max(200.0, total * 0.002)
        small_areas = []
        for i in range(1, n):
            a = float(stats[i, cv2.CC_STAT_AREA])
            if 4 <= a <= max_area:
                small_areas.append(a)
        petechiae_count = len(small_areas)
        petechiae = petechiae_count >= 4 and (sum(small_areas) / total) >= 0.0020
    except Exception:
        petechiae_count = 0
        petechiae = False

    # ---- 舌尖/舌边偏红：外周环 vs 中心核心的 a* 差异（排除舌苔与反光）----
    peripheral_red = False
    try:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        maxd = float(dist.max())
        if maxd > 4:
            ring = (dist > 0) & (dist <= maxd * 0.22) & ~coat & ~specular
            core = (dist >= maxd * 0.55) & ~coat & ~specular
            min_px = max(200, int(total * 0.05))
            if int(ring.sum()) >= min_px and int(core.sum()) >= min_px:
                delta = _median(A[ring]) - _median(A[core])
                peripheral_red = delta >= 6.0
    except Exception:
        peripheral_red = False

    metrics = {
        "body_L": round(body_L, 2),
        "body_A": round(body_A, 2),
        "body_B": round(body_B, 2),
        "body_S": round(body_S, 2),
        "specular_ratio": round(spec_ratio, 5),
        "coating_ratio": round(coating_ratio, 5),
        "yellow_fraction": round(yellow_fraction, 5),
        "crack_ratio": round(crack_ratio, 5),
        "circularity": round(circularity, 4),
        "width_length_ratio": round(ratio, 4),
        "area_ratio": round(float(seg.area_ratio), 4),
        "solidity": round(float(seg.solidity), 4),
        "tongue_pixels": total,
    }

    return TongueFeatures(
        body_color=classify_body_color(body_L, body_A, body_B),
        coating_color=coating_color,
        coating_greasy=bool(greasy),
        moisture=moisture,
        scalloped=bool(scalloped),
        scallop_count=int(scallop_count),
        cracks=bool(cracks),
        crack_ratio=float(crack_ratio),
        petechiae=bool(petechiae),
        petechiae_count=int(petechiae_count),
        peripheral_red=bool(peripheral_red),
        width_length_ratio=float(ratio),
        circularity=float(circularity),
        solidity=float(seg.solidity),
        area_ratio=float(seg.area_ratio),
        quality=seg.quality,
        metrics=metrics,
    )
