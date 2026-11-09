# -*- coding: utf-8 -*-
"""舌象图像采集、取景框（ROI）与舌体分割。

说明
----
这里只使用教科书级的通用图像处理原语：
  * Otsu 阈值（1979 年公开发表）
  * HSV / CIELAB 颜色空间阈值
  * 形态学开闭运算、连通域分析
  * 轮廓与凸包缺陷（OpenCV 公开 API）

不包含任何特定厂商的装置结构、训练模型或专利算法实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DEFAULT_ROI, MAX_WORK_SIDE


class ImageError(Exception):
    """图像相关错误的基类（对外显示友好提示）。"""


class ImageLoadError(ImageError):
    pass


class TongueNotFoundError(ImageError):
    pass


def _odd(n: float, minimum: int = 3) -> int:
    n = int(max(minimum, n))
    return n if n % 2 == 1 else n + 1


def _kernel(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> "ROI":
        x = max(0, min(int(self.x), max(0, width - 2)))
        y = max(0, min(int(self.y), max(0, height - 2)))
        w = max(2, min(int(self.w), width - x))
        h = max(2, min(int(self.h), height - y))
        return ROI(x, y, w, h)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def default_roi(width: int, height: int) -> ROI:
    rx, ry, rw, rh = DEFAULT_ROI
    return ROI(
        int(width * rx), int(height * ry), int(width * rw), int(height * rh)
    ).clamp(width, height)


def load_image(path: str) -> np.ndarray:
    """读取图片，兼容中文路径。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise ImageLoadError("文件为空或无法读取")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except ImageLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"无法读取图片：{exc}") from exc
    if img is None or img.size == 0:
        raise ImageLoadError("无法解码该图片，请换一张 JPG/PNG 格式的照片")
    return img


def resize_max(img: np.ndarray, max_side: int = MAX_WORK_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    return cv2.resize(
        img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def assess_capture_quality(bgr: np.ndarray) -> List[str]:
    """给出拍摄质量提示（不抛异常）。"""
    issues: List[str] = []
    try:
        if bgr is None or bgr.size == 0:
            return ["画面为空"]
        small = resize_max(bgr, 640)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mean_v = float(v.mean())
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if mean_v < 45:
            issues.append("画面偏暗，请补充正面光照")
        elif mean_v > 238:
            issues.append("画面过亮/过曝，请避开强光或关闭美颜补光")
        if blur < 12:
            issues.append("画面偏模糊，请保持手机/摄像头稳定并重新对焦")
        if small.shape[0] < 240 or small.shape[1] < 240:
            issues.append("分辨率偏低，建议使用 640×480 以上画面")
    except Exception:
        return []
    return issues


@dataclass
class Segmentation:
    mask: np.ndarray          # ROI 尺寸的 0/255 掩膜
    roi: ROI
    area_ratio: float         # 舌体面积 / 取景框面积
    solidity: float           # 面积 / 凸包面积
    quality: str              # high / medium / low
    method: str               # 使用的分割策略
    warnings: List[str] = field(default_factory=list)


def _fill_external(mask: np.ndarray) -> np.ndarray:
    """用最外层轮廓填充内部小洞（舌苔高光等），保留外轮廓。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def _border_fraction(mask: np.ndarray) -> float:
    """掩膜中贴近取景框边缘的像素比例（用于排除皮肤/背景大块）。"""
    m = mask > 0
    if not m.any():
        return 1.0
    b = max(2, int(min(mask.shape[:2]) * 0.012))
    edge = np.zeros_like(m)
    edge[:b, :] = True
    edge[-b:, :] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    return float((m & edge).sum()) / float(m.sum())



def _mask_metrics(mask: np.ndarray) -> Tuple[float, float, Optional[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 1:
        return 0.0, 0.0, None
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = area / hull_area if hull_area > 1 else 0.0
    return area, max(0.0, min(1.0, solidity)), cnt


def segment_tongue(
    bgr: np.ndarray,
    roi: Optional[Sequence[int]] = None,
    strict: bool = True,
) -> Segmentation:
    """在取景框内分割舌体。

    策略（依次尝试，取第一个面积/实心度达标的）：
      1. Otsu(a*) ∩ 红/紫色域掩膜  —— 最稳
      2. 仅 Otsu(a*)
      3. 仅红/紫色域掩膜
    无法找到足够大的区域时，strict=True 抛 TongueNotFoundError，
    否则返回低质量结果并附警告。
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ImageLoadError("没有可分析的图像")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ImageLoadError("图像格式不受支持（需要彩色图）")

    H, W = bgr.shape[:2]
    if roi is None:
        roi_obj = default_roi(W, H)
    else:
        try:
            x, y, w, h = [int(v) for v in roi]
            if w <= 0 or h <= 0:
                roi_obj = default_roi(W, H)
            else:
                roi_obj = ROI(x, y, w, h).clamp(W, H)
        except Exception:
            roi_obj = default_roi(W, H)

    crop = bgr[roi_obj.y : roi_obj.y + roi_obj.h, roi_obj.x : roi_obj.x + roi_obj.w]
    if crop.size == 0:
        raise ImageLoadError("取景框超出图像范围，请重新框选")

    ch, cw = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hch = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    a_ch = lab[:, :, 1]

    k = _odd(min(cw, ch) * 0.03, 3)
    a_blur = cv2.GaussianBlur(a_ch, (k, k), 0)
    try:
        _, otsu = cv2.threshold(a_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        otsu = np.zeros((ch, cw), np.uint8)
        otsu[a_ch >= int(np.percentile(a_ch, 60))] = 255

    red = (((hch <= 25) | (hch >= 155)) & (sat >= 30) & (val >= 30))
    purple = ((hch >= 120) & (hch < 155) & (sat >= 18) & (val >= 18))
    color_mask = ((red | purple).astype(np.uint8)) * 255
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, _kernel(k))

    candidates = [
        ("otsu+color", cv2.bitwise_and(otsu, color_mask)),
        ("otsu", otsu),
        ("color", color_mask),
    ]

    total_px = float(cw * ch)
    cx0, cy0 = cw / 2.0, ch / 2.0
    diag = math.hypot(cw, ch) or 1.0
    best = None  # (score, mask, method, area_ratio, solidity)

    def _consider(name: str, cand: np.ndarray, relaxed: bool = False):
        nonlocal best
        try:
            num, labels, stats, cents = cv2.connectedComponentsWithStats(
                cand, connectivity=8
            )
        except Exception:
            return
        if num <= 1:
            return
        for i in range(1, num):
            area_raw = float(stats[i, cv2.CC_STAT_AREA])
            if area_raw < max(80.0, total_px * 0.01):
                continue
            comp = (labels == i).astype(np.uint8) * 255
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, _kernel(k))
            touch = _border_fraction(comp)
            filled = _fill_external(comp)
            area, solidity, _ = _mask_metrics(filled)
            if area <= 0:
                continue
            ratio = area / total_px
            if ratio < 0.035 or ratio > 0.90:
                continue
            if not relaxed and (touch > 0.12 or solidity < 0.45):
                continue
            # 舌体应在取景框中央：中心点是否落在该连通域（含内部小洞）内
            inside = bool(filled[min(ch - 1, max(0, int(cy0))), min(cw - 1, max(0, int(cx0)))])
            dist = math.hypot(cents[i][0] - cx0, cents[i][1] - cy0) / diag
            score = (
                (1.0 if inside else 0.0)
                + 0.7 * (1.0 - min(1.0, dist * 3.0))
                + 0.3 * solidity
                - 0.6 * touch
                + 0.2 * min(ratio, 0.4)
            )
            if best is None or score > best[0]:
                best = (score, filled, name, ratio, solidity)
            if inside and ratio >= 0.08 and solidity >= 0.60:
                return

    for name, cand in candidates:
        _consider(name, cand, relaxed=False)
        if best is not None and best[0] >= 1.4:
            break

    # 所有候选都不理想时，放宽条件再取一次，保证流程不中断
    if best is None:
        for name, cand in candidates:
            _consider(name, cand, relaxed=True)

    if best is None:
        if strict:
            raise TongueNotFoundError(
                "没有在取景框内识别到舌体。请正对光源、张大嘴伸舌，"
                "让舌体填满中央方框，并避开嘴唇、牙齿和手指。"
            )
        # 兜底：用整幅 ROI 作为掩膜，保证流程不中断
        mask = np.full((ch, cw), 255, np.uint8)
        return Segmentation(
            mask=mask,
            roi=roi_obj,
            area_ratio=1.0,
            solidity=0.0,
            quality="low",
            method="fallback",
            warnings=["未能自动分割舌体，结果仅基于整幅取景框，参考价值很低"],
        )

    _, mask, method, ratio, solidity = best
    mask = _fill_external(mask)
    area, solidity, _ = _mask_metrics(mask)
    ratio = area / total_px

    # 颜色合理性检查：舌体在 CIELAB 的 a* 通道应明显偏红；
    # 随机噪声、灰色墙面等通常 a* 接近 128，直接判为未识别。
    try:
        mean_a = float(a_ch[mask > 0].mean())
    except Exception:
        mean_a = 128.0
    if strict and mean_a < 138.0:
        raise TongueNotFoundError(
            "取景框内的区域不像舌体（颜色偏灰/偏暗）。请正对光源、"
            "张大嘴伸舌，让舌体填满中央方框后重试。"
        )

    warnings: List[str] = []
    if ratio < 0.06:
        warnings.append("舌体在画面中偏小，建议靠近一些重新拍摄")
    if solidity < 0.72:
        warnings.append("舌体轮廓不够完整，可能被嘴唇或牙齿遮挡")
    if ratio > 0.97:
        warnings.append("取景框可能包含了过多非舌体区域")

    if ratio >= 0.12 and solidity >= 0.85:
        quality = "high"
    elif ratio >= 0.06 and solidity >= 0.65:
        quality = "medium"
    else:
        quality = "low"

    if quality == "low" and strict:
        # 仍然返回结果，但明确告知质量不足
        warnings.append("图像质量偏低，以下征象仅供粗略参考")

    return Segmentation(
        mask=mask,
        roi=roi_obj,
        area_ratio=float(ratio),
        solidity=float(solidity),
        quality=quality,
        method=method,
        warnings=warnings,
    )


def annotate(
    bgr: np.ndarray, seg: Optional[Segmentation], color=(0, 255, 0)
) -> np.ndarray:
    """把取景框和分割轮廓画到图上，用于报告配图。"""
    out = bgr.copy()
    try:
        if seg is not None:
            x, y, w, h = seg.roi.as_tuple()
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
            contours, _ = cv2.findContours(
                seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            shifted = [c + np.array([[x, y]]) for c in contours]
            cv2.drawContours(out, shifted, -1, color, 2)
    except Exception:
        pass
    return out


def encode_preview(bgr: np.ndarray, max_side: int = 520, quality: int = 82) -> bytes:
    """把图像编码为 JPEG 字节流（用于 HTML 报告内嵌）。"""
    try:
        img = resize_max(bgr, max_side)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return b""
