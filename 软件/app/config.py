# -*- coding: utf-8 -*-
"""全局配置与常量。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "舌象健康观察助手"
APP_NAME_EN = "Tongue Observation Assistant"
VERSION = "1.0.0"

# 分析前把长边缩到这个尺寸，兼顾速度与精度
MAX_WORK_SIDE = 1280

# 默认取景框（相对整幅画面的比例：x, y, w, h）
DEFAULT_ROI = (0.14, 0.10, 0.72, 0.80)

# 报告默认输出目录
REPORT_DIR_NAME = "reports"

DISCLAIMER_SHORT = (
    "本软件只做舌象观察与健康科普参考，不构成医学诊断，不输出疾病名称，"
    "不能替代医生面诊、化验和影像检查。请勿据此自行用药或停药。"
)

DISCLAIMER_LONG = (
    "本报告依据摄像头拍摄的舌象照片，使用公开、通用的图像处理规则（颜色阈值、"
    "形态学、轮廓分析）对舌质颜色、舌苔、形态等做描述，并给出中医舌象的"
    "“倾向性参考”。它不是医学诊断，不输出疾病名称，也不预测患病概率，"
    "不能替代医生面诊、化验和影像检查。任何不适请及时就医；请勿依据本报告"
    "自行用药、停药或改变治疗方案。"
)


def is_frozen() -> bool:
    """是否为 PyInstaller 打包后的可执行文件。"""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """可执行文件所在目录（打包后）或项目根目录（源码运行）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(relative: str) -> Path:
    """获取随程序打包的资源路径，兼容 PyInstaller onefile。"""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = Path(base) / relative
        if p.exists():
            return p
    return app_dir() / relative


def user_data_dir() -> Path:
    """可写的用户数据目录。优先使用程序目录，不可写时退回用户目录。"""
    candidates = [app_dir(), Path.home() / ".tongue_screening"]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return c
        except Exception:
            continue
    return Path(os.getcwd())


def reports_dir() -> Path:
    d = user_data_dir() / REPORT_DIR_NAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path(os.getcwd())
    return d


def logs_dir() -> Path:
    d = user_data_dir() / "logs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path(os.getcwd())
    return d
