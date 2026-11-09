# -*- coding: utf-8 -*-
"""通用小工具：日志、安全数值转换、时间戳。"""

from __future__ import annotations

import datetime as _dt
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

from .config import logs_dir

_LOG_READY = False


def setup_logging() -> Path:
    """初始化日志；任何异常都不得影响主流程。"""
    global _LOG_READY
    log_file = logs_dir() / "app.log"
    if _LOG_READY:
        return log_file
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        fh = logging.handlers.RotatingFileHandler(
            str(log_file), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        if not getattr(sys, "frozen", False):
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        _LOG_READY = True
    except Exception:
        # 日志不可用时静默降级
        pass
    return log_file


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def now_stamp() -> _dt.datetime:
    return _dt.datetime.now()


def report_id(now: Optional[_dt.datetime] = None) -> str:
    now = now or now_stamp()
    return "TS-" + now.strftime("%Y%m%d-%H%M%S")


def parse_roi(text: Optional[str]):
    """解析命令行 --roi x,y,w,h；失败返回 None。"""
    if not text:
        return None
    try:
        parts = [int(float(p.strip())) for p in text.replace("，", ",").split(",")]
        if len(parts) != 4:
            return None
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
    except Exception:
        return None


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"
