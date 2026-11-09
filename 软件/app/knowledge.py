# -*- coding: utf-8 -*-
"""知识库加载与规则匹配。

规则文件 data/knowledge_base.json 是唯一的数据源，方便非程序员修改建议文案。
规则语言很小，只支持：
    in / eq / between / gt / gte / lt / lte / is_true / is_false
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import resource_path


class KnowledgeBaseError(Exception):
    pass


DEFAULT_KB_REL = "data/knowledge_base.json"


@dataclass
class PatternMatch:
    id: str
    name: str
    summary: str
    match: float          # 规则命中权重 / 总权重，0~1
    score: float
    max_score: float
    hits: List[str] = field(default_factory=list)
    diet: List[str] = field(default_factory=list)
    lifestyle: List[str] = field(default_factory=list)
    seek_care: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "match": round(self.match, 4),
            "score": self.score,
            "max_score": self.max_score,
            "hits": list(self.hits),
        }


def load_knowledge_base(path: Optional[str] = None) -> Dict[str, Any]:
    kb_path = Path(path) if path else resource_path(DEFAULT_KB_REL)
    if not kb_path.exists():
        raise KnowledgeBaseError(
            f"知识库文件缺失：{kb_path}\n请重新安装软件，或用 --kb 指定正确的 JSON 文件。"
        )
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise KnowledgeBaseError(f"知识库格式错误：{exc}") from exc

    if not isinstance(kb, dict) or not isinstance(kb.get("patterns"), list):
        raise KnowledgeBaseError("知识库缺少 patterns 字段")
    for p in kb["patterns"]:
        if not isinstance(p, dict) or "id" not in p or "rules" not in p:
            raise KnowledgeBaseError("知识库中的 pattern 结构不完整")
        if not isinstance(p["rules"], list):
            raise KnowledgeBaseError(f"pattern {p.get('id')} 的 rules 必须是数组")
    kb.setdefault("red_flags", [])
    kb.setdefault("body_color_labels", {})
    kb.setdefault("coating_color_labels", {})
    kb.setdefault("moisture_labels", {})
    return kb


def _as_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def match_rule(value: Any, rule: Dict[str, Any]) -> bool:
    """判断单个特征是否命中规则；任何异常都视为未命中。"""
    try:
        op = str(rule.get("op", "in")).lower()
        target = rule.get("value")

        if op == "is_true":
            return value is True
        if op == "is_false":
            return value is False
        if value is None:
            return False
        if op == "eq":
            return value == target
        if op == "in":
            return value in (target or [])
        if op == "between":
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                return False
            num = _as_number(value)
            return num is not None and float(target[0]) <= num <= float(target[1])
        if op in ("gt", "gte", "lt", "lte"):
            num = _as_number(value)
            ref = _as_number(target)
            if num is None or ref is None:
                return False
            if op == "gt":
                return num > ref
            if op == "gte":
                return num >= ref
            if op == "lt":
                return num < ref
            return num <= ref
    except Exception:
        return False
    return False


def evaluate(features: Dict[str, Any], kb: Dict[str, Any]) -> List[PatternMatch]:
    """对全部证候倾向打分，按匹配度降序返回。"""
    results: List[PatternMatch] = []
    for p in kb.get("patterns", []):
        score = 0.0
        max_score = 0.0
        hits: List[str] = []
        for rule in p.get("rules", []):
            try:
                weight = float(rule.get("weight", 1.0))
            except Exception:
                weight = 1.0
            max_score += weight
            if match_rule(features.get(rule.get("feature")), rule):
                score += weight
                note = rule.get("note") or str(rule.get("feature"))
                hits.append(str(note))
        match = (score / max_score) if max_score > 0 else 0.0
        results.append(
            PatternMatch(
                id=str(p.get("id", "")),
                name=str(p.get("name", p.get("id", "未知"))),
                summary=str(p.get("summary", "")),
                match=max(0.0, min(1.0, match)),
                score=score,
                max_score=max_score,
                hits=hits,
                diet=list(p.get("diet", [])),
                lifestyle=list(p.get("lifestyle", [])),
                seek_care=list(p.get("seek_care", [])),
            )
        )
    results.sort(key=lambda m: (m.match, m.score), reverse=True)
    return results
