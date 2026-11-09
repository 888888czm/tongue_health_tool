# -*- coding: utf-8 -*-
"""报告生成：纯文本、HTML（可打印）、JSON。

报告只描述"观察到的征象 + 倾向性参考"，不输出疾病名称，不给出用药方案。
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DISCLAIMER_LONG, VERSION, reports_dir
from .features import TongueFeatures
from .imaging import Segmentation, annotate, encode_preview
from .knowledge import PatternMatch
from .utils import report_id

QUALITY_LABELS = {"high": "良好", "medium": "一般", "low": "较差"}


@dataclass
class Report:
    rid: str
    created_at: datetime
    features: TongueFeatures
    matches: List[PatternMatch]
    kb: Dict[str, Any]
    image_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    seg: Optional[Segmentation] = None
    preview_jpeg: bytes = b""

    # ---------------- 文本 ----------------
    def body_color_label(self) -> str:
        return self.kb.get("body_color_labels", {}).get(
            self.features.body_color, self.features.body_color
        )

    def coating_label(self) -> str:
        return self.kb.get("coating_color_labels", {}).get(
            self.features.coating_color, self.features.coating_color
        )

    def moisture_label(self) -> str:
        return self.kb.get("moisture_labels", {}).get(
            self.features.moisture, self.features.moisture
        )

    def top_matches(self, limit: int = 3, threshold: float = 0.35) -> List[PatternMatch]:
        return [m for m in self.matches if m.match >= threshold][:limit]

    def advice_patterns(self) -> List[PatternMatch]:
        """取匹配度最高、且非"平和"的 1~2 项作为建议来源。"""
        picked = []
        for m in self.matches:
            if m.id == "balanced":
                continue
            if m.match >= 0.55:
                picked.append(m)
            if len(picked) >= 2:
                break
        if not picked:
            picked = [m for m in self.matches if m.id == "balanced"][:1]
        return picked

    def _quality_line(self) -> str:
        q = QUALITY_LABELS.get(self.features.quality, "未知")
        return f"{q}（舌体占取景框 {self.features.area_ratio * 100:.0f}%）"

    def to_text(self) -> str:
        f = self.features
        m = f.metrics
        lines: List[str] = []
        add = lines.append

        add("=" * 62)
        add("            舌象观察与健康参考报告")
        add("=" * 62)
        add(f"报告编号：{self.rid}")
        add(f"生成时间：{self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        add(f"软件版本：{VERSION}")
        add("")
        add("【重要声明】")
        for seg_line in _wrap(DISCLAIMER_LONG, 58):
            add("  " + seg_line)
        add("")
        add("【图像质量】" + self._quality_line())
        if self.warnings:
            for w in self.warnings:
                add("  提示：" + w)
        else:
            add("  提示：图像质量可用于粗略观察。建议在自然光下、未进食染色食物时复测。")
        add("")

        add("【一、观察到的舌象征象】")
        add(f"  1. 舌质颜色：{self.body_color_label()}（估算）")
        add(
            f"     客观指标：L*={m.get('body_L', 0):.0f}，"
            f"a*={m.get('body_A', 0):.0f}，b*={m.get('body_B', 0):.0f}"
        )
        add(f"  2. 舌苔：{self.coating_label()}，覆盖约 {m.get('coating_ratio', 0) * 100:.0f}%")
        if f.coating_color != "none":
            add(f"     苔质：{'偏腻' if f.coating_greasy else '不腻'}")
        add(
            f"  3. 舌体形态：宽长比 {f.width_length_ratio:.2f}"
            f"（常见参考 0.50–0.78）；轮廓规整度 {f.circularity:.2f}"
        )
        add(
            f"  4. 齿痕：{'可见' if f.scalloped else '未见'}"
            f"（检测到 {f.scallop_count} 处边缘凹陷）"
        )
        add(
            f"  5. 裂纹：{'可见' if f.cracks else '未见'}"
            f"（暗线占比 {f.crack_ratio * 100:.2f}%）"
        )
        add(f"  6. 瘀点：{'可见' if f.petechiae else '未见'}")
        add(f"  7. 舌尖/舌边偏红：{'是' if f.peripheral_red else '否'}")
        add(f"  8. 润燥：{self.moisture_label()}（反光面积 {m.get('specular_ratio', 0) * 100:.2f}%）")
        add("")

        add("【二、倾向性分析】（规则匹配度，不是患病概率）")
        tops = self.top_matches()
        if not tops:
            add("  未发现明显倾向。若有不适应以医生面诊为准。")
        else:
            for i, item in enumerate(tops, 1):
                add(f"  {i}. {item.name}    匹配度 {item.match * 100:.0f}%")
                if item.summary:
                    for line in _wrap(item.summary, 54):
                        add("     " + line)
                if item.hits:
                    add("     依据：" + "；".join(item.hits[:5]))
        add("")
        add("  说明：匹配度表示“规则命中的权重占比”，只反映舌象特征与某类")
        add("  描述的接近程度，不代表患病概率，也不能用来排除任何疾病。")
        add("")

        add("【三、生活、饮食与就医建议】")
        picked = self.advice_patterns()
        if not picked:
            add("  暂无可给出的针对性建议，请保持均衡饮食与规律作息。")
        for p in picked:
            add(f"  ● 针对「{p.name}」：")
            if p.diet:
                add("    [饮食]")
                for item in p.diet:
                    for line in _wrap("· " + item, 54):
                        add("      " + line)
            if p.lifestyle:
                add("    [生活]")
                for item in p.lifestyle:
                    for line in _wrap("· " + item, 54):
                        add("      " + line)
            if p.seek_care:
                add("    [何时就医]")
                for item in p.seek_care:
                    for line in _wrap("· " + item, 54):
                        add("      " + line)
        add("")

        add("【四、需要尽快就医的情况（红旗征象）】")
        for item in self.kb.get("red_flags", []):
            for line in _wrap("· " + item, 56):
                add("  " + line)
        add("")

        add("【五、拍摄与复测建议】")
        for item in [
            "在自然光或白光下拍摄，避免美颜、滤镜、暖色灯光和逆光。",
            "拍摄前 30 分钟避免咖啡、浓茶、红酒、蓝莓、火龙果等染色食物。",
            "尽量在晨起刷牙前、未进食时拍摄，结果更有可比性。",
            "每次使用相同的取景框和距离，便于纵向对比。",
        ]:
            for line in _wrap("· " + item, 56):
                add("  " + line)
        add("")

        add("【六、隐私说明】")
        add("  照片与报告仅保存在本机 reports 目录；本软件不联网、不上传任何数据。")
        add("")
        add("【七、免责声明】")
        for line in _wrap(
            "本软件为健康科普与自我观察工具，不是医疗器械，不提供诊断、处方或治疗建议。"
            "报告中的颜色与形态描述由通用图像算法估算，受光线、设备、拍摄角度影响较大，"
            "可能出现偏差。任何健康问题请咨询具备资质的医生。",
            58,
        ):
            add("  " + line)
        add("")
        add("=" * 62)
        return "\n".join(lines)

    # ---------------- HTML ----------------
    def to_html(self) -> str:
        f = self.features
        m = f.metrics
        esc = html.escape
        tops = self.top_matches()
        picked = self.advice_patterns()

        def ul(items: List[str]) -> str:
            return "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>"

        match_rows = ""
        for item in tops:
            pct = max(2, int(round(item.match * 100)))
            match_rows += (
                "<div class='mrow'>"
                f"<div class='mname'>{esc(item.name)}</div>"
                f"<div class='mbar'><span style='width:{pct}%'></span></div>"
                f"<div class='mpct'>{pct}%</div>"
                "</div>"
            )
            if item.hits:
                match_rows += f"<div class='mbasis'>依据：{esc('；'.join(item.hits[:5]))}</div>"

        advice_html = ""
        for p in picked:
            advice_html += f"<h3>{esc(p.name)}</h3>"
            if p.diet:
                advice_html += "<h4>饮食</h4>" + ul(p.diet)
            if p.lifestyle:
                advice_html += "<h4>生活</h4>" + ul(p.lifestyle)
            if p.seek_care:
                advice_html += "<h4>何时就医</h4>" + ul(p.seek_care)

        img_html = ""
        if self.preview_jpeg:
            b64 = base64.b64encode(self.preview_jpeg).decode("ascii")
            img_html = (
                f"<div class='photo'><img src='data:image/jpeg;base64,{b64}' "
                f"alt='舌象照片'/></div>"
            )

        warn_html = ""
        if self.warnings:
            warn_html = "<div class='warn'>" + "".join(
                f"<p>{esc(w)}</p>" for w in self.warnings
            ) + "</div>"

        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>舌象观察与健康参考报告 {esc(self.rid)}</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{ font-family: "Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif;
         color:#222; line-height:1.65; max-width: 820px; margin: 0 auto; padding: 18px; }}
  h1 {{ font-size: 22px; text-align:center; margin: 6px 0 4px; }}
  .sub {{ text-align:center; color:#666; font-size:12px; margin-bottom:14px; }}
  h2 {{ font-size:16px; border-left:5px solid #2f7d6d; padding-left:8px; margin:22px 0 8px; }}
  h3 {{ font-size:14px; margin:14px 0 4px; color:#1f5f53; }}
  h4 {{ font-size:13px; margin:8px 0 2px; color:#444; }}
  table {{ border-collapse: collapse; width:100%; font-size:13px; }}
  th,td {{ border:1px solid #ddd; padding:6px 8px; text-align:left; }}
  th {{ background:#f3f7f6; width:34%; font-weight:600; }}
  .notice {{ background:#fff7e6; border:1px solid #f0d9a8; padding:10px 12px;
             font-size:12.5px; border-radius:6px; }}
  .warn {{ background:#fdecea; border:1px solid #f5c2bd; padding:8px 12px;
           font-size:12.5px; border-radius:6px; margin:8px 0; }}
  .photo {{ text-align:center; margin:10px 0; }}
  .photo img {{ max-width: 320px; border-radius:8px; border:1px solid #ddd; }}
  .mrow {{ display:flex; align-items:center; gap:8px; margin-top:8px; }}
  .mname {{ width:190px; font-size:13px; }}
  .mbar {{ flex:1; background:#eee; height:12px; border-radius:6px; overflow:hidden; }}
  .mbar span {{ display:block; height:100%; background:#2f7d6d; }}
  .mpct {{ width:44px; text-align:right; font-size:13px; }}
  .mbasis {{ font-size:12px; color:#666; margin:2px 0 6px 198px; }}
  ul {{ margin:4px 0 8px 18px; padding:0; font-size:13px; }}
  li {{ margin:3px 0; }}
  .flag li {{ color:#8a1f11; }}
  .foot {{ margin-top:24px; font-size:11.5px; color:#777; border-top:1px dashed #bbb;
           padding-top:10px; }}
  .dim {{ color:#777; font-size:12px; }}
  @media print {{ body {{ padding:0; }} .noprint {{ display:none; }} }}
</style></head><body>
<h1>舌象观察与健康参考报告</h1>
<div class="sub">报告编号 {esc(self.rid)} ｜ {self.created_at.strftime('%Y-%m-%d %H:%M:%S')} ｜ v{esc(VERSION)}</div>

<div class="notice"><b>重要声明：</b>{esc(DISCLAIMER_LONG)}</div>

<h2>图像质量</h2>
<p>{esc(QUALITY_LABELS.get(f.quality, '未知'))}（舌体占取景框 {f.area_ratio * 100:.0f}%）</p>
{warn_html}
{img_html}

<h2>一、观察到的舌象征象</h2>
<table>
<tr><th>舌质颜色（估算）</th><td>{esc(self.body_color_label())}
  <span class="dim">（L*={m.get('body_L',0):.0f}, a*={m.get('body_A',0):.0f}, b*={m.get('body_B',0):.0f}）</span></td></tr>
<tr><th>舌苔</th><td>{esc(self.coating_label())}，覆盖约 {m.get('coating_ratio',0)*100:.0f}%
  {'，苔质偏腻' if f.coating_greasy else ''}</td></tr>
<tr><th>舌体形态</th><td>宽长比 {f.width_length_ratio:.2f}（常见参考 0.50–0.78）；
  轮廓规整度 {f.circularity:.2f}</td></tr>
<tr><th>齿痕</th><td>{'可见' if f.scalloped else '未见'}（{f.scallop_count} 处边缘凹陷）</td></tr>
<tr><th>裂纹</th><td>{'可见' if f.cracks else '未见'}（暗线占比 {f.crack_ratio*100:.2f}%）</td></tr>
<tr><th>瘀点</th><td>{'可见' if f.petechiae else '未见'}</td></tr>
<tr><th>舌尖/舌边偏红</th><td>{'是' if f.peripheral_red else '否'}</td></tr>
<tr><th>润燥</th><td>{esc(self.moisture_label())}（反光面积 {m.get('specular_ratio',0)*100:.2f}%）</td></tr>
</table>

<h2>二、倾向性分析<span class="dim">（规则匹配度，不是患病概率）</span></h2>
{match_rows or '<p>未发现明显倾向。</p>'}
<p class="dim">匹配度只表示舌象特征与某类描述的接近程度，不代表患病概率，也不能排除任何疾病。</p>

<h2>三、生活、饮食与就医建议</h2>
{advice_html or '<p>请保持均衡饮食与规律作息。</p>'}

<h2>四、需要尽快就医的情况（红旗征象）</h2>
<ul class="flag">{''.join(f'<li>{esc(x)}</li>' for x in self.kb.get('red_flags', []))}</ul>

<h2>五、拍摄与复测建议</h2>
<ul>
<li>在自然光或白光下拍摄，避免美颜、滤镜、暖色灯光和逆光。</li>
<li>拍摄前 30 分钟避免咖啡、浓茶、红酒、蓝莓、火龙果等染色食物。</li>
<li>尽量在晨起刷牙前、未进食时拍摄，结果更有可比性。</li>
<li>每次使用相同的取景框和距离，便于纵向对比。</li>
</ul>

<h2>六、隐私与免责</h2>
<p>照片与报告仅保存在本机 reports 目录；本软件不联网、不上传任何数据。</p>
<p class="dim">本软件为健康科普与自我观察工具，不是医疗器械，不提供诊断、处方或治疗建议。
颜色与形态描述由通用图像算法估算，受光线、设备、拍摄角度影响较大，可能出现偏差。
任何健康问题请咨询具备资质的医生。</p>

<div class="foot">舌象健康观察助手 v{esc(VERSION)} ｜ 本报告由程序自动生成</div>
</body></html>"""

    # ---------------- JSON ----------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.rid,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "version": VERSION,
            "image_path": self.image_path,
            "quality": self.features.quality,
            "warnings": self.warnings,
            "features": self.features.to_dict(),
            "matches": [m.as_dict() for m in self.matches],
            "disclaimer": DISCLAIMER_LONG,
        }

    # ---------------- 保存 ----------------
    def save(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        target = Path(out_dir) if out_dir else reports_dir()
        target.mkdir(parents=True, exist_ok=True)
        base = target / self.rid
        txt_path = base.with_suffix(".txt")
        html_path = base.with_suffix(".html")
        json_path = base.with_suffix(".json")

        txt_path.write_text(self.to_text(), encoding="utf-8")
        html_path.write_text(self.to_html(), encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"txt": str(txt_path), "html": str(html_path), "json": str(json_path)}


def _wrap(text: str, width: int) -> List[str]:
    """按显示宽度（中文算 2）简单折行。"""
    if not text:
        return [""]
    lines: List[str] = []
    cur = ""
    cur_w = 0
    for ch in text:
        w = 2 if ord(ch) > 127 else 1
        if cur_w + w > width and cur:
            lines.append(cur)
            cur = ""
            cur_w = 0
        cur += ch
        cur_w += w
    if cur:
        lines.append(cur)
    return lines


def build_report(
    features: TongueFeatures,
    matches: List[PatternMatch],
    kb: Dict[str, Any],
    seg: Optional[Segmentation] = None,
    image: Any = None,
    image_path: Optional[str] = None,
) -> Report:
    now = datetime.now()
    preview = b""
    if image is not None:
        try:
            preview = encode_preview(annotate(image, seg))
        except Exception:
            preview = b""
    return Report(
        rid=report_id(now),
        created_at=now,
        features=features,
        matches=matches,
        kb=kb,
        image_path=image_path,
        warnings=list(seg.warnings) if seg else [],
        seg=seg,
        preview_jpeg=preview,
    )
