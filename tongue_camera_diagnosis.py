# -*- coding: utf-8 -*-
"""
中医舌诊摄像头辅助分析（知识科普版 · 原创CV实现）
功能：调用电脑摄像头采集舌象，提取舌质颜色、舌体形态、舌苔特征，
      输出证型倾向、经典依据、生活饮食建议及食疗汤方。
声明：本程序不构成医疗诊断，不能替代执业医师面诊。
打包：pyinstaller --onefile --windowed --name 舌诊摄像头辅助 tongue_camera_diagnosis.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import textwrap

# 尝试导入 OpenCV 和 Pillow，失败时给出明确提示
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# ==================== 特征标签字典 ====================
COLOR_LABELS = {
    "pale": "淡白",
    "pink": "淡红（大致正常）",
    "red": "红绛",
    "purple": "青紫",
}

SHAPE_LABELS = {
    "normal": "正常",
    "teeth_mark": "胖大齿痕",
    "thin": "瘦小",
    "cracked": "裂纹倾向",
}

COATING_LABELS = {
    "thin_white": "薄白",
    "white_thick": "白厚腻",
    "yellow_thick": "黄厚腻",
    "none": "少苔或无苔",
}

# ==================== 规则库（原创整理） ====================
RULES = [
    {
        "match": {"color": "pale", "shape": "teeth_mark", "coating": "white_thick"},
        "syndrome": "脾虚湿盛，寒湿困脾",
        "report": (
            "舌质淡白提示气血不足或阳气偏弱；舌体胖大伴齿痕，多为脾虚运化无力、水湿内停；"
            "白厚腻苔进一步提示寒湿困阻中焦。常见于平素怕冷、四肢不温、大便偏溏、身体困重、"
            "食欲不振或食后腹胀的人群。"
        ),
        "classic": "《黄帝内经·灵枢》言“足太阴之正……贯舌中”，舌中候脾胃；《本草纲目》口舌篇载“舌淡，白术、半夏、生姜、茯苓”，以健脾燥湿、温中行水为法。",
        "lifestyle": "避免久居潮湿环境，注意腹部与足部保暖；规律作息，避免熬夜耗伤脾气；适度运动以助阳气升发，但不宜大汗淋漓。",
        "diet": "宜温性、易消化食物；忌生冷瓜果、冷饮、肥甘厚味。可适量食用山药、茯苓、白扁豆、生姜、大枣、陈皮。",
        "recipes": [
            {
                "name": "赤小豆薏苡仁山药粥",
                "ingredients": "赤小豆150克，薏苡仁100克，山药50克，红糖20克（糖尿病者不放）。",
                "method": "赤小豆、薏苡仁洗净，清水浸泡3小时；山药切小块。三者入砂锅，武火煮沸后转文火熬40分钟至烂熟，加红糖续煮5分钟。",
                "note": "健脾祛湿、益气和胃。薏苡仁性偏凉，寒湿重者可将薏苡仁炒至微黄再用。",
            },
            {
                "name": "冬瓜薏苡仁山药生姜汤",
                "ingredients": "冬瓜（连皮）适量，薏苡仁30克，山药30克，生姜3片。",
                "method": "冬瓜洗净切块，与薏苡仁、山药、生姜同煮成汤，少盐调味。",
                "note": "利水渗湿、温中健脾，适合白厚腻苔、身体困重者日常食用。",
            },
        ],
    },
    {
        "match": {"color": "red", "shape": ["thin", "cracked"], "coating": "none"},
        "syndrome": "阴虚火旺（或胃阴不足）",
        "report": (
            "舌红绛主热证，舌体瘦小或见裂纹提示阴血亏耗、舌失濡养；少苔或无苔进一步说明胃阴或肾阴不足。"
            "常见于口干咽燥、手足心热、夜间盗汗、心烦失眠、大便偏干的人群。"
        ),
        "classic": "《黄帝内经·素问》云“心主舌”“南方生热……在窍为舌”，红绛多从心、胃、肾阴亏论治；《本草纲目》口舌篇载“舌衄，生地黄同阿胶末，米饮服”，生地黄为养阴清热常用之品。",
        "lifestyle": "避免熬夜与过度劳累，节制辛辣燥热饮食；保持情绪平稳，避免急躁动火；可练习静坐、深呼吸等放松方式。",
        "diet": "宜滋阴润燥之品，如银耳、百合、雪梨、莲藕、黑芝麻、桑葚、鸭肉。忌辣椒、花椒、烧烤、酒类及温补峻补之品。",
        "recipes": [
            {
                "name": "沙参麦冬猪龙骨汤",
                "ingredients": "生地、沙参、麦冬、枸杞各9克，石斛6克，当归3克，猪龙骨一节。",
                "method": "猪龙骨焯水后与诸药同入锅，加水适量，文火煲1.5—2小时，少盐调味。",
                "note": "养阴生津、清虚热。脾胃虚寒、大便溏薄者不宜过量。",
            },
            {
                "name": "石斛麦冬茶",
                "ingredients": "石斛5克，麦冬10克。",
                "method": "沸水冲泡，焖10—15分钟后代茶饮。",
                "note": "养阴生津、润燥止渴。",
            },
        ],
    },
    {
        "match": {"color": "red", "shape": "teeth_mark", "coating": "yellow_thick"},
        "syndrome": "湿热内蕴，脾虚湿阻",
        "report": (
            "舌红提示有热，胖大齿痕提示脾虚湿盛，黄厚腻苔为湿热交阻之象。常见于口苦口干、身体困重、"
            "大便黏滞不爽、面部易出油或易生痤疮的人群，长期饮酒、嗜食肥甘厚味者尤为多见。"
        ),
        "classic": "《黄帝内经》论舌中候脾胃；《本草纲目》载“舌苦是胆热”“舌酸是湿热”，黄厚腻苔多从脾胃湿热、肝胆郁热论治。",
        "lifestyle": "务必减少饮酒，避免熬夜；增加运动以助湿邪外散；保持居所通风干燥。",
        "diet": "宜清淡、偏凉而利湿之品，如冬瓜、赤小豆、薏苡仁、绿豆、苦瓜、芹菜。忌酒类、油炸、烧烤、甜腻糕点及辛辣刺激。",
        "recipes": [
            {
                "name": "绵茵陈赤小豆鲫鱼汤",
                "ingredients": "绵茵陈15—20克，赤小豆30克，鲫鱼1条，陈皮3克，生姜3片。",
                "method": "鲫鱼煎至微黄；绵茵陈、赤小豆、陈皮洗净。全部入锅，加水适量，文火煲1—1.5小时，少盐调味。",
                "note": "清热利湿、健脾和胃，适合黄厚腻苔、湿热偏盛者。茵陈用量不宜过大。",
            },
            {
                "name": "苦瓜黄豆排骨汤",
                "ingredients": "苦瓜1条，黄豆100克（提前浸泡），排骨500克，生姜2片。",
                "method": "排骨焯水，与泡好的黄豆、苦瓜块、生姜同煲1.5—2小时，少盐调味。",
                "note": "清热解暑、利湿健脾。苦瓜性寒，脾胃虚寒者不宜多食。",
            },
        ],
    },
    {
        "match": {"color": "pale", "shape": "thin", "coating": ["thin_white", "none"]},
        "syndrome": "气血两虚（偏血虚或气血不足）",
        "report": "舌质淡白主气血亏虚或虚寒；舌体瘦小提示阴血不足、舌体失充。常见于面色萎黄或苍白、头晕心悸、气短乏力、女性经量偏少或色淡的人群。",
        "classic": "《黄帝内经》论舌与心、脾经络相连，气血充盛则舌色淡红润泽；《本草纲目》口舌篇“舌淡，白术、半夏、生姜、茯苓”，偏气虚者以健脾益气为要。",
        "lifestyle": "避免过度劳累与节食减肥；保证充足睡眠；适度进行缓和运动以促进气血运行。",
        "diet": "宜益气养血之品，如红枣、桂圆、山药、小米、瘦肉、猪肝、乌鸡（当归少量入汤）。忌生冷、寒凉及过度耗气之物。",
        "recipes": [
            {
                "name": "当归补血汤（简化食疗版）",
                "ingredients": "当归10克，黄芪30—50克，白鸽1只或乌鸡半只。",
                "method": "药材洗净，与焯水后的禽肉同入锅，加水适量，文火煲1.5—2小时，少盐调味。",
                "note": "益气生血。感冒发热、湿热内盛时不宜；服用前建议咨询医师。",
            },
            {
                "name": "参枣乌鸡汤",
                "ingredients": "党参15克，红枣6枚，乌鸡半只，生姜2片。",
                "method": "乌鸡焯水，与党参、红枣、生姜同煲1.5小时，少盐调味。",
                "note": "健脾益气、养血安神，适合气血不足、容易疲劳者日常调养。",
            },
        ],
    },
    {
        "match": {"color": "purple", "shape": ["teeth_mark", "normal"], "coating": ["white_thick", "thin_white"]},
        "syndrome": "寒凝血瘀（或阳虚血瘀）",
        "report": "全舌青紫多主寒证或血瘀。寒性收引，气血运行不畅，舌络瘀阻而见青紫；若伴舌体胖大、苔白，多兼阳虚水湿不化。常见于畏寒肢冷、局部疼痛（遇寒加重）、女性痛经或经色暗有血块的人群。",
        "classic": "《黄帝内经》重视“血气者，喜温而恶寒，寒则泣不能流”；《本草纲目》口舌篇对舌部瘀滞多从行气活血、温通经络论治。",
        "lifestyle": "注意全身保暖，尤其是腰腹与下肢；避免久坐不动；可适度热敷或温水泡脚。",
        "diet": "宜温通活血之品，如生姜、肉桂（少量）、山楂、红糖、玫瑰花、桃仁（少量，入膳）。忌生冷寒凉及收涩之物。",
        "recipes": [
            {
                "name": "鲤鱼肉桂葱姜汤",
                "ingredients": "鲤鱼1尾（约500—1000克），肉桂3—5克，葱白3根，生姜5片。",
                "method": "鲤鱼处理干净，将肉桂、葱白、生姜放入鱼腹，入锅加水，武火煮沸后转文火炖至鱼熟汤浓。",
                "note": "温阳散寒、益气利水。肉桂用量宜小，阴虚火旺、孕妇慎用。",
            },
            {
                "name": "山楂红糖生姜茶",
                "ingredients": "山楂干10克，生姜3片，红糖适量。",
                "method": "沸水冲泡或短时煎煮，代茶饮。",
                "note": "温中散寒、活血化瘀。胃酸过多者山楂减量；糖尿病患者去红糖。",
            },
        ],
    },
    {
        "match": {"color": "pink", "shape": "normal", "coating": "thin_white"},
        "syndrome": "大致正常舌象（气血调和，脾胃功能良好）",
        "report": "舌质淡红、舌体大小适中、薄白苔，为中医典型的正常舌象，提示气血调和、脏腑功能大致正常。",
        "classic": "《黄帝内经》以“淡红润泽”为舌之常色；正常舌象是“有胃气、有神”的表现之一。",
        "lifestyle": "保持规律作息、均衡饮食与适度运动，无需特殊调补。",
        "diet": "均衡膳食即可，不必刻意进补；避免长期偏食或过度食用单一性味的食物。",
        "recipes": [
            {
                "name": "淮山瘦肉汤（平和调养）",
                "ingredients": "鲜山药200克，瘦肉200克，生姜2片，枸杞少许。",
                "method": "瘦肉焯水，与山药块、生姜同煲1—1.5小时，出锅前加枸杞，少盐调味。",
                "note": "平补脾胃，性质平和，适合正常舌象者日常保健食用。",
            }
        ],
    },
]

FALLBACK = {
    "pale": {
        "syndrome": "气血不足或虚寒倾向（建议结合其他症状综合判断）",
        "report": "舌质淡白提示气血偏弱或阳气不足。建议结合是否怕冷、乏力、头晕、食欲差等表现综合评估。",
        "classic": "《黄帝内经》重视气血与舌色的关系；《本草纲目》对舌淡多从健脾益气论治。",
        "lifestyle": "注意保暖，避免过度劳累与节食。",
        "diet": "可适量食用红枣、山药、小米、瘦肉等平和益气之品；避免长期生冷饮食。",
        "recipes": [],
    },
    "pink": {
        "syndrome": "舌色大致正常，需结合舌形与舌苔进一步判断",
        "report": "淡红舌本身不提示明显偏颇，请重点参考舌体形态与舌苔的匹配结果。",
        "classic": "淡红舌为常色，中医以“淡红润泽”为平人之舌。",
        "lifestyle": "保持当前良好习惯即可。",
        "diet": "均衡饮食，无需特殊调整。",
        "recipes": [],
    },
    "red": {
        "syndrome": "热证倾向（实热或虚热，需结合舌形舌苔分辨）",
        "report": "红绛舌主热。若舌体不瘦、苔黄厚，偏实热或湿热；若舌瘦、少苔，偏阴虚火旺。",
        "classic": "《黄帝内经》“心主舌”，红绛多从心、胃、肝热论治。",
        "lifestyle": "避免熬夜、辛辣与酒类。",
        "diet": "可适量食用绿豆、冬瓜、梨、莲藕等清热之品；虚热者宜滋阴而非苦寒。",
        "recipes": [],
    },
    "purple": {
        "syndrome": "寒证或血瘀倾向",
        "report": "青紫舌多与寒凝或血行不畅有关，需结合舌形、舌苔及全身症状综合判断。",
        "classic": "《黄帝内经》有“寒则泣不能流”之论，青紫多从寒、瘀论治。",
        "lifestyle": "注意保暖，避免久坐。",
        "diet": "可少量食用生姜、山楂、玫瑰花等温通之品；避免生冷。",
        "recipes": [],
    },
}


# ==================== 图像特征提取 ====================
def extract_features(frame_bgr):
    """
    从 BGR 帧中提取舌体区域，返回 (color_key, shape_key, coating_key, debug_mask)。
    完全基于 HSV 阈值 + 形态学 + 轮廓几何，不依赖深度学习或专利分割方案。
    """
    h, w = frame_bgr.shape[:2]
    # 取画面中央区域，引导用户将舌头置于中央
    y0, y1 = int(h * 0.15), int(h * 0.85)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    roi = frame_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 舌体在 HSV 中的经验性双区间（红色/粉红色）
    # 区间1: H ∈ [0, 15]，区间2: H ∈ [165, 180]；S>60，V>50
    mask1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学净化：闭运算补洞，开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 寻找最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, mask

    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    roi_area = roi.shape[0] * roi.shape[1]
    if area < roi_area * 0.03:   # 面积太小，可能未检测到舌体
        return None, None, None, mask

    # ---- 舌质颜色：取主轮廓内部 HSV 中位数 ----
    mask_main = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_main, [main], -1, 255, thickness=cv2.FILLED)
    # 轻微腐蚀，避开边缘阴影
    mask_inner = cv2.erode(mask_main, kernel, iterations=2)
    if cv2.countNonZero(mask_inner) < 50:
        mask_inner = mask_main

    h_vals = H[mask_inner > 0]
    s_vals = S[mask_inner > 0]
    v_vals = V[mask_inner > 0]
    if len(h_vals) == 0:
        return None, None, None, mask

    h_med = float(np.median(h_vals))
    s_med = float(np.median(s_vals))
    v_med = float(np.median(v_vals))

    # 颜色分类（简化经验规则）
    if v_med < 80:
        color_key = "purple"          # 偏暗、偏青紫
    elif h_med > 170 or h_med < 5:
        color_key = "red" if s_med > 120 else "pink"
    elif 5 <= h_med <= 15:
        color_key = "pink" if s_med < 130 else "red"
    else:
        color_key = "pink"

    # 额外：如果整体亮度低且饱和度偏高，倾向青紫
    if v_med < 70 and s_med > 100:
        color_key = "purple"

    # ---- 舌体形态：面积、长宽比、凸包缺陷 ----
    peri = cv2.arcLength(main, True)
    x, y, bw, bh = cv2.boundingRect(main)
    aspect = bw / max(bh, 1)
    area_ratio = area / roi_area

    # 凸包缺陷：检测边缘的周期性凹陷 → 齿痕倾向
    hull = cv2.convexHull(main, returnPoints=False)
    teeth_score = 0
    try:
        defects = cv2.convexityDefects(main, hull)
        if defects is not None:
            deep_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0
                if depth > 8:          # 凹陷深度阈值（像素）
                    deep_count += 1
            if deep_count >= 3:
                teeth_score = 1
    except Exception:
        pass

    if teeth_score or (area_ratio > 0.18 and aspect > 0.9):
        shape_key = "teeth_mark"
    elif area_ratio < 0.06 and aspect < 0.8:
        shape_key = "thin"
    else:
        # 裂纹倾向：检测内部亮度不均或边缘复杂度
        # 简化：若轮廓周长/面积比偏大，提示边缘复杂
        compactness = (peri * peri) / (4 * np.pi * area + 1e-6)
        shape_key = "cracked" if compactness > 2.2 and area_ratio < 0.14 else "normal"

    # ---- 舌苔特征：亮度与饱和度分布 ----
    # 白厚腻：亮度高、饱和度低；黄厚腻：H 偏黄、S 偏高；少苔：亮度低且纹理少
    # 这里用内区统计做简化判断
    h_mean = float(np.mean(h_vals))
    s_mean = float(np.mean(s_vals))
    v_mean = float(np.mean(v_vals))

    # 黄苔倾向：H 在 15~35 附近（黄色）
    yellow_ratio = np.sum((h_vals >= 15) & (h_vals <= 35)) / max(len(h_vals), 1)
    high_v_ratio = np.sum(v_vals > 170) / max(len(v_vals), 1)
    low_v_ratio = np.sum(v_vals < 70) / max(len(v_vals), 1)

    if yellow_ratio > 0.25 and s_mean > 80:
        coating_key = "yellow_thick"
    elif v_mean > 150 and s_mean < 100 and high_v_ratio > 0.3:
        coating_key = "white_thick"
    elif v_mean < 75 and low_v_ratio > 0.3:
        coating_key = "none"
    else:
        coating_key = "thin_white"

    return color_key, shape_key, coating_key, mask


# ==================== 报告生成 ====================
def diagnose(color_key, shape_key, coating_key):
    for rule in RULES:
        m = rule["match"]
        c_ok = (m["color"] == color_key)
        s_match = m["shape"]
        s_ok = (shape_key in s_match) if isinstance(s_match, list) else (s_match == shape_key)
        t_match = m["coating"]
        t_ok = (coating_key in t_match) if isinstance(t_match, list) else (t_match == coating_key)
        if c_ok and s_ok and t_ok:
            return rule
    fb = dict(FALLBACK[color_key])
    return fb


def build_report(color_key, shape_key, coating_key):
    r = diagnose(color_key, shape_key, coating_key)
    lines = []
    lines.append("=" * 52)
    lines.append("        中医舌诊摄像头辅助分析报告")
    lines.append("=" * 52)
    lines.append("")
    lines.append("【舌象采集】")
    lines.append(f"  舌质：{COLOR_LABELS.get(color_key, color_key)}")
    lines.append(f"  舌体：{SHAPE_LABELS.get(shape_key, shape_key)}")
    lines.append(f"  舌苔：{COATING_LABELS.get(coating_key, coating_key)}")
    lines.append("")
    lines.append("【证型倾向】")
    lines.append(f"  {r['syndrome']}")
    lines.append("")
    lines.append("【病理报告】")
    for seg in textwrap.wrap(r["report"], width=46):
        lines.append(f"  {seg}")
    lines.append("")
    lines.append("【经典依据】")
    for seg in textwrap.wrap(r["classic"], width=46):
        lines.append(f"  {seg}")
    lines.append("")
    lines.append("【生活注意】")
    for seg in textwrap.wrap(r["lifestyle"], width=46):
        lines.append(f"  {seg}")
    lines.append("")
    lines.append("【饮食宜忌】")
    for seg in textwrap.wrap(r["diet"], width=46):
        lines.append(f"  {seg}")
    lines.append("")

    if r.get("recipes"):
        lines.append("【推荐食疗方】")
        for i, rec in enumerate(r["recipes"], 1):
            lines.append(f"  {i}. {rec['name']}")
            lines.append(f"     材料：{rec['ingredients']}")
            lines.append(f"     做法：{rec['method']}")
            lines.append(f"     说明：{rec['note']}")
            lines.append("")
    else:
        lines.append("【推荐食疗方】")
        lines.append("  当前组合无预置食疗方，建议咨询中医师进行个体化调养。")
        lines.append("")

    lines.append("-" * 52)
    lines.append("⚠️ 免责声明：本报告仅为中医舌诊知识科普与健康自测参考，")
    lines.append("不构成医疗诊断或治疗建议，不能替代执业医师面诊。")
    lines.append("使用任何食疗方或中药前，请咨询执业中医师或营养师。")
    lines.append("孕妇、儿童、慢性病患者及服药者尤须谨慎。")
    lines.append("-" * 52)
    return "\n".join(lines)


# ==================== 图形界面 ====================
class TongueCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中医舌诊摄像头辅助分析（知识科普版）")
        self.root.geometry("900x720")
        self.root.minsize(820, 640)

        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame = None
        self.last_mask = None
        self._photo = None

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 顶部标题
        ttk.Label(main, text="中医舌诊摄像头辅助分析", font=("Microsoft YaHei", 15, "bold")).pack(pady=(0, 2))
        ttk.Label(
            main,
            text="请将舌头置于画面中央自然光下，点击“拍照分析”",
            font=("Microsoft YaHei", 9),
            foreground="#666666",
        ).pack(pady=(0, 8))

        # 视频与报告左右分栏
        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.video_label = tk.Label(left, bg="#1a1a1a", text="摄像头未开启", fg="#888888",
                                    font=("Microsoft YaHei", 11))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 按钮栏
        btn_bar = ttk.Frame(left)
        btn_bar.pack(fill=tk.X, pady=(8, 0))

        self.btn_open = ttk.Button(btn_bar, text="开启摄像头", command=self.open_camera)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_capture = ttk.Button(btn_bar, text="拍照分析", command=self.capture_and_analyze,
                                      state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_close = ttk.Button(btn_bar, text="关闭摄像头", command=self.close_camera,
                                    state=tk.DISABLED)
        self.btn_close.pack(side=tk.LEFT)

        right = ttk.LabelFrame(body, text="分析报告", padding=4)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.output = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=("Microsoft YaHei", 10),
            relief=tk.FLAT, bg="#fbfbf8", padx=8, pady=8,
        )
        self.output.pack(fill=tk.BOTH, expand=True)
        self._set_output("点击“开启摄像头”后，将舌头置于画面中央。\n\n"
                         "本程序为中医舌诊知识科普工具，不构成医疗诊断。")

        if not CV_AVAILABLE:
            messagebox.showwarning("依赖缺失",
                                   "未检测到 OpenCV 或 Pillow。\n"
                                   "请先执行：pip install opencv-python pillow numpy\n"
                                   "然后再运行本程序。")
            self.btn_open.config(state=tk.DISABLED)

    def _set_output(self, text):
        self.output.config(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.config(state=tk.DISABLED)

    # ---------- 摄像头控制 ----------
    def open_camera(self):
        if not CV_AVAILABLE:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("无法打开摄像头", "请检查摄像头是否被其他程序占用，或尝试更换索引（1、2）。")
            self.cap = None
            return
        self.running = True
        self.btn_open.config(state=tk.DISABLED)
        self.btn_capture.config(state=tk.NORMAL)
        self.btn_close.config(state=tk.NORMAL)
        self.thread = threading.Thread(target=self._video_loop, daemon=True)
        self.thread.start()

    def _video_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.last_frame = frame.copy()
            try:
                color_key, shape_key, coating_key, mask = extract_features(frame)
                self.last_mask = mask
                if color_key:
                    text = f"检测：{COLOR_LABELS.get(color_key, color_key)} · " \
                           f"{SHAPE_LABELS.get(shape_key, shape_key)} · " \
                           f"{COATING_LABELS.get(coating_key, coating_key)}"
                    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                pass

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((480, 420), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._photo = photo
            try:
                self.root.after(0, self._update_video_label, photo)
            except Exception:
                break

    def _update_video_label(self, photo):
        if self.running:
            self.video_label.config(image=photo, text="")

    def capture_and_analyze(self):
        if self.last_frame is None:
            messagebox.showinfo("提示", "尚未采集到画面，请确认摄像头已开启并正对舌头。")
            return
        try:
            color_key, shape_key, coating_key, mask = extract_features(self.last_frame)
            if color_key is None:
                messagebox.showwarning("未检测到舌体",
                                       "画面中未找到足够大的舌体区域。\n"
                                       "请确保舌头置于画面中央，光线充足，避免背景中出现大面积红色物体。")
                return
            report = build_report(color_key, shape_key, coating_key)
            self._set_output(report)
        except Exception as e:
            messagebox.showerror("分析失败", f"处理图像时发生错误：\n{e}")

    def close_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_open.config(state=tk.NORMAL)
        self.btn_capture.config(state=tk.DISABLED)
        self.btn_close.config(state=tk.DISABLED)
        self.video_label.config(image="", text="摄像头已关闭")
        self._photo = None


# ==================== 入口 ====================
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TongueCameraApp(root)
        root.mainloop()
    except Exception as e:
        try:
            messagebox.showerror("程序错误", f"发生未预期错误：\n{e}")
        except Exception:
            pass