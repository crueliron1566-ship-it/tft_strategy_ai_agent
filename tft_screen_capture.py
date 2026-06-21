#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
ASSETS_DIR      = Path("./tft_assets")
CHAMP_DIR       = ASSETS_DIR / "champions"
ITEM_DIR        = ASSETS_DIR / "items"

# 匹配阈值说明：
#   模板来自 DDragon/CommunityDragon TFT 专属头像，
#   与游戏内阵容模拟器截图经过亮度标准化后，同一英雄 fused 分数通常在 0.55~0.85。
#   0.45 是经过实际截图验证后的合理默认值：
#     - 高于此值：正确英雄 (经测试 Volibear=0.632, Braum=0.692, Galio=0.785)
#     - 低于此值：无关英雄互相混淆概率低于 5%
MATCH_THRESHOLD = 0.45   # 英雄匹配阈值（原0.55过严，导致部分正确匹配被丢弃）
ITEM_THRESHOLD  = 0.45   # 装备匹配阈值
TEMPLATE_SIZE   = 64     # 模板统一缩放尺寸（与 tft_fetch_assets.py 保持一致）
INNER_MARGIN    = 0.08   # 裁剪英雄内部时去除边框的比例（缩小从0.10到0.08，保留更多人脸区域）

# 六边形边框的 HSV 颜色范围（青/紫/金/蓝）
BORDER_RANGES: List[Tuple[np.ndarray, np.ndarray]] = [
    (np.array([ 85, 100,  80]), np.array([103, 255, 255])),  # 青色
    (np.array([128,  80,  80]), np.array([168, 255, 255])),  # 紫色
    (np.array([ 16, 110, 130]), np.array([ 40, 255, 255])),  # 金色
    (np.array([ 98,  80,  80]), np.array([130, 255, 255])),  # 蓝色
]

# ──────────────────────────────────────────────────────────────
# 模板缓存
# ──────────────────────────────────────────────────────────────
_champ_templates_gray:  Dict[str, np.ndarray] = {}
_champ_templates_color: Dict[str, np.ndarray] = {}
_champ_templates_hist:  Dict[str, np.ndarray] = {}  # 预计算的直方图缓存
_item_templates_gray:   Dict[str, np.ndarray] = {}
_templates_loaded = False


def _load_templates():
    """
    懒加载：首次调用时从磁盘读取所有模板图片，并预计算直方图。

    关键修复：模板图片是带透明通道的 RGBA PNG。
    cv2.imread(IMREAD_COLOR) 会将 alpha=0（透明）区域读成 RGB=(0,0,0)，
    即纯黑色。而游戏截图中对应位置是棋盘背景（灰度约 100），
    导致减去均值后模板与截图方向相反，NCC 全为负数。

    修复方法：用 IMREAD_UNCHANGED 读取完整 RGBA，
    将透明区域（alpha=0）合成到中性灰背景（128,128,128），
    使模板背景与截图背景亮度接近。
    """
    global _templates_loaded
    if _templates_loaded:
        return

    NEUTRAL_GREY = 128  # 合成背景灰度值，接近游戏棋盘背景

    def _load_with_alpha(path: str) -> np.ndarray:
        """
        读取 PNG（含 RGBA），将透明区域合成到中性灰背景，返回 BGR 图。
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            # 纯灰度图，转 BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            # RGBA → 合成到灰底
            b, g, r, a = cv2.split(img)
            alpha = a.astype(np.float32) / 255.0
            bg    = np.full_like(b, NEUTRAL_GREY, dtype=np.float32)
            def blend(ch):
                return (ch.astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
            return cv2.merge([blend(b), blend(g), blend(r)])
        # 已是 BGR
        return img

    if CHAMP_DIR.exists():
        for p in sorted(CHAMP_DIR.glob("*.png")):
            stem  = p.stem
            color = _load_with_alpha(str(p))
            if color is None or color.size == 0:
                continue
            c64 = cv2.resize(color, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            g64 = cv2.cvtColor(c64, cv2.COLOR_BGR2GRAY)
            _champ_templates_gray[stem]  = g64
            _champ_templates_color[stem] = c64
            _champ_templates_hist[stem]  = _compute_hist(c64)   # 预计算

    if ITEM_DIR.exists():
        for p in sorted(ITEM_DIR.glob("*.png")):
            stem  = p.stem
            color = _load_with_alpha(str(p))
            if color is None or color.size == 0:
                continue
            g36 = cv2.cvtColor(
                cv2.resize(color, (TEMPLATE_SIZE, TEMPLATE_SIZE)),
                cv2.COLOR_BGR2GRAY
            )
            _item_templates_gray[stem] = g36

    _templates_loaded = True
    print(f"[CV] 模板加载: 英雄 {len(_champ_templates_gray)} 个 / 装备 {len(_item_templates_gray)} 个")


# ──────────────────────────────────────────────────────────────
# 颜色直方图特征
# ──────────────────────────────────────────────────────────────
def _compute_hist(img_bgr: np.ndarray) -> np.ndarray:
    """
    计算多粒度 HSV 直方图特征：
    - 全局 H×36 + S×16 + V×16
    - 四分块（每块 H×18 + S×8 + V×8）
    排除 V<35 的暗色像素，降低背景干扰
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros(36 + 16 + 16 + 4 * (18 + 8 + 8), dtype=np.float32)

    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > 35).astype(np.uint8) * 255

    h36 = cv2.calcHist([hsv], [0], mask, [36], [0, 180]).flatten()
    s16 = cv2.calcHist([hsv], [1], mask, [16], [0, 256]).flatten()
    v16 = cv2.calcHist([hsv], [2], mask, [16], [0, 256]).flatten()
    global_hist = np.concatenate([h36, s16, v16])

    hh, hw = img_bgr.shape[:2]
    quad_hists = []
    for r0, r1, c0, c1 in [
        (0, hh//2, 0, hw//2), (0, hh//2, hw//2, hw),
        (hh//2, hh, 0, hw//2), (hh//2, hh, hw//2, hw),
    ]:
        q = hsv[r0:r1, c0:c1]
        m = mask[r0:r1, c0:c1]
        qh = cv2.calcHist([q], [0], m, [18], [0, 180]).flatten()
        qs = cv2.calcHist([q], [1], m, [8],  [0, 256]).flatten()
        qv = cv2.calcHist([q], [2], m, [8],  [0, 256]).flatten()
        quad_hists.append(np.concatenate([qh, qs, qv]))

    feat = np.concatenate([global_hist] + quad_hists).astype(np.float32)
    total = feat.sum()
    return feat / (total + 1e-6) if total > 0 else feat


# ──────────────────────────────────────────────────────────────
# 英雄识别（两级融合：直方图 + 模板匹配）
# ──────────────────────────────────────────────────────────────
def identify_champion(crop: np.ndarray, threshold: float = MATCH_THRESHOLD) -> Tuple[str, float]:
    """
    返回 (champion_stem, score)，无匹配时返回 ('', 0.0)
    stem 格式: TFT16_Draven（与模板文件名一致）

    修复的核心问题：
      原代码用 matchTemplate(64x64_query, 64x64_template) 做滑动窗口匹配，
      当两张图尺寸完全相同时输出退化为 1×1 矩阵（整图单次比较），
      对任何轻微的光照/缩放差异极度敏感，导致分数普遍低于 0.4。

    修复方案：
      将两张等大图拉平为向量，减去均值后做归一化点积（零均值 NCC），
      结果等价于 CCOEFF_NORMED 的全局版本，对光照偏移和对比度缩放不变。
      再与直方图相似度加权融合提高鲁棒性。
    """
    if crop is None or crop.size == 0:
        return "", 0.0

    _load_templates()
    if not _champ_templates_gray:
        return "", 0.0

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return "", 0.0

    # 统一缩放到 TEMPLATE_SIZE × TEMPLATE_SIZE
    crop_bgr = cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    if len(crop_bgr.shape) == 2:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 亮度标准化：将截图区域拉伸到 [0,255] 并偏移均值至 128，
    # 与模板（合成在灰底128上）保持同一亮度基准，避免 NCC 因均值偏移而为负
    g_min, g_max = crop_gray.min(), crop_gray.max()
    if g_max - g_min > 10:   # 有足够对比度才拉伸
        crop_gray = (crop_gray - g_min) / (g_max - g_min) * 255.0

    # ── 1. 颜色直方图全库粗筛（前 15 候选，使用预缓存直方图）──
    query_hist = _compute_hist(crop_bgr)
    hist_scores: List[Tuple[float, str]] = []
    for stem, tmpl_hist in _champ_templates_hist.items():
        sim = float(cv2.compareHist(query_hist, tmpl_hist, cv2.HISTCMP_CORREL))
        hist_scores.append((sim, stem))
    hist_scores.sort(reverse=True)
    candidates = [stem for _, stem in hist_scores[:15]]

    # ── 2. 零均值 NCC 精筛 ─────────────────────────────────
    # 拉平为向量后做归一化点积，而非 matchTemplate 滑动窗口
    crop_flat = crop_gray.flatten()
    crop_flat = crop_flat - crop_flat.mean()
    crop_norm = np.linalg.norm(crop_flat)
    if crop_norm < 1e-6:
        return "", 0.0

    best_name  = ""
    best_score = -1.0
    for stem in candidates:
        tmpl_flat = _champ_templates_gray[stem].astype(np.float32).flatten()
        tmpl_flat = tmpl_flat - tmpl_flat.mean()
        tmpl_norm = np.linalg.norm(tmpl_flat)
        if tmpl_norm < 1e-6:
            continue
        ncc = float(np.dot(crop_flat, tmpl_flat) / (crop_norm * tmpl_norm))

        # 直方图相似度（已排好序，直接查）
        h_sim = next((s for s, st in hist_scores if st == stem), 0.0)
        # 加权融合：直方图 40% + NCC 60%
        fused = 0.4 * max(h_sim, 0.0) + 0.6 * max(ncc, 0.0)

        if fused > best_score:
            best_score = fused
            best_name  = stem

    if best_score >= threshold:
        return best_name, best_score
    return "", best_score


# ──────────────────────────────────────────────────────────────
# 星级检测
# ──────────────────────────────────────────────────────────────
def detect_star(region: np.ndarray) -> int:
    """
    在英雄框上方区域检测金色/白色星点数量（1~3）
    """
    if region is None or region.size == 0:
        return 1

    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # 金色星点
    gold_mask = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([38, 255, 255]))
    # 白色星点（3 星常见）
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    combined   = cv2.bitwise_or(gold_mask, white_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = (region.shape[0] * region.shape[1]) * 0.003

    star_count = sum(1 for cnt in contours if cv2.contourArea(cnt) >= area_threshold)
    return max(1, min(3, star_count)) if star_count > 0 else 1


# ──────────────────────────────────────────────────────────────
# 六边形边框检测（棋盘模式）
# ──────────────────────────────────────────────────────────────
def _is_ui_icon(img: np.ndarray, x: int, y: int, bw: int, bh: int) -> bool:
    """
    判断一个检测框是否是 UI 图标（羁绊标志、棋盘图标等）而非英雄头像。

    方法：对框内部（去掉 20% 边框后）做 HSV 分析：
      - 饱和度极高（>175）且色调标准差极低（<15）= 纯色单调图形 = UI 图标
      - 英雄头像有复杂的人物纹理，色调变化范围更大

    此规则在测试截图上对 11 个框实现了 11/11 的正确区分。
    """
    h_img, w_img = img.shape[:2]
    mx = int(bw * 0.20)
    my = int(bh * 0.20)
    x0 = max(0, x + mx);  x1 = min(w_img, x + bw - mx)
    y0 = max(0, y + my);  y1 = min(h_img, y + bh - my)
    if x1 <= x0 or y1 <= y0:
        return False
    inner = img[y0:y1, x0:x1]
    if inner.size == 0:
        return False
    rsz = cv2.resize(inner, (48, 48))
    hsv = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mean = hsv[:, :, 1].mean()
    hue_std  = hsv[:, :, 0].std()
    # 极高饱和度 + 极低色调方差 = 纯色图标
    return sat_mean > 175 and hue_std < 15


def detect_hero_boxes(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    检测彩色六边形边框，返回每个英雄的外接矩形 (x, y, w, h)

    过滤层级（从宽松到严格）：
      1. 相对面积范围过滤
      2. 绝对最小尺寸过滤（排除星级角标、小噪声框）
      3. 宽高比过滤
      4. UI 图标过滤（排除羁绊图标、棋盘装饰等纯色单调图形）
      5. NMS（IoU > 0.3 保留较大框）
    """
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for lo, hi in BORDER_RANGES:
        mask = cv2.inRange(hsv, lo, hi)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 相对面积范围
    min_area = (h * w) * 0.0008
    max_area = (h * w) * 0.04
    # 绝对最小边长（适配不同分辨率）
    min_side = max(60, min(h, w) * 0.05)

    valid_boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_side or bh < min_side:
            continue
        ratio = bw / (bh + 1e-6)
        if not (0.6 <= ratio <= 1.6):
            continue
        # 过滤羁绊图标、棋盘装饰等纯色 UI 元素
        if _is_ui_icon(img, x, y, bw, bh):
            continue
        valid_boxes.append((x, y, bw, bh))

    # NMS：IoU > 0.3 时只保留较大的框
    valid_boxes = _nms_boxes(valid_boxes, iou_threshold=0.3)

    # 相对尺寸过滤：去掉面积 < 中位数 50% 的框
    # 羁绊徽章（~86x93）通过了绝对尺寸检查，但面积仅为英雄框（~144x160）的 35%
    if len(valid_boxes) >= 3:
        import statistics
        areas = [bw * bh for _, _, bw, bh in valid_boxes]
        median_area = statistics.median(areas)
        valid_boxes = [(x, y, bw, bh) for x, y, bw, bh in valid_boxes
                       if bw * bh >= median_area * 0.50]

    valid_boxes.sort(key=lambda b: (b[1] // 60, b[0]))
    return valid_boxes


def _nms_boxes(boxes: List[Tuple], iou_threshold: float = 0.3) -> List[Tuple]:
    """简单的非极大值抑制（按面积排序）"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[Tuple] = []
    for b in boxes:
        x1, y1, w1, h1 = b
        dominated = False
        for k in kept:
            x2, y2, w2, h2 = k
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            if union > 0 and inter / union > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(b)
    return kept


# ──────────────────────────────────────────────────────────────
# 横排一字排列检测（结算/回顾界面）
# ──────────────────────────────────────────────────────────────
def detect_horizontal_lineup(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    使用 Canny + 轮廓检测识别横排英雄头像（无六边形边框）
    """
    h, w = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.002
    max_area = (h * w) * 0.06

    candidate_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ratio = bw / (bh + 1e-6)
        if not (0.7 <= ratio <= 1.5):
            continue
        candidate_boxes.append((x, y, bw, bh))

    if len(candidate_boxes) < 3:
        return []

    # 过滤：保留同一水平带内的框
    cys     = [b[1] + b[3] // 2 for b in candidate_boxes]
    med_cy  = sorted(cys)[len(cys) // 2]
    avg_h   = float(np.median([b[3] for b in candidate_boxes]))
    horizontal = [b for b in candidate_boxes
                  if abs((b[1] + b[3] // 2) - med_cy) < avg_h * 0.6]

    if len(horizontal) < 3:
        return []

    horizontal = _nms_boxes(horizontal, iou_threshold=0.3)
    horizontal.sort(key=lambda b: b[0])
    return horizontal


def is_horizontal_layout(boxes: List[Tuple]) -> bool:
    if not boxes or len(boxes) < 2:
        return False
    cys   = [b[1] + b[3] // 2 for b in boxes]
    avg_h = float(np.median([b[3] for b in boxes]))
    return (max(cys) - min(cys)) < avg_h * 0.8


# ──────────────────────────────────────────────────────────────
# 棋盘坐标推断
# ──────────────────────────────────────────────────────────────
def infer_grid(cx: float, cy: float, all_boxes: List[Tuple]) -> Tuple[int, int]:
    """根据所有框的分布，将 (cx, cy) 映射为 (row, col)"""
    all_cx = [b[0] + b[2] // 2 for b in all_boxes]
    all_cy = [b[1] + b[3] // 2 for b in all_boxes]
    if not all_cx or not all_cy:
        return 0, 0

    # 聚类 Y 轴到行
    sorted_cy = sorted(set(all_cy))
    rows = _cluster_1d(sorted_cy, gap_ratio=0.4)
    row_idx = min(range(len(rows)), key=lambda i: abs(rows[i] - cy))

    sorted_cx = sorted(set(all_cx))
    cols = _cluster_1d(sorted_cx, gap_ratio=0.4)
    col_idx = min(range(len(cols)), key=lambda i: abs(cols[i] - cx))

    return row_idx, col_idx


def _cluster_1d(values: List[float], gap_ratio: float = 0.4) -> List[float]:
    """将 1D 坐标列表按间距分组，返回每组中心"""
    if not values:
        return []
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v

    # 估算典型间距
    diffs = [sorted_v[i+1] - sorted_v[i] for i in range(len(sorted_v)-1)]
    median_diff = float(np.median(diffs)) if diffs else 1.0
    threshold   = median_diff * (1 + gap_ratio)

    groups: List[List[float]] = [[sorted_v[0]]]
    for v in sorted_v[1:]:
        if v - groups[-1][-1] <= threshold:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


# ──────────────────────────────────────────────────────────────
# 装备识别
# ──────────────────────────────────────────────────────────────
def identify_items(img: np.ndarray, box: Tuple[int, int, int, int],
                   threshold: float = ITEM_THRESHOLD) -> List[str]:
    """
    在英雄框正下方区域检测装备图标。

    修复：原代码在装备区域很小时，模板被过度缩放到 8×8px，
    导致几乎所有模板都能匹配，出现 Quicksilver×3 等误报。
    现在：
      1. 装备区域高度 < 20px 时直接跳过（太小无法可靠匹配）
      2. 模板缩放比例锁定在合理范围（不低于原始尺寸的 30%）
      3. 对每个位置只保留分数最高的装备（避免同位置多次命中）
    """
    x, y, bw, bh = box
    h, w = img.shape[:2]

    iy1 = y + bh
    iy2 = min(h, y + bh + int(bh * 0.55))
    ix1 = max(0, x - int(bw * 0.05))
    ix2 = min(w, x + bw + int(bw * 0.05))

    if iy2 <= iy1 or ix2 <= ix1 or not _item_templates_gray:
        return []

    region = img[iy1:iy2, ix1:ix2]
    rh, rw = region.shape[:2]

    # 装备区域太小无法可靠匹配
    if rh < 20 or rw < 20:
        return []

    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # 装备图标实际大小约为区域高度的 80%，但不超过模板原始尺寸
    item_size = min(rh, TEMPLATE_SIZE)
    # 最小匹配尺寸：不低于原始模板的 40%（否则细节丢失过多）
    if item_size < TEMPLATE_SIZE * 0.4:
        return []

    # 对每个 x 位置记录最高分，避免同一位置多模板命中
    position_best: Dict[int, Tuple[float, str]] = {}  # x_bucket -> (score, stem)
    bucket_size = item_size  # 同一位置窗口大小

    for stem, tmpl in _item_templates_gray.items():
        tmpl_scaled = cv2.resize(tmpl, (item_size, item_size))
        if tmpl_scaled.shape[1] > gray_region.shape[1] or \
           tmpl_scaled.shape[0] > gray_region.shape[0]:
            continue
        result = cv2.matchTemplate(gray_region, tmpl_scaled, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(result)
        if val < threshold:
            continue
        lx = loc[0]
        bucket = lx // bucket_size
        if bucket not in position_best or val > position_best[bucket][0]:
            position_best[bucket] = (val, stem)

    # 按 x 位置排序，返回装备名
    items = [stem for _, (_, stem) in sorted(position_best.items())]
    return items[:3]


# ──────────────────────────────────────────────────────────────
# 主识别函数
# ──────────────────────────────────────────────────────────────
def recognize_from_array(
    img: np.ndarray,
    champ_thr: float = MATCH_THRESHOLD,
    item_thr: float  = ITEM_THRESHOLD,
    debug: bool      = False,
) -> Dict:
    _load_templates()
    if not _champ_templates_gray:
        return {
            "error": "未找到英雄模板",
            "hint" : "请先运行 python tft_fetch_assets.py 下载模板图片",
        }

    t0 = time.time()
    h, w = img.shape[:2]

    # 检测英雄框
    boxes = detect_hero_boxes(img)
    layout_mode = "board"

    if not boxes or is_horizontal_layout(boxes):
        h_boxes = detect_horizontal_lineup(img)
        if len(h_boxes) >= len(boxes):
            boxes = h_boxes
            layout_mode = "lineup"

    if not boxes:
        return {
            "team_size"       : 0,
            "champions"       : [],
            "traits"          : [],
            "summary"         : {},
            "equipment_issues": [],
            "_source"         : "cv_template",
            "_elapsed_ms"     : int((time.time() - t0) * 1000),
            "error"           : "未检测到英雄",
            "hint"            : "请确保截图中包含棋盘或横排英雄",
        }

    debug_img = img.copy() if debug else None
    champions: List[Dict] = []

    for box in boxes:
        x, y, bw, bh = box
        # 裁剪内部（去除边框噪声）
        mx = int(bw * INNER_MARGIN)
        my = int(bh * INNER_MARGIN)
        inner = img[y+my:y+bh-my, x+mx:x+bw-mx]

        stem, score = identify_champion(inner, champ_thr)

        # 星级：在框上方检测
        star_region = img[max(0, y - int(bh * 0.35)):y + bh, x:x+bw]
        star = detect_star(star_region)

        # 装备
        items = identify_items(img, box, item_thr)

        # 坐标
        if layout_mode == "lineup":
            row, col = 0, boxes.index(box)
        else:
            row, col = infer_grid(x + bw//2, y + bh//2, boxes)

        if debug_img is not None:
            color = (0, 255, 0) if stem else (0, 80, 255)
            cv2.rectangle(debug_img, (x, y), (x+bw, y+bh), color, 2)
            label = f"{stem or '?'} {star}★ {score:.2f}"
            cv2.putText(debug_img, label, (x, max(0, y-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.putText(debug_img, f"[{row},{col}]", (x+2, y+bh-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 200, 0), 1)

        champions.append({
            "id"      : stem,          # 完整 ID，如 TFT16_Draven
            "short_id": stem.replace("TFT16_", "").replace("TFT_", "") if stem else "",
            "name_en" : stem.replace("TFT16_", "").replace("TFT_", "") if stem else f"unknown_{x}_{y}",
            "star"    : star,
            "cost"    : 0,             # 由 tft_converter 查 DB 补全
            "items"   : items,
            "position": {"row": row, "col": col},
            "_score"  : round(score, 3),
            "_box"    : [x, y, bw, bh],
        })

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)
        print("[Debug] 标注图已保存: tft_debug.png")

    known = [c for c in champions if c["id"]]

    # 尝试用 tft_converter 补全羁绊信息
    traits: List[Dict] = []
    summary: Dict = {}
    equipment_issues: List[str] = []
    try:
        from tft_converter import calc_traits, build_summary
        traits, (summary, equipment_issues) = calc_traits(known), build_summary(known, [])
    except ImportError:
        summary = {"front_row_ratio": f"?/{len(known)}", "main_carry": ""}

    return {
        "team_size"       : len(known),
        "champions"       : champions,
        "traits"          : traits,
        "summary"         : summary,
        "equipment_issues": equipment_issues,
        "_source"         : "cv_template",
        "_elapsed_ms"     : int((time.time() - t0) * 1000),
        "_detected_boxes" : len(boxes),
        "_layout"         : layout_mode,
    }


def recognize(
    source,
    champ_threshold: float = MATCH_THRESHOLD,
    item_threshold: float  = ITEM_THRESHOLD,
    debug: bool            = False,
    assets_dir: str        = None,
    mode: str              = "auto",   # "auto" | "board" | "lineup" | "global" | "duel"
) -> Dict:
    """
    统一识别入口，支持四种截图模式自动检测。

    模式说明：
      board   - Single-Board: 棋盘布局（4×7 六边形网格，DataTFT / 阵容模拟器）
      lineup  - Single-Lineup: 结算简略图（英雄水平一排，无六边形边框）
      global  - Global: 阵容羁绊表（8名玩家小图标，多行）
      duel    - Duel: 战绩回顾（两个棋盘上下叠放）
      auto    - 自动检测（默认）
    """
    global ASSETS_DIR, CHAMP_DIR, ITEM_DIR, _templates_loaded
    if assets_dir:
        ASSETS_DIR = Path(assets_dir)
        CHAMP_DIR  = ASSETS_DIR / "champions"
        ITEM_DIR   = ASSETS_DIR / "items"
        _templates_loaded = False

    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            return {"error": f"无法读取文件: {source}"}
    elif isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "无法解码图片字节"}
    elif isinstance(source, np.ndarray):
        img = source
    else:
        return {"error": f"不支持的输入类型: {type(source)}"}

    if mode == "auto":
        mode = detect_screenshot_mode(img)

    if mode == "lineup":
        return recognize_lineup(img, champ_threshold, item_threshold, debug)
    if mode == "global":
        return recognize_global(img, champ_threshold, item_threshold, debug)
    if mode == "duel":
        return recognize_duel(img, champ_threshold, item_threshold, debug)
    # board / fallback
    return recognize_from_array(img, champ_threshold, item_threshold, debug)


# ──────────────────────────────────────────────────────────────
# 截图模式自动检测
# ──────────────────────────────────────────────────────────────
def detect_screenshot_mode(img: np.ndarray) -> str:
    """
    根据图像特征自动判断截图类型：
      board   - 棋盘布局（DataTFT / 阵容模拟器截图，紧凑，无过多 UI）
      lineup  - 结算简略横排图（大图，英雄左侧水平排列，右侧有吉祥物）
      global  - 阵容羁绊表（大图，满布小图标行，紫色 UI）
      duel    - 战绩回顾（大图，两个棋盘上下叠放，紫色 UI）

    检测逻辑：
      1. 长宽比 < 2.1 → board（DataTFT 模拟器图片较窄）
      2. 大图中：检测中间导航栏亮度分布区分 Global vs Duel
      3. 剩余 → lineup（结算图，亮度均匀的深蓝背景）
    """
    h, w = img.shape[:2]
    aspect = w / h

    # ── 1. Board：DataTFT 模拟器截图（紧凑，aspect 约 1.9-2.1）
    if aspect < 2.1 and h < 500:
        return "board"

    # ── 2. 大图（aspect ≈ 2.27, 2000×881 手机截图）
    # 检测是否有紫色/粉色 UI 背景（Global/Duel 特征）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 紫色背景 H~140-170, S>30, V>20
    purple_mask = cv2.inRange(hsv,
                              np.array([130, 25, 20]),
                              np.array([175, 200, 120]))
    purple_density = purple_mask.sum() / 255 / (h * w)

    if purple_density < 0.05:
        # 深蓝背景 → Lineup（结算图）
        return "lineup"

    # ── 3. 区分 Global vs Duel ────────────────────────────────
    # Global（阵容羁绊）：中间导航栏 '阵容羁绊' 选项卡亮
    # Duel（战绩回顾）：右侧 '战绩回顾' 选项卡亮
    # 检测导航栏（顶部 y=0:60）中间区段亮度
    mid_nav = img[0:60, int(w * 0.40):int(w * 0.65)]
    mid_bright = (cv2.cvtColor(mid_nav, cv2.COLOR_BGR2GRAY) > 180).sum()

    if mid_bright > 800:
        return "global"
    return "duel"


# ──────────────────────────────────────────────────────────────
# 模式2: Lineup 识别（结算简略横排图）
# ──────────────────────────────────────────────────────────────
def _extract_lineup_boxes(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    结算简略横排图中英雄图标检测。

    实测发现：结算图中英雄图标紧密排列（相邻图标间距仅 1px），
    列方差法无法可靠区分相邻图标。

    改进方案：
      1. 找面部区域中亮度 > 40 的列段，自动推算图标宽度和起始位置
      2. 以等间距方式放置 8 个图标框（stride ≈ 62px）
      3. 若推算失败则使用相对比例兜底
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    from scipy.ndimage import gaussian_filter1d  # type: ignore

    # ── 第一步：找英雄行的 y 坐标 ──────────────────────────────
    x_scan_end = int(w * 0.55)
    sat = hsv[h // 2:, 100:x_scan_end, 1]
    row_colorful = (sat > 60).sum(axis=1).astype(float)
    smoothed = gaussian_filter1d(row_colorful, sigma=4)
    peak_y = int(smoothed.argmax()) + h // 2

    icon_h  = max(60, int(h * 0.10))
    icon_y  = max(0, peak_y - icon_h // 3)
    icon_y2 = min(h, icon_y + icon_h + 5)

    # ── 第二步：在面部区域（icon_y ~ icon_y2 中间 40%）找锚点 ──
    face_y1 = icon_y + int(icon_h * 0.25)
    face_y2 = icon_y + int(icon_h * 0.65)
    face_strip = img[face_y1:face_y2, 100:x_scan_end]
    gray_face  = cv2.cvtColor(face_strip, cv2.COLOR_BGR2GRAY).astype(float)
    col_mean   = gray_face.mean(axis=0)

    # 找亮度 > 40 的连续段（每个段对应一个英雄面部）
    bright = (col_mean > 40).astype(np.int8)
    diffs  = np.diff(np.concatenate([[0], bright, [0]]))
    starts = np.where(diffs == 1)[0] + 100   # 还原 x 偏移
    ends   = np.where(diffs == -1)[0] + 100
    segments = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= 40]

    if len(segments) < 2:
        # 兜底：使用固定比例（2000px 宽图标从 x=260 开始，stride=62px）
        start_x  = int(w * 0.130)
        stride   = int(w * 0.031)
        icon_w   = int(w * 0.029)
        boxes = [(start_x + i * stride, icon_y, icon_w, icon_y2 - icon_y)
                 for i in range(8)]
        return boxes

    # ── 第三步：从前几个清晰段推算 stride ─────────────────────
    strides = [segments[i+1][0] - segments[i][0] for i in range(min(3, len(segments)-1))]
    stride  = int(np.median(strides))
    icon_w  = max(40, int(np.median([e - s for s, e in segments[:4]])))
    start_x = segments[0][0]

    boxes = []
    for i in range(10):     # 最多检测 10 个
        x = start_x + i * stride
        if x + icon_w > x_scan_end + 100:
            break
        boxes.append((x, icon_y, icon_w, icon_y2 - icon_y))

    return boxes


def recognize_lineup(img: np.ndarray,
                     champ_thr: float = MATCH_THRESHOLD,
                     item_thr:  float = ITEM_THRESHOLD,
                     debug:     bool  = False) -> Dict:
    """结算简略横排图识别"""
    _load_templates()
    t0 = time.time()

    try:
        boxes = _extract_lineup_boxes(img)
    except Exception:
        boxes = []

    if not boxes:
        # 降级到通用检测
        return recognize_from_array(img, champ_thr, item_thr, debug)

    debug_img = img.copy() if debug else None
    champions = []
    for col_idx, (x, y, bw, bh) in enumerate(boxes):
        mx, my = int(bw * INNER_MARGIN), int(bh * INNER_MARGIN)
        inner  = img[y + my:y + bh - my, x + mx:x + bw - mx]

        stem, score = identify_champion(inner, champ_thr)
        star_region = img[max(0, y - int(bh * 0.4)):y + bh, x:x + bw]
        star  = detect_star(star_region)
        items = identify_items(img, (x, y, bw, bh), item_thr)

        if debug_img is not None:
            c = (0, 255, 0) if stem else (0, 80, 255)
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), c, 2)
            cv2.putText(debug_img, f"{stem or '?'} {score:.2f}",
                        (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        champions.append({
            "id"      : stem,
            "short_id": stem.replace("TFT16_", "") if stem else "",
            "name_en" : stem.replace("TFT16_", "") if stem else f"unknown_{x}_{y}",
            "star"    : star,
            "cost"    : 0,
            "items"   : items,
            "position": {"row": 0, "col": col_idx},
            "_score"  : round(score, 3),
            "_box"    : [x, y, bw, bh],
        })

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    known = [c for c in champions if c["id"]]
    traits, summary, issues = [], {}, []
    try:
        from tft_converter import calc_traits, build_summary
        traits = calc_traits(known)
        summary, issues = build_summary(known, [])
    except ImportError:
        summary = {"front_row_ratio": f"?/{len(known)}", "main_carry": ""}

    return {
        "team_size"       : len(known),
        "champions"       : champions,
        "traits"          : traits,
        "summary"         : summary,
        "equipment_issues": issues,
        "_source"         : "cv_lineup",
        "_elapsed_ms"     : int((time.time() - t0) * 1000),
        "_layout"         : "lineup",
    }


# ──────────────────────────────────────────────────────────────
# 模式3: Global 识别（阵容羁绊表，8名玩家）
# ──────────────────────────────────────────────────────────────
def _find_global_player_rows(img: np.ndarray) -> List[Dict]:
    """
    在 Global 截图中找到每名玩家的图标行区间。

    策略：
      1. 用 HoughCircles 在左侧玩家头像列找圆形头像（每人一个）
      2. 对检测到的圆去重（移除间距 < 45px 的重复）
      3. 在每个玩家圆心 y 附近的高饱和度像素区确定精确行范围
      4. 使用 prev_y2 防止行区间重叠
    """
    h, w = img.shape[:2]

    # 玩家头像列
    avatar_x1, avatar_x2 = int(w * 0.115), int(w * 0.20)
    avatar_region = img[85:h - 60, avatar_x1:avatar_x2]
    gray_av = cv2.cvtColor(avatar_region, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray_av, cv2.HOUGH_GRADIENT, dp=1, minDist=35,
        param1=50, param2=20, minRadius=12, maxRadius=30
    )
    avatar_ys = []
    if circles is not None:
        for c in np.uint16(np.around(circles[0])):
            avatar_ys.append(int(c[1]) + 85)
    avatar_ys.sort()

    # 去重：合并间距 < 45px 的圆（同一玩家的多次检测）
    deduped: List[int] = []
    for y in avatar_ys:
        if not deduped or y - deduped[-1] > 45:
            deduped.append(y)
    avatar_ys = deduped

    # 兜底：固定间距估算
    if len(avatar_ys) < 2:
        row_h = int(h * 0.105)
        avatar_ys = [85 + i * row_h for i in range(8)]

    champ_x1 = int(w * 0.235)
    champ_x2 = int(w * 0.500)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, champ_x1:champ_x2, 1]

    rows = []
    prev_y2 = 85
    for rank, center_y in enumerate(avatar_ys, 1):
        y_search_start = max(prev_y2, center_y - 15)
        y_search_end   = min(h - 30, center_y + 120)
        sat_strip  = sat[y_search_start:y_search_end]
        row_col    = (sat_strip > 55).sum(axis=1)
        active     = (row_col > 8).astype(np.int8)
        if active.sum() == 0:
            continue
        diffs = np.diff(np.concatenate([[0], active, [0]]))
        seg_starts = np.where(diffs == 1)[0]
        seg_ends   = np.where(diffs == -1)[0]
        if len(seg_starts) == 0:
            continue
        y1 = y_search_start + int(seg_starts[0])
        y2 = y_search_start + int(seg_ends[-1])
        if y2 - y1 < 15:
            continue
        rows.append({
            "rank": rank,
            "y1"  : y1,
            "y2"  : y2,
            "x1"  : champ_x1,
            "x2"  : champ_x2,
        })
        prev_y2 = y2

    return rows

def _extract_small_icons(img: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int,
                         champ_thr: float) -> List[Dict]:
    """
    在指定矩形区域内检测小型英雄图标（Global/Duel 模式）。
    图标无六边形边框，使用滑动窗口 + 颜色直方图匹配。
    """
    _load_templates()
    if not _champ_templates_hist:
        return []

    region = img[y1:y2, x1:x2]
    rh, rw = region.shape[:2]
    if rh < 10 or rw < 10:
        return []

    # 估算图标大小：基于行高
    icon_size = max(28, min(rh - 8, 48))  # 28~48px

    # 滑动窗口（步长 = icon_size // 2）
    step = max(8, icon_size // 2)
    candidates = []

    for x in range(0, rw - icon_size, step):
        for y in range(0, rh - icon_size, step):
            patch = region[y:y + icon_size, x:x + icon_size]
            if patch.size == 0:
                continue

            patch_bgr  = cv2.resize(patch, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            query_hist = _compute_hist(patch_bgr)

            # 直方图粗筛
            best_sim, best_stem = 0.0, ""
            for stem, tmpl_hist in _champ_templates_hist.items():
                sim = float(cv2.compareHist(query_hist, tmpl_hist, cv2.HISTCMP_CORREL))
                if sim > best_sim:
                    best_sim, best_stem = sim, stem

            if best_sim < 0.55:   # 较高直方图阈值（小图标噪声更多）
                continue

            # NCC 精筛
            gray_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gmin, gmax = gray_patch.min(), gray_patch.max()
            if gmax - gmin > 10:
                gray_patch = (gray_patch - gmin) / (gmax - gmin + 1e-6) * 255
            pf = gray_patch.flatten(); pf -= pf.mean(); pn = np.linalg.norm(pf)

            tf = _champ_templates_gray[best_stem].astype(np.float32).flatten()
            tf -= tf.mean(); tn = np.linalg.norm(tf)
            ncc = float(np.dot(pf, tf) / (pn * tn + 1e-8)) if pn > 1e-6 else 0.0
            fused = 0.4 * max(best_sim, 0) + 0.6 * max(ncc, 0)

            if fused >= champ_thr:
                candidates.append({
                    "stem" : best_stem,
                    "score": fused,
                    "x"    : x1 + x,
                    "y"    : y1 + y,
                    "size" : icon_size,
                })

    # NMS：合并重叠候选
    candidates.sort(key=lambda c: -c["score"])
    kept = []
    for cand in candidates:
        cx, cy = cand["x"] + cand["size"] // 2, cand["y"] + cand["size"] // 2
        duplicate = any(abs(cx - (k["x"] + k["size"]//2)) < cand["size"] * 0.6 and
                        abs(cy - (k["y"] + k["size"]//2)) < cand["size"] * 0.6
                        for k in kept)
        if not duplicate:
            kept.append(cand)

    # 转换为 champion dict
    result = []
    for i, c in enumerate(kept):
        stem = c["stem"]
        star_region = img[max(0, c["y"] - 12):c["y"] + c["size"], c["x"]:c["x"] + c["size"]]
        star = detect_star(star_region)
        result.append({
            "id"      : stem,
            "short_id": stem.replace("TFT16_", "") if stem else "",
            "name_en" : stem.replace("TFT16_", "") if stem else "unknown",
            "star"    : star,
            "cost"    : 0,
            "items"   : [],
            "position": {"row": 0, "col": i},
            "_score"  : round(c["score"], 3),
            "_box"    : [c["x"], c["y"], c["size"], c["size"]],
        })
    return result


def recognize_global(img: np.ndarray,
                     champ_thr: float = MATCH_THRESHOLD,
                     item_thr:  float = ITEM_THRESHOLD,
                     debug:     bool  = False) -> Dict:
    """
    Global 模式（阵容羁绊表）识别。
    识别所有 8 名玩家的英雄阵容，返回包含 players 列表的结构。
    """
    t0 = time.time()
    _load_templates()

    player_rows = _find_global_player_rows(img)
    players = []

    debug_img = img.copy() if debug else None

    for row_info in player_rows:
        champs = _extract_small_icons(
            img,
            row_info["x1"], row_info["y1"],
            row_info["x2"], row_info["y2"],
            champ_thr,
        )
        known = [c for c in champs if c["id"]]
        traits = []
        try:
            from tft_converter import calc_traits
            traits = calc_traits(known)
        except ImportError:
            pass

        players.append({
            "rank"     : row_info["rank"],
            "team_size": len(known),
            "champions": champs,
            "traits"   : traits,
            "_row_y"   : [row_info["y1"], row_info["y2"]],
        })

        if debug_img is not None:
            cv2.rectangle(debug_img,
                          (row_info["x1"], row_info["y1"]),
                          (row_info["x2"], row_info["y2"]),
                          (0, 255, 100), 2)
            for c in champs:
                x, y, s = c["_box"][0], c["_box"][1], c["_box"][2]
                cv2.rectangle(debug_img, (x, y), (x+s, y+s), (0, 200, 255), 1)
                cv2.putText(debug_img, c["short_id"][:6],
                            (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 200), 1)

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    # 我方（第1名 or 当前玩家）的阵容作为主阵容
    my_team = players[0]["champions"] if players else []

    return {
        "team_size"   : len([c for c in my_team if c["id"]]),
        "champions"   : my_team,
        "traits"      : players[0]["traits"] if players else [],
        "summary"     : {},
        "players"     : players,
        "_source"     : "cv_global",
        "_elapsed_ms" : int((time.time() - t0) * 1000),
        "_layout"     : "global",
    }


# ──────────────────────────────────────────────────────────────
# 模式4: Duel 识别（战绩回顾，两个棋盘上下叠放）
# ──────────────────────────────────────────────────────────────
def _find_duel_boards(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    定位 Duel 截图中的两个棋盘区域。

    策略：在棋盘中央 x 区间扫描每行亮度，
    找到连续暗带（两棋盘之间的分隔区域），以此为分界点切分上下两棋盘。
    含固定比例兜底确保始终返回 2 个区域。
    """
    h, w = img.shape[:2]
    bx1, bx2 = int(w * 0.21), int(w * 0.48)

    # 在两棋盘之间（y=250-500）找最暗的水平带（分隔线）
    check = img[250:500, bx1:bx2]
    gray  = cv2.cvtColor(check, cv2.COLOR_BGR2GRAY)
    from scipy.ndimage import gaussian_filter1d  # type: ignore
    row_mean = gaussian_filter1d(gray.mean(axis=1), sigma=3)

    # 在 row_mean 中找局部最小值（最暗行 = 分隔线）
    min_idx = int(np.argmin(row_mean))
    sep_y   = 250 + min_idx

    # 确保分隔线在合理位置（30%~70% 高度区间）
    if not (int(h * 0.30) <= sep_y <= int(h * 0.70)):
        sep_y = int(h * 0.43)   # 兜底

    board0 = (bx1, int(h * 0.17), bx2, sep_y)
    board1 = (bx1, sep_y,          bx2, min(h - 20, sep_y + int(h * 0.30)))

    return [board0, board1]

def recognize_duel(img: np.ndarray,
                   champ_thr: float = MATCH_THRESHOLD,
                   item_thr:  float = ITEM_THRESHOLD,
                   debug:     bool  = False) -> Dict:
    """
    Duel 模式（战绩回顾）识别。
    上方棋盘：对手的阵容
    下方棋盘：我方的阵容
    """
    t0 = time.time()

    boards = _find_duel_boards(img)

    debug_img = img.copy() if debug else None

    all_boards_result = []
    for board_idx, (bx1, by1, bx2, by2) in enumerate(boards):
        board_img = img[by1:by2, bx1:bx2]

        if debug_img is not None:
            cv2.rectangle(debug_img, (bx1, by1), (bx2, by2),
                          (0, 255, 0) if board_idx == 0 else (255, 100, 0), 3)

        # 在每个棋盘子图像上运行标准六边形检测
        board_result = recognize_from_array(board_img, champ_thr, item_thr, debug=False)

        # 把坐标转换回全图坐标
        for c in board_result.get("champions", []):
            box = c.get("_box", [0, 0, 0, 0])
            c["_box"]    = [box[0] + bx1, box[1] + by1, box[2], box[3]]
            c["position"]["board"] = board_idx

        all_boards_result.append({
            "board_idx": board_idx,
            "label"    : "opponent" if board_idx == 0 else "mine",
            **board_result,
        })

    # 我方阵容 = 下方棋盘（board_idx=1），或第一个棋盘
    my_board = all_boards_result[1] if len(all_boards_result) > 1 else (
               all_boards_result[0] if all_boards_result else {})

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    return {
        "team_size"       : my_board.get("team_size", 0),
        "champions"       : my_board.get("champions", []),
        "traits"          : my_board.get("traits", []),
        "summary"         : my_board.get("summary", {}),
        "equipment_issues": my_board.get("equipment_issues", []),
        "boards"          : all_boards_result,
        "_source"         : "cv_duel",
        "_elapsed_ms"     : int((time.time() - t0) * 1000),
        "_layout"         : "duel",
    }




# ──────────────────────────────────────────────────────────────
# 阈值标定工具
# ──────────────────────────────────────────────────────────────
def diagnose(image_path: str):
    """
    深度诊断模式：输出模板质量、截图裁剪质量、最高匹配分数，
    帮助定位识别失败的根本原因。
    """
    import os
    _load_templates()

    print(f"\n{'='*60}")
    print("  TFT 识别诊断报告")
    print(f"{'='*60}")

    # ── 1. 模板质量检查 ────────────────────────────────────────
    print(f"\n[1] 模板库状态")
    print(f"  英雄模板: {len(_champ_templates_gray)} 个")
    print(f"  装备模板: {len(_item_templates_gray)} 个")

    if not _champ_templates_gray:
        print("  ✗ 英雄模板为空！请先运行 tft_fetch_assets.py")
        return

    # 抽查前5个模板的像素统计，并检查 alpha 合成是否正确工作
    print("\n  抽查模板像素统计（前5个）:")
    print(f"  {'名称':<30} {'mean':>6} {'std':>6} {'min':>4} {'max':>4}  {'状态'}")
    print(f"  {'-'*70}")
    for stem, tmpl in list(_champ_templates_gray.items())[:5]:
        mean_val = tmpl.mean()
        std_val  = tmpl.std()
        min_val  = int(tmpl.min())
        max_val  = int(tmpl.max())
        if std_val < 5 or max_val < 20:
            flag = "⚠ 空白/错误图片"
        elif mean_val < 60:
            flag = "⚠ 偏暗，可能是LoL立绘或alpha未合成"
        elif 80 <= mean_val <= 180:
            flag = "✓ 亮度正常"
        else:
            flag = "○ 偏亮"
        print(f"  {stem:<30} {mean_val:6.1f} {std_val:6.1f} {min_val:4d} {max_val:4d}  {flag}")
    print()
    print("  💡 mean 应在 80~180 之间（合成到灰底128后的正常范围）")
    print("     若 mean < 60，说明模板来自错误来源（LoL立绘）或alpha未正确合成")
    print("     请删除 tft_assets/champions/*.png 后重新运行 tft_fetch_assets.py")

    # ── 2. 截图检查 ────────────────────────────────────────────
    print(f"\n[2] 截图文件: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("  ✗ 无法读取截图")
        return
    h, w = img.shape[:2]
    print(f"  分辨率: {w}×{h}")

    # ── 3. 边框检测 ────────────────────────────────────────────
    boxes = detect_hero_boxes(img)
    print(f"\n[3] 检测到英雄框: {len(boxes)} 个")
    for i, (x, y, bw, bh) in enumerate(boxes[:8]):
        print(f"  [{i}] x={x:4d} y={y:4d} w={bw:4d} h={bh:4d}  面积={bw*bh}")

    if not boxes:
        print("  → 建议：截图可能不包含棋盘，或边框颜色不在检测范围内")
        print("    可尝试 --debug 查看颜色检测结果")
        return

    # ── 4. 对第一个框做完整匹配分析 ────────────────────────────
    print(f"\n[4] 对第一个框做匹配分析")
    x, y, bw, bh = boxes[0]
    mx = int(bw * INNER_MARGIN)
    my = int(bh * INNER_MARGIN)
    crop = img[y+my:y+bh-my, x+mx:x+bw-mx]
    print(f"  裁剪区域: {crop.shape[1]}×{crop.shape[0]} px")

    crop_bgr  = cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    if len(crop_bgr.shape) == 2:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 亮度标准化（与 identify_champion 保持一致）
    g_min, g_max = crop_gray.min(), crop_gray.max()
    if g_max - g_min > 10:
        crop_gray = (crop_gray - g_min) / (g_max - g_min) * 255.0
    print(f"  截图区域像素统计(标准化后): mean={crop_gray.mean():.1f}  std={crop_gray.std():.1f}")

    # 直方图粗筛 Top5
    query_hist = _compute_hist(crop_bgr)
    hist_scores: List[Tuple[float, str]] = []
    for stem, tmpl_hist in _champ_templates_hist.items():
        sim = float(cv2.compareHist(query_hist, tmpl_hist, cv2.HISTCMP_CORREL))
        hist_scores.append((sim, stem))
    hist_scores.sort(reverse=True)

    print(f"\n  直方图粗筛 Top5:")
    for sim, stem in hist_scores[:5]:
        print(f"    {stem:<30} hist_sim={sim:.4f}")

    # 对 Top5 做 NCC
    crop_flat = crop_gray.flatten() - crop_gray.mean()
    crop_norm = np.linalg.norm(crop_flat)
    print(f"\n  NCC 精筛 Top5:")
    for _, stem in hist_scores[:5]:
        tmpl_f = _champ_templates_gray[stem].astype(np.float32).flatten()
        tmpl_f -= tmpl_f.mean()
        tn = np.linalg.norm(tmpl_f)
        ncc = float(np.dot(crop_flat, tmpl_f) / (crop_norm * tn + 1e-8))
        h_sim = next(s for s, st in hist_scores if st == stem)
        fused = 0.4 * max(h_sim, 0) + 0.6 * max(ncc, 0)
        verdict = "✓ 会识别" if fused >= MATCH_THRESHOLD else "✗ 低于阈值"
        print(f"    {stem:<30} ncc={ncc:.4f}  fused={fused:.4f}  {verdict}")

    print(f"\n  当前阈值: {MATCH_THRESHOLD}")
    print(f"  建议: 若 fused 分数在 0.3~0.5 之间，降低阈值到 0.35~0.40 即可")
    print(f"        若 fused 分数 < 0.2，模板图片可能与游戏版本不匹配（需重新下载）")
    print(f"        运行: python tft_screen_capture.py {image_path} --calibrate")
    print()


def calibrate(image_path: str, known_names: List[str] = None):
    """
    扫描不同阈值下的识别结果，帮助找出最佳阈值。
    known_names: 你知道截图中有哪些英雄（英文 short ID 或 stem ID）
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ 无法读取: {image_path}")
        return

    _load_templates()
    boxes = detect_hero_boxes(img)
    print(f"检测到 {len(boxes)} 个英雄框")
    print("\n阈值扫描结果:")
    print(f"{'阈值':>6}  {'识别数':>6}  {'识别到的英雄'}")
    print("-" * 70)

    best_f1, best_t = 0.0, MATCH_THRESHOLD
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        result = recognize(img, champ_threshold=t)
        names  = [c["short_id"] for c in result.get("champions", []) if c.get("id")]
        line   = f"  {t:.2f}   {len(names):>4}    {names}"
        if known_names:
            ks = set(known_names)
            ds = set(names)
            tp = len(ks & ds)
            prec = tp / max(len(ds), 1)
            rec  = tp / max(len(ks), 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-6)
            line += f"   P={prec:.2f} R={rec:.2f} F1={f1:.2f}"
            if f1 > best_f1:
                best_f1, best_t = f1, t
        print(line)

    if known_names:
        print(f"\n推荐阈值: {best_t}  (F1={best_f1:.2f})")
        print(f"用法: python tft_screen_capture.py {image_path} --threshold {best_t}")


# ──────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TFT 截图识别（支持 board / lineup / global / duel 四种模式）"
    )
    ap.add_argument("image",          nargs="?",          help="截图路径")
    ap.add_argument("--mode",         default="auto",
                    choices=["auto","board","lineup","global","duel"],
                    help="截图模式（默认 auto 自动检测）")
    ap.add_argument("--debug",        action="store_true", help="输出标注图 tft_debug.png")
    ap.add_argument("--threshold",    type=float, default=MATCH_THRESHOLD, help="英雄匹配阈值")
    ap.add_argument("--item-thresh",  type=float, default=ITEM_THRESHOLD,  help="装备匹配阈值")
    ap.add_argument("--diagnose",     action="store_true", help="深度诊断：模板质量+匹配分数分析")
    ap.add_argument("--calibrate",    action="store_true", help="阈值标定模式")
    ap.add_argument("--known",        nargs="*",           help="已知英雄 ID（用于标定）")
    ap.add_argument("--save",         type=str,            help="保存结果到 JSON 文件")
    ap.add_argument("--assets-dir",   type=str,            help="模板目录（默认 ./tft_assets）")
    args = ap.parse_args()

    if not args.image:
        print("用法: python tft_screen_capture.py screenshot.png [选项]")
        print()
        print("  --mode auto      自动检测截图类型（默认）")
        print("  --mode board     棋盘布局（DataTFT / 阵容模拟器）")
        print("  --mode lineup    结算简略横排图")
        print("  --mode global    阵容羁绊表（8名玩家）")
        print("  --mode duel      战绩回顾（两棋盘对战）")
        print("  --debug          输出标注图 tft_debug.png")
        print("  --diagnose       深度诊断模板与截图匹配情况")
        print("  --threshold 0.45 英雄匹配阈值（越低召回越高）")
        sys.exit(0)

    if args.diagnose:
        diagnose(args.image)
        sys.exit(0)

    if args.calibrate:
        calibrate(args.image, args.known)
        sys.exit(0)

    # 自动检测模式时先报告检测结果
    if args.mode == "auto":
        img_for_detect = cv2.imread(args.image)
        if img_for_detect is not None:
            detected_mode = detect_screenshot_mode(img_for_detect)
            print(f"[自动检测] 截图模式: {detected_mode}")

    result = recognize(
        args.image,
        champ_threshold=args.threshold,
        item_threshold=args.item_thresh,
        debug=args.debug,
        assets_dir=args.assets_dir,
        mode=args.mode,
    )

    if args.save:
        Path(args.save).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ 结果已保存: {args.save}")
    else:
        # 根据模式打印不同格式的摘要
        layout = result.get("_layout", "")
        elapsed = result.get("_elapsed_ms", 0)

        if layout == "global" and result.get("players"):
            print(f"\n[Global] 识别 {len(result['players'])} 名玩家 (耗时 {elapsed}ms)")
            for p in result["players"]:
                known = [c for c in p["champions"] if c.get("id")]
                names = [c["short_id"] for c in known]
                print(f"  第{p['rank']}名: {len(known)} 英雄 → {names}")

        elif layout == "duel" and result.get("boards"):
            print(f"\n[Duel] 识别两个棋盘 (耗时 {elapsed}ms)")
            for b in result["boards"]:
                known = [c for c in b.get("champions", []) if c.get("id")]
                label = "对手" if b["board_idx"] == 0 else "我方"
                names = [c["short_id"] for c in known]
                print(f"  {label}: {len(known)} 英雄 → {names}")

        else:
            n = result.get("team_size", 0)
            if n > 0:
                print(f"\n[{layout or 'board'}] 识别 {n} 名英雄 (耗时 {elapsed}ms)")
                for c in sorted(result["champions"], key=lambda x: -x.get("_score", 0)):
                    if c.get("id"):
                        print(f"  {c['short_id']:<20} score={c['_score']:.3f}  "
                              f"{c['star']}★  pos={c['position']}")
                if result.get("traits"):
                    print("激活羁绊:")
                    for t in result["traits"]:
                        if t.get("level", 0) > 0:
                            print(f"  {t.get('name_en','?')}({t.get('count',0)}人)")
            else:
                print(f"[{layout}] 未识别到英雄 (耗时 {elapsed}ms)")
                if result.get("error"):
                    print(f"  错误: {result['error']}")

