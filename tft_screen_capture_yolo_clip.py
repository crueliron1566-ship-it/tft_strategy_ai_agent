#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_screen_capture_yolo_clip.py
TFT 截图识别引擎（YOLO 定位 + CLIP 分类）

支持四种截图模式（自动检测）：
  board   - Single-Board: 棋盘布局（4×7 六边形网格，DataTFT / 阵容模拟器）
  lineup  - Single-Lineup: 结算简略图（英雄水平一排，无六边形边框）
  global  - Global: 阵容羁绊表（8名玩家小图标，多行）— 不输出位置信息
  duel    - Duel: 战绩回顾（两个棋盘上下叠放）

识别流程：
  1. YOLO 检测器（tft_detector.pt）定位：
       - unit_box   英雄头像框（board/duel/lineup 模式）
       - unit_icon  英雄小图标（global/lineup 模式）
       - item_slot  装备图标
       - star_pip   星星点
  2. 棋盘模式：用检测框中心点 → 聚类推断 (row, col) 棋盘坐标
     缩略图模式（lineup/global）：不输出位置信息
  3. CLIP 零样本分类：
       - 英雄：prompt "a TFT champion portrait of {name}"
       - 装备：prompt "a TFT item icon of {name}"
  4. 星级：对 star_pip 计数（通用），兜底用 HSV 星点检测

用法:
  python tft_screen_capture_yolo_clip.py screenshot.png
  python tft_screen_capture_yolo_clip.py screenshot.png --mode lineup
  python tft_screen_capture_yolo_clip.py screenshot.png --debug
"""

import cv2
import numpy as np
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────
ASSETS_DIR   = Path("./tft_assets")
CHAMP_DIR    = ASSETS_DIR / "champions"    # 英雄头像 PNG（用于构建 CLIP 候选嵌入）
ITEM_DIR     = ASSETS_DIR / "items"        # 装备图标 PNG
YOLO_MODEL   = Path("./tft_detector.pt")   # 训练好的 YOLO 权重
CLIP_DEVICE  = "cpu"                        # "cuda" 或 "cpu"

# YOLO 置信度阈值
YOLO_CONF_UNIT  = 0.30
YOLO_CONF_ITEM  = 0.30
YOLO_CONF_STAR  = 0.40

# CLIP 分类：top-1 置信度低于此值时标记为 unknown
CLIP_CONF_MIN   = 0.10

# YOLO 类别名（与训练标签一致）
CLS_UNIT_BOX  = "unit_box"    # board/duel 模式：带六边形框的英雄
CLS_UNIT_ICON = "unit_icon"   # lineup/global 模式：无框小图标
CLS_ITEM      = "item_slot"   # 装备图标
CLS_STAR      = "star_pip"    # 单颗星星点

# ──────────────────────────────────────────────────────────────
# 延迟加载的全局模型实例
# ──────────────────────────────────────────────────────────────
_yolo_model  = None
_clip_model  = None
_clip_preprocess = None

# CLIP 候选嵌入缓存
_champ_embeddings: Optional[np.ndarray] = None   # shape (N, D)
_champ_names:      List[str]            = []
_item_embeddings:  Optional[np.ndarray] = None
_item_names:       List[str]            = []
_embeddings_loaded = False


# ──────────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────────

def _load_yolo():
    """懒加载 YOLO 模型（ultralytics YOLOv8/v11）"""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics: pip install ultralytics")
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO 权重文件不存在: {YOLO_MODEL}\n"
            "请参考 README 训练或下载 tft_detector.pt"
        )
    _yolo_model = YOLO(str(YOLO_MODEL))
    print(f"[YOLO] 已加载: {YOLO_MODEL}")
    return _yolo_model


def _load_clip():
    """懒加载 CLIP 模型（openai/clip）"""
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    try:
        import clip as openai_clip
    except ImportError:
        raise ImportError("请先安装 clip: pip install git+https://github.com/openai/CLIP.git")
    import torch
    _clip_model, _clip_preprocess = openai_clip.load("ViT-B/32", device=CLIP_DEVICE)
    _clip_model.eval()
    print(f"[CLIP] 已加载 ViT-B/32 on {CLIP_DEVICE}")
    return _clip_model, _clip_preprocess


def _build_clip_embeddings():
    """
    预计算所有英雄 / 装备的 CLIP 图像嵌入，缓存到全局变量。
    直接从 tft_assets/champions 和 tft_assets/items 下的 PNG 读取图像特征。
    """
    global _champ_embeddings, _champ_names
    global _item_embeddings,  _item_names
    global _embeddings_loaded
    if _embeddings_loaded:
        return
    import torch
    import clip as openai_clip
    from PIL import Image as PILImage

    model, preprocess = _load_clip()

    def _encode_image_assets(dir_path: Path) -> Tuple[np.ndarray, List[str]]:
        if not dir_path.exists():
            return np.array([]), []
        paths = sorted(dir_path.glob("*.png"))
        tensors = []
        stems = []
        for p in paths:
            try:
                # CLIP preprocess 接收 PIL RGB 图像
                img = preprocess(PILImage.open(p).convert("RGB"))
                tensors.append(img)
                stems.append(p.stem)
            except Exception as e:
                print(f"[CLIP] 警告: 跳过 {p.name}，加载失败: {e}")
        if not tensors:
            return np.array([]), []
        # 批量编码图像特征
        batch = torch.stack(tensors).to(CLIP_DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 归一化
        return feats.cpu().numpy().astype(np.float32), stems

    # ── 英雄 ──────────────────────────────────────────────────
    _champ_embeddings, _champ_names = _encode_image_assets(CHAMP_DIR)
    if _champ_embeddings.size > 0:
        print(f"[CLIP] 英雄图像特征库: {len(_champ_names)} 个")
    else:
        print("[CLIP] 警告: 未找到英雄模板图片，将无法识别英雄名称")

    # ── 装备 ──────────────────────────────────────────────────
    _item_embeddings, _item_names = _encode_image_assets(ITEM_DIR)
    if _item_embeddings.size > 0:
        print(f"[CLIP] 装备图像特征库: {len(_item_names)} 个")

    _embeddings_loaded = True





# ──────────────────────────────────────────────────────────────
# CLIP 分类核心
# ──────────────────────────────────────────────────────────────

def _clip_classify_image(
    crop_bgr: np.ndarray,
    embeddings: np.ndarray,
    names: List[str],
) -> Tuple[str, float]:
    """
    将裁剪图像与预计算的候选嵌入做余弦相似度匹配。
    返回 (最佳名称, 相似度得分)。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    if embeddings is None or len(names) == 0:
        return "", 0.0

    import torch
    from PIL import Image as PILImage

    model, preprocess = _load_clip()

    # BGR → PIL RGB
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    img_tensor = preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    img_np = img_feat.cpu().numpy().astype(np.float32)          # (1, D)
    sims   = (img_np @ embeddings.T).squeeze(0)                 # (N,)
    best_i = int(np.argmax(sims))
    return names[best_i], float(sims[best_i])


def identify_champion_clip(crop: np.ndarray) -> Tuple[str, float]:
    """用 CLIP 识别英雄，返回 (stem, score)"""
    _build_clip_embeddings()
    if _champ_embeddings is None:
        return "", 0.0
    return _clip_classify_image(crop, _champ_embeddings, _champ_names)


def identify_item_clip(crop: np.ndarray) -> Tuple[str, float]:
    """用 CLIP 识别装备，返回 (stem, score)"""
    _build_clip_embeddings()
    if _item_embeddings is None:
        return "", 0.0
    return _clip_classify_image(crop, _item_embeddings, _item_names)


# ──────────────────────────────────────────────────────────────
# 星级检测（两路融合：YOLO star_pip 计数 + HSV 兜底）
# ──────────────────────────────────────────────────────────────

def count_stars_from_pips(
    star_dets: List[Dict],
    unit_box: Tuple[int, int, int, int],
) -> int:
    """
    从 YOLO 检测到的 star_pip 列表中，统计属于该英雄框的星星数。
    unit_box: (x, y, w, h)
    """
    if not star_dets:
        return 1
    ux, uy, uw, uh = unit_box
    # 搜索区域：英雄框上方 40% + 框内顶部
    search_y1 = max(0, uy - int(uh * 0.45))
    search_y2 = uy + int(uh * 0.35)
    count = 0
    for det in star_dets:
        sx, sy = det["cx"], det["cy"]
        if ux - 8 <= sx <= ux + uw + 8 and search_y1 <= sy <= search_y2:
            count += 1
    return max(1, min(3, count)) if count > 0 else 1


def detect_star_hsv(region: np.ndarray) -> int:
    """HSV 颜色检测星点数量（兜底方法，与原版相同）"""
    if region is None or region.size == 0:
        return 1
    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([38, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))
    combined = cv2.bitwise_or(gold, white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_thr = region.shape[0] * region.shape[1] * 0.003
    count = sum(1 for c in contours if cv2.contourArea(c) >= area_thr)
    return max(1, min(3, count)) if count > 0 else 1


# ──────────────────────────────────────────────────────────────
# YOLO 推理辅助
# ──────────────────────────────────────────────────────────────

def _run_yolo(img: np.ndarray, conf_thr: float = 0.25) -> List[Dict]:
    """
    运行 YOLO 检测，返回检测结果列表。
    每条结果格式：
      { "cls": str, "conf": float,
        "x": int, "y": int, "w": int, "h": int,
        "cx": int, "cy": int }
    """
    model = _load_yolo()
    results = model(img, conf=conf_thr, verbose=False)
    dets = []
    names_map = model.names   # {0: "unit_box", 1: "unit_icon", ...}
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            dets.append({
                "cls" : names_map[cls_id],
                "conf": float(box.conf[0]),
                "x"   : x1, "y": y1,
                "w"   : x2 - x1, "h": y2 - y1,
                "cx"  : (x1 + x2) // 2, "cy": (y1 + y2) // 2,
            })
    return dets


def _filter(dets: List[Dict], cls: str, conf: float = 0.0) -> List[Dict]:
    return [d for d in dets if d["cls"] == cls and d["conf"] >= conf]


# ──────────────────────────────────────────────────────────────
# 棋盘坐标推断（与原版算法相同，只换输入来源）
# ──────────────────────────────────────────────────────────────

def _cluster_1d(values: List[float], gap_ratio: float = 0.4) -> List[float]:
    if not values:
        return []
    sv = sorted(values)
    if len(sv) == 1:
        return sv
    diffs = [sv[i+1] - sv[i] for i in range(len(sv)-1)]
    med_d = float(np.median(diffs)) if diffs else 1.0
    thr = med_d * (1 + gap_ratio)
    groups: List[List[float]] = [[sv[0]]]
    for v in sv[1:]:
        if v - groups[-1][-1] <= thr:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def infer_grid(cx: float, cy: float, all_cxs: List[float], all_cys: List[float]) -> Tuple[int, int]:
    rows = _cluster_1d(sorted(set(all_cys)))
    cols = _cluster_1d(sorted(set(all_cxs)))
    row_idx = min(range(len(rows)), key=lambda i: abs(rows[i] - cy)) if rows else 0
    col_idx = min(range(len(cols)), key=lambda i: abs(cols[i] - cx)) if cols else 0
    return row_idx, col_idx


# ──────────────────────────────────────────────────────────────
# 截图模式自动检测（与原版逻辑一致）
# ──────────────────────────────────────────────────────────────

def detect_screenshot_mode(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    aspect = w / h
    if aspect < 2.1 and h < 500:
        return "board"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    purple_mask = cv2.inRange(hsv, np.array([130, 25, 20]), np.array([175, 200, 120]))
    if purple_mask.sum() / 255 / (h * w) < 0.05:
        return "lineup"
    mid_nav = img[0:60, int(w * 0.40):int(w * 0.65)]
    if (cv2.cvtColor(mid_nav, cv2.COLOR_BGR2GRAY) > 180).sum() > 800:
        return "global"
    return "duel"


# ──────────────────────────────────────────────────────────────
# 模式 1: board — 单棋盘 (4×7)
# ──────────────────────────────────────────────────────────────

def recognize_board(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """
    棋盘模式识别。
    - YOLO 检测 unit_box / item_slot / star_pip
    - CLIP 识别英雄名 + 装备名
    - 棋盘坐标由框中心点聚类推断
    - 输出每个英雄的 position: {row, col}
    """
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM,     YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR,     YOLO_CONF_STAR)

    all_cxs = [d["cx"] for d in unit_dets]
    all_cys = [d["cy"] for d in unit_dets]

    debug_img = img.copy() if debug else None
    champions = []

    for det in unit_dets:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        crop = img[y:y+h, x:x+w]

        stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)
        row, col    = infer_grid(det["cx"], det["cy"], all_cxs, all_cys)

        # 星级：优先用 star_pip 计数
        star = count_stars_from_pips(star_dets, (x, y, w, h))
        if star == 1:   # 兜底 HSV
            sr = img[max(0, y - int(h*0.4)):y+h, x:x+w]
            star = detect_star_hsv(sr)

        # 装备：找在该英雄框正下方的 item_slot
        item_names = _items_for_unit(img, item_dets, (x, y, w, h))

        short = _make_short_id(stem)
        champions.append({
            "id"      : stem,
            "short_id": short,
            "name_en" : short or f"unknown_{x}_{y}",
            "star"    : star,
            "cost"    : 0,
            "items"   : item_names,
            "position": {"row": row, "col": col},
            "_score"  : round(score, 3),
            "_box"    : [x, y, w, h],
        })

        if debug_img is not None:
            color = (0, 255, 0) if stem else (0, 80, 255)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"{short or '?'} {star}★ {score:.2f}",
                        (x, max(0, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.putText(debug_img, f"[{row},{col}]",
                        (x+2, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 200, 0), 1)

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)
        print("[Debug] 标注图已保存: tft_debug.png")

    known = [c for c in champions if c["id"]]
    return _wrap_result(known, champions, "board", t0)


# ──────────────────────────────────────────────────────────────
# 模式 2: lineup — 单人横排缩略图
# ──────────────────────────────────────────────────────────────

def recognize_lineup(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """
    横排缩略图：英雄无六边形框，水平一字排列。
    - YOLO 检测 unit_icon
    - 不输出棋盘坐标（position 返回 null）
    - 输出 col 索引表示从左到右顺序
    """
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    # lineup 优先用 unit_icon，fallback unit_box
    unit_dets = _filter(dets, CLS_UNIT_ICON, YOLO_CONF_UNIT)
    if len(unit_dets) < 2:
        unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    # 按 x 坐标从左到右排序
    unit_dets.sort(key=lambda d: d["x"])

    debug_img = img.copy() if debug else None
    champions = []

    for col_idx, det in enumerate(unit_dets):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        crop = img[y:y+h, x:x+w]

        stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)
        star = count_stars_from_pips(star_dets, (x, y, w, h))
        if star == 1:
            sr = img[max(0, y - int(h*0.4)):y+h, x:x+w]
            star = detect_star_hsv(sr)
        item_names = _items_for_unit(img, item_dets, (x, y, w, h))

        short = _make_short_id(stem)

        # 缩略图模式：position 只记录顺序编号，不含棋盘坐标
        champions.append({
            "id"      : stem,
            "short_id": short,
            "name_en" : short or f"unknown_{col_idx}",
            "star"    : star,
            "cost"    : 0,
            "items"   : item_names,
            "position": None,           # 缩略图无位置信息
            "_order"  : col_idx,
            "_score"  : round(score, 3),
            "_box"    : [x, y, w, h],
        })

        if debug_img is not None:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 200, 255), 2)
            cv2.putText(debug_img, f"{short or '?'} {star}★",
                        (x, max(0, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1)

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    known = [c for c in champions if c["id"]]
    return _wrap_result(known, champions, "lineup", t0)


# ──────────────────────────────────────────────────────────────
# 模式 3: global — 8人全局缩略图（8×n 多行小图标）
# ──────────────────────────────────────────────────────────────

def recognize_global(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """
    全局 8 人缩略图模式。
    - YOLO 检测所有 unit_icon，按 Y 行聚类分配给 8 名玩家
    - 不输出棋盘坐标，只输出英雄名+星级+装备
    - 图标上方/下方注明的星级和装备由 star_pip / item_slot 配对
    - 返回 players 列表，每名玩家包含 champions 列表
    """
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    unit_dets = _filter(dets, CLS_UNIT_ICON, YOLO_CONF_UNIT)
    if not unit_dets:
        unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    # 按 Y 聚类分配行（每名玩家一行，最多 8 行）
    all_cys = [d["cy"] for d in unit_dets]

    # 用图标高度中位数作为绝对间距阈值，比 gap_ratio 更稳定
    median_h = float(np.median([d["h"] for d in unit_dets])) if unit_dets else 40.0

    def _cluster_by_abs_gap(values: List[float], min_gap: float) -> List[float]:
        """按绝对间距阈值聚类：行内 Y 抖动 < min_gap 时归同一行。"""
        if not values:
            return []
        sv = sorted(values)
        groups: List[List[float]] = [[sv[0]]]
        for v in sv[1:]:
            if v - groups[-1][-1] <= min_gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(g)) for g in groups]

    # 行内 Y 抖动通常 < 0.5 倍图标高，行间间距通常 > 0.8 倍图标高
    row_centers = _cluster_by_abs_gap(sorted(all_cys), min_gap=median_h * 0.55)

    # ── 关键修复：global 模式严格限制为最多 8 名玩家 ────────────
    # 超过 8 行时，按各行内 unit 数量降序保留最多的 8 行（排除噪声行）
    MAX_PLAYERS = 8
    if len(row_centers) > MAX_PLAYERS:
        # 先做一次粗分配，统计各行单元格数
        _avg_gap = float(np.median(np.diff(sorted(row_centers)))) if len(row_centers) > 1 else median_h
        _tmp: Dict[int, List[Dict]] = {i: [] for i in range(len(row_centers))}
        for det in unit_dets:
            ri = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - det["cy"]))
            if abs(row_centers[ri] - det["cy"]) < _avg_gap * 0.75:
                _tmp[ri].append(det)
        # 保留 unit 最多的前 8 行，按 Y 升序重排
        top8 = sorted(
            sorted(range(len(row_centers)), key=lambda i: -len(_tmp[i]))[:MAX_PLAYERS]
        )
        row_centers = [row_centers[i] for i in top8]

    if not row_centers:
        return {
            "players"     : [],
            "team_size"   : 0,
            "champions"   : [],
            "_source"     : "yolo_clip_global",
            "_elapsed_ms" : int((time.time() - t0) * 1000),
            "_layout"     : "global",
        }

    # 将每个 unit 分配到最近的行（容忍范围 = 行间距的 75%）
    avg_row_h = float(np.median(np.diff(sorted(row_centers)))) if len(row_centers) > 1 else median_h
    row_buckets: Dict[int, List[Dict]] = {i: [] for i in range(len(row_centers))}
    for det in unit_dets:
        ri = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - det["cy"]))
        if abs(row_centers[ri] - det["cy"]) < avg_row_h * 0.75:
            row_buckets[ri].append(det)

    debug_img = img.copy() if debug else None
    players   = []

    for rank, (row_idx, row_units) in enumerate(sorted(row_buckets.items()), 1):
        row_units.sort(key=lambda d: d["x"])
        champions = []
        for col_idx, det in enumerate(row_units):
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            crop = img[y:y+h, x:x+w]
            stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)

            star = count_stars_from_pips(star_dets, (x, y, w, h))
            if star == 1:
                sr = img[max(0, y - int(h*0.5)):y+h, x:x+w]
                star = detect_star_hsv(sr)

            item_names = _items_for_unit(img, item_dets, (x, y, w, h),
                                         search_above=True)  # global 模式装备可能在图标上方

            short = _make_short_id(stem)
            champions.append({
                "id"      : stem,
                "short_id": short,
                "name_en" : short or f"unknown_{rank}_{col_idx}",
                "star"    : star,
                "cost"    : 0,
                "items"   : item_names,
                "position": None,           # 全局缩略图无棋盘坐标
                "_order"  : col_idx,
                "_score"  : round(score, 3),
                "_box"    : [x, y, w, h],
            })

            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 200, 255), 1)
                cv2.putText(debug_img, short[:5] or "?",
                            (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 200), 1)

        known = [c for c in champions if c["id"]]
        traits = _try_calc_traits(known)
        players.append({
            "rank"     : rank,
            "team_size": len(known),
            "champions": champions,
            "traits"   : traits,
        })

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    my_team = players[0]["champions"] if players else []
    return {
        "team_size"  : len([c for c in my_team if c["id"]]),
        "champions"  : my_team,
        "traits"     : players[0]["traits"] if players else [],
        "players"    : players,
        "_source"    : "yolo_clip_global",
        "_elapsed_ms": int((time.time() - t0) * 1000),
        "_layout"    : "global",
    }


# ──────────────────────────────────────────────────────────────
# 模式 4: duel — 双人对战棋盘（2×4×7）
# ──────────────────────────────────────────────────────────────

def recognize_duel(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """
    双棋盘模式：上下两个 4×7 棋盘。
    - YOLO 检测所有 unit_box，按 Y 坐标中点分成上下两棋盘
    - 每个棋盘独立推断棋盘坐标
    - 输出 boards 列表（两条，board_idx=0 为对手，1 为我方）
    """
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    if not unit_dets:
        return {
            "boards"      : [],
            "team_size"   : 0,
            "champions"   : [],
            "_source"     : "yolo_clip_duel",
            "_elapsed_ms" : int((time.time() - t0) * 1000),
            "_layout"     : "duel",
        }

    # 用 Y 中位数分割上下棋盘
    med_y  = float(np.median([d["cy"] for d in unit_dets]))
    top    = [d for d in unit_dets if d["cy"] <= med_y]
    bottom = [d for d in unit_dets if d["cy"] >  med_y]

    debug_img = img.copy() if debug else None
    boards    = []

    for board_idx, group in enumerate([top, bottom]):
        all_cxs = [d["cx"] for d in group]
        all_cys = [d["cy"] for d in group]
        champions = []
        for det in group:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            crop = img[y:y+h, x:x+w]
            stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)
            row, col    = infer_grid(det["cx"], det["cy"], all_cxs, all_cys)
            star = count_stars_from_pips(star_dets, (x, y, w, h))
            if star == 1:
                sr = img[max(0, y - int(h*0.4)):y+h, x:x+w]
                star = detect_star_hsv(sr)
            item_names = _items_for_unit(img, item_dets, (x, y, w, h))
            short = _make_short_id(stem)
            champions.append({
                "id"      : stem,
                "short_id": short,
                "name_en" : short or f"unknown_{x}_{y}",
                "star"    : star,
                "cost"    : 0,
                "items"   : item_names,
                "position": {"row": row, "col": col},
                "_score"  : round(score, 3),
                "_box"    : [x, y, w, h],
            })
            if debug_img is not None:
                color = (0, 255, 0) if stem else (0, 80, 255)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_img, f"{short or '?'} {star}★",
                            (x, max(0, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)

        known  = [c for c in champions if c["id"]]
        traits = _try_calc_traits(known)
        label  = "opponent" if board_idx == 0 else "self"
        boards.append({
            "board_idx": board_idx,
            "label"    : label,
            "team_size": len(known),
            "champions": champions,
            "traits"   : traits,
        })

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    my_champs = boards[1]["champions"] if len(boards) > 1 else []
    return {
        "boards"      : boards,
        "team_size"   : boards[1]["team_size"] if len(boards) > 1 else 0,
        "champions"   : my_champs,
        "traits"      : boards[1]["traits"] if len(boards) > 1 else [],
        "_source"     : "yolo_clip_duel",
        "_elapsed_ms" : int((time.time() - t0) * 1000),
        "_layout"     : "duel",
    }


# ──────────────────────────────────────────────────────────────
# 装备分配辅助
# ──────────────────────────────────────────────────────────────

def _items_for_unit(
    img: np.ndarray,
    item_dets: List[Dict],
    unit_box: Tuple[int, int, int, int],
    search_above: bool = False,
) -> List[str]:
    """
    从 YOLO item_slot 检测结果中，找属于该英雄的装备。

    装备位置关系：
      board/duel 模式：装备在英雄框正下方（y > unit_y2，x 重叠）
      global 模式（search_above=True）：装备在图标上方或下方

    每个英雄最多 3 件装备，按 X 坐标从左到右排序后送 CLIP 识别。
    """
    ux, uy, uw, uh = unit_box
    ux2, uy2 = ux + uw, uy + uh

    matched = []
    for det in item_dets:
        ix, iy = det["cx"], det["cy"]
        # X 轴重叠：装备中心在英雄框 ±50% 宽度内
        if not (ux - uw * 0.5 <= ix <= ux2 + uw * 0.5):
            continue
        if search_above:
            # global 模式：允许在图标上方或下方
            if not (uy - uh * 1.2 <= iy <= uy2 + uh * 0.8):
                continue
        else:
            # board 模式：只在英雄框正下方
            if not (uy2 - 5 <= iy <= uy2 + uh * 0.6):
                continue
        matched.append(det)

    # 按 X 排序，最多取 3 件
    matched.sort(key=lambda d: d["x"])
    items = []
    for det in matched[:3]:
        ix, iy, iw, ih = det["x"], det["y"], det["w"], det["h"]
        crop = img[iy:iy+ih, ix:ix+iw]
        if crop.size == 0:
            continue
        stem, score = identify_item_clip(crop)
        if stem and score >= CLIP_CONF_MIN:
            items.append(stem)
    return items


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _make_short_id(stem: str) -> str:
    if not stem:
        return ""
    for prefix in ["TFT16_", "TFT15_", "TFT14_", "TFT13_", "TFT12_", "TFT_"]:
        stem = stem.replace(prefix, "")
    return stem


def _try_calc_traits(known: List[Dict]) -> List[Dict]:
    try:
        from tft_converter import calc_traits
        return calc_traits(known)
    except ImportError:
        return []


def _wrap_result(known: List[Dict], all_champs: List[Dict], layout: str, t0: float) -> Dict:
    traits, summary, issues = [], {}, []
    try:
        from tft_converter import calc_traits, build_summary
        traits = calc_traits(known)
        summary, issues = build_summary(known, [])
    except ImportError:
        summary = {"front_row_ratio": f"?/{len(known)}", "main_carry": ""}
    return {
        "team_size"       : len(known),
        "champions"       : all_champs,
        "traits"          : traits,
        "summary"         : summary,
        "equipment_issues": issues,
        "_source"         : f"yolo_clip_{layout}",
        "_elapsed_ms"     : int((time.time() - t0) * 1000),
        "_layout"         : layout,
    }


# ──────────────────────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────────────────────

def recognize(
    source,
    debug:     bool = False,
    assets_dir: str = None,
    mode:       str = "auto",
) -> Dict:
    """
    统一识别入口。

    Args:
        source:     图片路径(str/Path)、bytes 或 np.ndarray
        debug:      是否输出 tft_debug.png
        assets_dir: 模板目录（默认 ./tft_assets）
        mode:       "auto" | "board" | "lineup" | "global" | "duel"
    """
    global ASSETS_DIR, CHAMP_DIR, ITEM_DIR, _embeddings_loaded
    if assets_dir:
        ASSETS_DIR = Path(assets_dir)
        CHAMP_DIR  = ASSETS_DIR / "champions"
        ITEM_DIR   = ASSETS_DIR / "items"
        _embeddings_loaded = False   # 重新加载嵌入

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
        print(f"[自动检测] 截图模式: {mode}")

    if mode == "lineup":
        return recognize_lineup(img, debug)
    if mode == "global":
        return recognize_global(img, debug)
    if mode == "duel":
        return recognize_duel(img, debug)
    return recognize_board(img, debug)


# ──────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TFT 截图识别（YOLO + CLIP，支持 board / lineup / global / duel）"
    )
    ap.add_argument("image",        nargs="?",         help="截图路径")
    ap.add_argument("--mode",       default="auto",
                    choices=["auto","board","lineup","global","duel"])
    ap.add_argument("--debug",      action="store_true", help="输出标注图 tft_debug.png")
    ap.add_argument("--save",       type=str,            help="保存结果到 JSON 文件")
    ap.add_argument("--assets-dir", type=str,            help="模板目录（默认 ./tft_assets）")
    ap.add_argument("--device",     type=str,            help="CLIP 设备 cuda/cpu（默认 cpu）")
    args = ap.parse_args()

    if not args.image:
        print("用法: python tft_screen_capture_yolo_clip.py screenshot.png [选项]")
        print()
        print("  --mode auto      自动检测截图类型（默认）")
        print("  --mode board     棋盘布局（DataTFT / 阵容模拟器）")
        print("  --mode lineup    结算简略横排图（无位置信息）")
        print("  --mode global    全局 8 人缩略图（无位置信息）")
        print("  --mode duel      双棋盘对战图")
        print("  --debug          输出标注图 tft_debug.png")
        print("  --save out.json  保存 JSON 结果")
        print("  --device cuda    使用 GPU 推理（需要 CUDA）")
        sys.exit(0)

    if args.device:
        CLIP_DEVICE = args.device

    result = recognize(
        args.image,
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
        layout  = result.get("_layout", "")
        elapsed = result.get("_elapsed_ms", 0)

        if layout == "global" and result.get("players"):
            print(f"\n[Global] 识别 {len(result['players'])} 名玩家 (耗时 {elapsed}ms)")
            for p in result["players"]:
                names = [c["short_id"] for c in p["champions"] if c.get("id")]
                print(f"  第{p['rank']}名: {len(names)} 英雄 → {names}")

        elif layout == "duel" and result.get("boards"):
            print(f"\n[Duel] 识别两个棋盘 (耗时 {elapsed}ms)")
            for b in result["boards"]:
                names = [c["short_id"] for c in b.get("champions", []) if c.get("id")]
                label = "对手" if b["board_idx"] == 0 else "我方"
                print(f"  {label}: {len(names)} 英雄 → {names}")

        else:
            n = result.get("team_size", 0)
            if n > 0:
                print(f"\n[{layout}] 识别 {n} 名英雄 (耗时 {elapsed}ms)")
                for c in sorted(result["champions"],
                                key=lambda x: -x.get("_score", 0)):
                    if c.get("id"):
                        pos = c.get("position")
                        pos_str = f"pos={pos['row']},{pos['col']}" if pos else "（无位置）"
                        print(f"  {c['short_id']:<20} score={c['_score']:.3f}  "
                              f"{c['star']}★  {pos_str}  装备={c['items']}")
            else:
                print(f"[{layout}] 未识别到英雄 (耗时 {elapsed}ms)")
                if result.get("error"):
                    print(f"  错误: {result['error']}")
