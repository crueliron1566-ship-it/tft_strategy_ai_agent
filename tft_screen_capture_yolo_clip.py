#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_screen_capture_yolo_clip.py
TFT screenshot recognition pipeline.
Uses YOLO for detection and CLIP for classification.
Supports board, lineup, global, duel and auto modes.
"""
import cv2
import numpy as np
import json
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
ASSETS_DIR = Path("./tft_assets")
CHAMP_DIR = ASSETS_DIR / "champions"
ITEM_DIR = ASSETS_DIR / "items"
YOLO_MODEL = Path("./tft_detector.pt")
CLIP_DEVICE = "cpu"
YOLO_CONF_UNIT = 0.22
YOLO_CONF_ITEM = 0.30
YOLO_CONF_STAR = 0.40
CLIP_CONF_MIN = 0.08
CLIP_TOP2_MARGIN = 0.008
CLIP_ITEM_CONF_MIN = 0.14
CLIP_ITEM_TOP2_MARGIN = 0.020
CLS_UNIT_BOX = "unit_box"
CLS_UNIT_ICON = "unit_icon"
CLS_ITEM = "item_slot"
CLS_STAR = "star_pip"
_yolo_model = None
_clip_model = None
_clip_preprocess = None
_champ_embeddings: Optional[np.ndarray] = None
_champ_names: List[str] = []
_item_embeddings: Optional[np.ndarray] = None
_item_names: List[str] = []
_embeddings_loaded = False
_asset_index_cache: Optional[dict] = None
def _load_yolo():
    """Load the YOLO detector lazily."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Missing ultralytics dependency. Install with: pip install ultralytics")
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO model file not found: {YOLO_MODEL}\n"
            "Follow README or TFT_YOLO.md to train a model, then place tft_detector.pt in the project root."
        )
    _yolo_model = YOLO(str(YOLO_MODEL))
    print(f"[YOLO] loaded model: {YOLO_MODEL}")
    return _yolo_model
def _load_clip():
    """Load the CLIP model lazily."""
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    try:
        import clip as openai_clip
    except ImportError:
        raise ImportError("Missing clip dependency. Install with: pip install git+https://github.com/openai/CLIP.git")
    import torch
    _clip_model, _clip_preprocess = openai_clip.load("ViT-B/32", device=CLIP_DEVICE)
    _clip_model.eval()
    print(f"[CLIP] loaded model: ViT-B/32 on {CLIP_DEVICE}")
    return _clip_model, _clip_preprocess
def _load_asset_index() -> dict:
    global _asset_index_cache
    if _asset_index_cache is not None:
        return _asset_index_cache

    path = ASSETS_DIR / "asset_index.json"
    if not path.exists():
        _asset_index_cache = {}
        return _asset_index_cache

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _asset_index_cache = data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[CLIP] failed to load asset index: {path.name} ({exc})")
        _asset_index_cache = {}
    return _asset_index_cache



def _label_map_for_dir(dir_path: Path) -> Dict[str, str]:
    if dir_path.name not in {"champions", "items"}:
        return {}

    section = _load_asset_index().get(dir_path.name, {})
    if not isinstance(section, dict):
        return {}

    label_map: Dict[str, str] = {}
    for stem, meta in section.items():
        if isinstance(meta, dict):
            mapped = meta.get("id")
            if isinstance(mapped, str) and mapped:
                label_map[str(stem)] = mapped
    return label_map



def _build_clip_embeddings():
    global _champ_embeddings, _champ_names
    global _item_embeddings, _item_names
    global _embeddings_loaded
    if _embeddings_loaded:
        return

    import torch
    from PIL import Image as PILImage

    model, preprocess = _load_clip()

    def _encode_image_assets(dir_path: Path) -> Tuple[np.ndarray, List[str]]:
        if not dir_path.exists():
            return np.array([]), []

        label_map = _label_map_for_dir(dir_path)
        paths = sorted(dir_path.glob("*.png"))
        tensors = []
        labels: List[str] = []
        seen_labels = set()

        for p in paths:
            label = label_map.get(p.stem, p.stem)
            if label in seen_labels:
                continue
            try:
                img = preprocess(PILImage.open(p).convert("RGB"))
                tensors.append(img)
                labels.append(label)
                seen_labels.add(label)
            except Exception as e:
                print(f"[CLIP] failed to encode asset {p.name}: {e}")

        if not tensors:
            return np.array([]), []

        batch = torch.stack(tensors).to(CLIP_DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32), labels

    _champ_embeddings, _champ_names = _encode_image_assets(CHAMP_DIR)
    if _champ_embeddings.size > 0:
        print(f"[CLIP] loaded champion embeddings: {len(_champ_names)}")
    else:
        print("[CLIP] no champion asset embeddings loaded")

    _item_embeddings, _item_names = _encode_image_assets(ITEM_DIR)
    if _item_embeddings.size > 0:
        print(f"[CLIP] loaded item embeddings: {len(_item_names)}")
    else:
        print("[CLIP] no item asset embeddings loaded")

    _embeddings_loaded = True

def _crop_xyxy(
    img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """Crop an image region safely using absolute corner coordinates."""
    h, w = img.shape[:2]
    ix1 = max(0, min(w, int(round(x1))))
    iy1 = max(0, min(h, int(round(y1))))
    ix2 = max(0, min(w, int(round(x2))))
    iy2 = max(0, min(h, int(round(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[iy1:iy2, ix1:ix2].copy()


def _prepare_clip_crop(
    crop_bgr: np.ndarray,
    min_side: int = 96,
    sharpen: bool = False,
    contrast: bool = False,
) -> np.ndarray:
    """Clean and square-pad a crop before sending it to CLIP."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    crop = crop_bgr.copy()
    h, w = crop.shape[:2]
    side = max(h, w)
    bg = int(np.median(crop)) if crop.size > 0 else 0
    square = np.full((side, side, 3), bg, dtype=crop.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    square[y0:y0 + h, x0:x0 + w] = crop

    target_side = max(min_side, side)
    if target_side != side:
        square = cv2.resize(square, (target_side, target_side), interpolation=cv2.INTER_LANCZOS4)

    if contrast:
        lab = cv2.cvtColor(square, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        square = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if sharpen:
        blur = cv2.GaussianBlur(square, (0, 0), 0.8)
        square = cv2.addWeighted(square, 1.14, blur, -0.14, 0)

    return square


def _crop_champion_for_clip(
    img: np.ndarray,
    unit_box: Tuple[int, int, int, int],
    icon_mode: bool = False,
) -> np.ndarray:
    """Trim borders, star area, and item area from a YOLO champion box."""
    x, y, w, h = unit_box
    if icon_mode:
        x1 = x + w * 0.06
        x2 = x + w * 0.94
        y1 = y + h * 0.05
        y2 = y + h * 0.90
    else:
        x1 = x + w * 0.12
        x2 = x + w * 0.88
        y1 = y + h * 0.08
        y2 = y + h * 0.80
    crop = _crop_xyxy(img, x1, y1, x2, y2)
    target_side = 128 if max(w, h) < 90 else 104
    return _prepare_clip_crop(crop, min_side=target_side, sharpen=True, contrast=False)


def _crop_item_for_clip(img: np.ndarray, item_det: Dict) -> np.ndarray:
    """Expand and clean a YOLO item crop before CLIP classification."""
    x, y, w, h = item_det["x"], item_det["y"], item_det["w"], item_det["h"]
    pad = max(2, int(max(w, h) * 0.25))
    crop = _crop_xyxy(img, x - pad, y - pad, x + w + pad, y + h + pad)
    return _prepare_clip_crop(crop, min_side=80, sharpen=True, contrast=True)


def _item_slot_centers(
    unit_box: Tuple[int, int, int, int],
    search_above: bool = False,
) -> List[Tuple[int, float, float]]:
    """Return logical YOLO item-slot anchor positions for one unit."""
    x, y, w, h = unit_box
    xs = [x + w * 0.22, x + w * 0.50, x + w * 0.78]
    if search_above:
        ys = [y - h * 0.18, y + h * 1.10]
    else:
        ys = [y + h * 1.10]

    anchors: List[Tuple[int, float, float]] = []
    for slot_idx, cx in enumerate(xs):
        for cy in ys:
            anchors.append((slot_idx, float(cx), float(cy)))
    return anchors


def _crop_star_hsv_region(
    img: np.ndarray,
    unit_box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Crop a narrow top strip where YOLO star pips are expected."""
    x, y, w, h = unit_box
    return _crop_xyxy(
        img,
        x - w * 0.08,
        y - h * 0.42,
        x + w * 1.08,
        y + h * 0.14,
    )

def _clip_rank_image(
    crop_bgr: np.ndarray,
    embeddings: np.ndarray,
    names: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-K CLIP matches for one crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    if embeddings is None or len(names) == 0:
        return []

    import torch
    from PIL import Image as PILImage

    model, preprocess = _load_clip()

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    img_tensor = preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    img_np = img_feat.cpu().numpy().astype(np.float32)
    sims = (img_np @ embeddings.T).squeeze(0)
    order = np.argsort(-sims)[:max(1, min(top_k, len(names)))]
    return [(names[int(i)], float(sims[int(i)])) for i in order]


def _clip_classify_image(
    crop_bgr: np.ndarray,
    embeddings: np.ndarray,
    names: List[str],
    conf_min: float = CLIP_CONF_MIN,
    margin_min: float = CLIP_TOP2_MARGIN,
) -> Tuple[str, float]:
    """
    Run CLIP image classification and reject low-confidence or low-margin matches.
    """
    ranking = _clip_rank_image(crop_bgr, embeddings, names, top_k=2)
    if not ranking:
        return "", 0.0

    best_name, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else -1.0

    if best_score < conf_min:
        return "", best_score
    if len(ranking) > 1 and (best_score - second_score) < margin_min:
        return "", best_score
    return best_name, best_score

def identify_champion_clip(
    crop: np.ndarray,
    conf_min: Optional[float] = None,
    margin_min: Optional[float] = None,
) -> Tuple[str, float]:
    """Use CLIP to classify one champion portrait crop."""
    _build_clip_embeddings()
    if _champ_embeddings is None:
        return "", 0.0
    if conf_min is None:
        conf_min = CLIP_CONF_MIN
    if margin_min is None:
        margin_min = CLIP_TOP2_MARGIN
    return _clip_classify_image(
        crop,
        _champ_embeddings,
        _champ_names,
        conf_min=conf_min,
        margin_min=margin_min,
    )


def identify_champion_candidates(
    crop: np.ndarray,
    top_k: int = 5,
    conf_min: float = 0.0,
) -> List[Dict]:
    """Return top CLIP candidates for one champion crop."""
    _build_clip_embeddings()
    if _champ_embeddings is None:
        return []
    ranking = _clip_rank_image(crop, _champ_embeddings, _champ_names, top_k=top_k)
    result: List[Dict] = []
    for stem, score in ranking:
        if score < conf_min:
            continue
        result.append({
            "id": stem,
            "short_id": _make_short_id(stem),
            "score": float(score),
        })
    return result

def identify_item_clip(crop: np.ndarray) -> Tuple[str, float]:
    """Use CLIP to classify one item icon crop."""
    _build_clip_embeddings()
    if _item_embeddings is None:
        return "", 0.0
    return _clip_classify_image(
        crop,
        _item_embeddings,
        _item_names,
        conf_min=CLIP_ITEM_CONF_MIN,
        margin_min=CLIP_ITEM_TOP2_MARGIN,
    )


def count_stars_from_pips(
    star_dets: List[Dict],
    unit_box: Tuple[int, int, int, int],
) -> int:
    """
    Count YOLO star-pip detections in a narrow region above one unit.
    Returns 0 when YOLO did not find any plausible pip for this unit.
    """
    if not star_dets:
        return 0
    ux, uy, uw, uh = unit_box
    search_x1 = ux - int(uw * 0.08)
    search_x2 = ux + uw + int(uw * 0.08)
    search_y1 = uy - int(uh * 0.42)
    search_y2 = uy + int(uh * 0.12)

    xs: List[float] = []
    min_gap = max(4.0, uw * 0.10)
    for det in star_dets:
        sx, sy = det["cx"], det["cy"]
        if not (search_x1 <= sx <= search_x2 and search_y1 <= sy <= search_y2):
            continue
        if det["w"] > uw * 0.45 or det["h"] > uh * 0.45:
            continue
        xs.append(float(sx))

    if not xs:
        return 0

    xs.sort()
    count = 1
    last_x = xs[0]
    for sx in xs[1:]:
        if sx - last_x > min_gap:
            count += 1
            last_x = sx
    return max(1, min(3, count))

def detect_star_hsv(region: np.ndarray) -> int:
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



def _run_yolo(img: np.ndarray, conf_thr: float = 0.25) -> List[Dict]:
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


def _box_iou(a: Dict, b: Dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0

    area_a = max(1.0, float(a["w"] * a["h"]))
    area_b = max(1.0, float(b["w"] * b["h"]))
    union = area_a + area_b - inter
    return inter / max(1.0, union)


def _same_unit_detection(a: Dict, b: Dict) -> bool:
    iou = _box_iou(a, b)
    if iou >= 0.28:
        return True

    dx = abs(float(a["cx"]) - float(b["cx"]))
    dy = abs(float(a["cy"]) - float(b["cy"]))
    max_w = max(float(a["w"]), float(b["w"]), 1.0)
    max_h = max(float(a["h"]), float(b["h"]), 1.0)
    area_a = max(1.0, float(a["w"] * a["h"]))
    area_b = max(1.0, float(b["w"] * b["h"]))
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))

    return dx <= max_w * 0.30 and dy <= max_h * 0.30 and area_ratio <= 4.5


def _prefer_unit_detection(current: Dict, candidate: Dict) -> Dict:
    cur_icon = current.get("cls") == CLS_UNIT_ICON
    cand_icon = candidate.get("cls") == CLS_UNIT_ICON
    if cand_icon and not cur_icon:
        return candidate
    if cur_icon and not cand_icon:
        return current
    if candidate.get("conf", 0.0) > current.get("conf", 0.0):
        return candidate
    return current


def _merge_unit_detections(icon_dets: List[Dict], box_dets: List[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    ordered = sorted(
        list(icon_dets) + list(box_dets),
        key=lambda d: (
            0 if d.get("cls") == CLS_UNIT_ICON else 1,
            -float(d.get("conf", 0.0)),
            d.get("cy", 0),
            d.get("cx", 0),
        ),
    )
    for det in ordered:
        match_idx = None
        for idx, kept in enumerate(merged):
            if _same_unit_detection(kept, det):
                match_idx = idx
                break
        if match_idx is None:
            merged.append(det)
        else:
            merged[match_idx] = _prefer_unit_detection(merged[match_idx], det)
    return sorted(merged, key=lambda d: (d.get("cy", 0), d.get("cx", 0)))


def _dedupe_row_units(row_units: List[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    for det in sorted(row_units, key=lambda d: (d.get("x", 0), -float(d.get("conf", 0.0)))):
        match_idx = None
        for idx, kept in enumerate(deduped):
            if _same_unit_detection(kept, det):
                match_idx = idx
                break
        if match_idx is None:
            deduped.append(det)
        else:
            deduped[match_idx] = _prefer_unit_detection(deduped[match_idx], det)
    return sorted(deduped, key=lambda d: d.get("x", 0))

def _champion_repeat_exempt(stem: str) -> bool:
    if not stem:
        return False
    tokens = ("TraitClone", "Clone", "Summon", "Turret", "Mech")
    return any(token in stem for token in tokens)


def _resolve_row_champion_candidates(candidates_by_unit: List[List[Dict]]) -> List[Dict]:
    if not candidates_by_unit:
        return []

    resolved: List[Dict] = [{"id": "", "short_id": "", "score": 0.0} for _ in candidates_by_unit]
    used_counts: Dict[str, int] = {}

    def _margin(cands: List[Dict]) -> float:
        if not cands:
            return -1.0
        if len(cands) == 1:
            return 1.0
        return float(cands[0]["score"] - cands[1]["score"])

    order = sorted(
        range(len(candidates_by_unit)),
        key=lambda i: (
            -_margin(candidates_by_unit[i]),
            -(candidates_by_unit[i][0]["score"] if candidates_by_unit[i] else -1.0),
        ),
    )

    for idx in order:
        cands = candidates_by_unit[idx]
        if not cands:
            continue

        best_choice = None
        best_value = float("-inf")
        for rank, cand in enumerate(cands):
            stem = cand.get("id", "")
            penalty = rank * 0.010
            if stem and not _champion_repeat_exempt(stem):
                penalty += used_counts.get(stem, 0) * 0.080
            value = float(cand.get("score", 0.0)) - penalty
            if best_choice is None or value > best_value:
                best_choice = cand
                best_value = value

        resolved[idx] = dict(best_choice) if best_choice else {"id": "", "short_id": "", "score": 0.0}
        stem = resolved[idx].get("id", "")
        if stem:
            used_counts[stem] = used_counts.get(stem, 0) + 1

    changed = True
    while changed:
        changed = False
        counts: Dict[str, int] = {}
        for cand in resolved:
            stem = cand.get("id", "")
            if stem and not _champion_repeat_exempt(stem):
                counts[stem] = counts.get(stem, 0) + 1

        duplicated = [stem for stem, count in counts.items() if count > 1]
        if not duplicated:
            break

        for stem in duplicated:
            dup_indices = [i for i, cand in enumerate(resolved) if cand.get("id") == stem]
            if len(dup_indices) <= 1:
                continue
            keep_idx = max(dup_indices, key=lambda i: resolved[i].get("score", 0.0))
            swap_indices = sorted(
                [i for i in dup_indices if i != keep_idx],
                key=lambda i: resolved[i].get("score", 0.0),
            )
            for idx in swap_indices:
                current = resolved[idx]
                alts = candidates_by_unit[idx][1:]
                for alt in alts:
                    alt_stem = alt.get("id", "")
                    if not alt_stem:
                        continue
                    if (not _champion_repeat_exempt(alt_stem)) and any(
                        resolved[j].get("id") == alt_stem for j in range(len(resolved)) if j != idx
                    ):
                        continue
                    if float(alt.get("score", 0.0)) + 0.060 < float(current.get("score", 0.0)):
                        continue
                    resolved[idx] = dict(alt)
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break

    return resolved



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


def _auto_mode_candidates(img: np.ndarray) -> List[str]:
    first = detect_screenshot_mode(img)
    if first == "duel":
        ordered = ["duel", "board", "lineup", "global"]
    elif first == "board":
        ordered = ["board", "duel", "lineup", "global"]
    elif first == "lineup":
        ordered = ["lineup", "board", "duel", "global"]
    else:
        ordered = ["global", "lineup", "board", "duel"]

    seen = set()
    result = []
    for mode in ordered:
        if mode not in seen:
            seen.add(mode)
            result.append(mode)
    return result


def _extract_known_champions(result: Dict) -> List[Dict]:
    layout = result.get("_layout")
    if layout == "global":
        known = []
        for player in result.get("players", []):
            known.extend([c for c in player.get("champions", []) if c.get("id")])
        return known
    if layout == "duel":
        known = []
        for board in result.get("boards", []):
            known.extend([c for c in board.get("champions", []) if c.get("id")])
        return known
    return [c for c in result.get("champions", []) if c.get("id")]


def _score_recognition_result(result: Dict) -> float:
    """Score candidate recognition results for auto mode selection."""
    layout = result.get("_layout", "")
    known = _extract_known_champions(result)
    known_count = len(known)
    avg_score = float(np.mean([c.get("_score", 0.0) for c in known])) if known else 0.0
    score = known_count * 10.0 + avg_score * 5.0

    if layout == "duel":
        boards = result.get("boards", [])
        if len(boards) < 2:
            return score - 20.0
        sizes = [int(b.get("team_size", 0)) for b in boards[:2]]
        non_empty = sum(1 for s in sizes if s > 0)
        score += non_empty * 6.0
        if non_empty < 2:
            score -= 14.0
        score -= abs(sizes[0] - sizes[1]) * 2.0
        plausible = sum(1 for s in sizes if 1 <= s <= 10)
        score += plausible * 4.0
        score -= sum(max(0, s - 10) * 6.0 for s in sizes)
    elif layout == "global":
        score += min(len(result.get("players", [])), 8) * 2.0
    elif layout == "lineup":
        boxes = [c.get("_box", []) for c in known if c.get("_box")]
        if boxes:
            centers_y = [b[1] + b[3] / 2.0 for b in boxes]
            median_h = float(np.median([b[3] for b in boxes])) if boxes else 1.0
            spread_y = max(centers_y) - min(centers_y) if len(centers_y) > 1 else 0.0
            if spread_y > median_h * 2.2:
                score -= min(80.0, (spread_y / max(1.0, median_h) - 2.2) * 18.0)
        if known_count > 12:
            score -= (known_count - 12) * 6.0
    else:
        positioned = sum(1 for c in known if c.get("position") is not None)
        score += positioned * 0.5

    return score

def recognize_board(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """Recognize a standard board screenshot."""
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    all_cxs = [d["cx"] for d in unit_dets]
    all_cys = [d["cy"] for d in unit_dets]

    debug_img = img.copy() if debug else None
    champions = []

    for det in unit_dets:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        crop = _crop_champion_for_clip(img, (x, y, w, h), icon_mode=det.get("cls") == CLS_UNIT_ICON)

        stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)
        row, col = infer_grid(det["cx"], det["cy"], all_cxs, all_cys)

        star = count_stars_from_pips(star_dets, (x, y, w, h))
        if star <= 0:
            sr = _crop_star_hsv_region(img, (x, y, w, h))
            star = detect_star_hsv(sr)

        item_names = _items_for_unit(img, item_dets, (x, y, w, h), debug_img=debug_img)

        short = _make_short_id(stem)
        champions.append({
            "id": stem,
            "short_id": short,
            "name_en": short or f"unknown_{x}_{y}",
            "star": star,
            "cost": 0,
            "items": item_names,
            "position": {"row": row, "col": col},
            "_score": round(score, 3),
            "_box": [x, y, w, h],
        })

        if debug_img is not None:
            color = (0, 255, 0) if stem else (0, 80, 255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                debug_img,
                f"{short or '?'} {star}* {score:.2f}",
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                debug_img,
                f"[{row},{col}]",
                (x + 2, y + h - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 200, 0),
                1,
            )

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)
        print("[Debug] saved tft_debug.png")

    known = [c for c in champions if c["id"]]
    return _wrap_result(known, champions, "board", t0)

def recognize_lineup(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    unit_dets = _filter(dets, CLS_UNIT_ICON, YOLO_CONF_UNIT)
    if len(unit_dets) < 2:
        unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    unit_dets.sort(key=lambda d: d["x"])

    debug_img = img.copy() if debug else None
    champions = []

    for col_idx, det in enumerate(unit_dets):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        crop = _crop_champion_for_clip(img, (x, y, w, h), icon_mode=det.get("cls") == CLS_UNIT_ICON)

        stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)
        star = count_stars_from_pips(star_dets, (x, y, w, h))
        if star <= 0:
            sr = _crop_star_hsv_region(img, (x, y, w, h))
            star = detect_star_hsv(sr)
        item_names = _items_for_unit(img, item_dets, (x, y, w, h), debug_img=debug_img)

        short = _make_short_id(stem)

        champions.append({
            "id"      : stem,
            "short_id": short,
            "name_en" : short or f"unknown_{col_idx}",
            "star"    : star,
            "cost"    : 0,
            "items"   : item_names,
            "position": None,
            "_order"  : col_idx,
            "_score"  : round(score, 3),
            "_box"    : [x, y, w, h],
        })

        if debug_img is not None:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 200, 255), 2)
            cv2.putText(debug_img, f"{short or '?'} {star}* {score:.2f}",
                        (x, max(0, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1)

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    known = [c for c in champions if c["id"]]
    return _wrap_result(known, champions, "lineup", t0)



def recognize_global(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """Recognize the full scouting overview with multiple player rows."""
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    icon_dets = _filter(dets, CLS_UNIT_ICON, YOLO_CONF_UNIT)
    box_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    unit_dets = _merge_unit_detections(icon_dets, box_dets)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    all_cys = [d["cy"] for d in unit_dets]
    median_h = float(np.median([d["h"] for d in unit_dets])) if unit_dets else 40.0

    def _cluster_by_abs_gap(values: List[float], min_gap: float) -> List[float]:
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

    row_centers = _cluster_by_abs_gap(sorted(all_cys), min_gap=median_h * 0.42)

    max_players = 8
    if len(row_centers) > max_players:
        avg_gap = float(np.median(np.diff(sorted(row_centers)))) if len(row_centers) > 1 else median_h
        tmp: Dict[int, List[Dict]] = {i: [] for i in range(len(row_centers))}
        keep_slack = max(median_h * 0.90, avg_gap * 0.95)
        for det in unit_dets:
            ri = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - det["cy"]))
            if abs(row_centers[ri] - det["cy"]) <= keep_slack:
                tmp[ri].append(det)
        keep_rows = sorted(sorted(range(len(row_centers)), key=lambda i: -len(tmp[i]))[:max_players])
        row_centers = [row_centers[i] for i in keep_rows]

    if not row_centers:
        return {
            "players": [],
            "team_size": 0,
            "champions": [],
            "_source": "yolo_clip_global",
            "_elapsed_ms": int((time.time() - t0) * 1000),
            "_layout": "global",
        }

    avg_row_gap = float(np.median(np.diff(sorted(row_centers)))) if len(row_centers) > 1 else median_h
    row_match_slack = max(median_h * 1.05, avg_row_gap * 1.05)
    row_buckets: Dict[int, List[Dict]] = {i: [] for i in range(len(row_centers))}
    for det in unit_dets:
        ri = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - det["cy"]))
        if abs(row_centers[ri] - det["cy"]) <= row_match_slack:
            row_buckets[ri].append(det)

    debug_img = img.copy() if debug else None
    players = []

    if debug_img is not None:
        for det in item_dets:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 180, 0), 1)

    for rank, (row_idx, row_units) in enumerate(sorted(row_buckets.items()), 1):
        row_units = _dedupe_row_units(row_units)
        row_center = int(round(row_centers[row_idx]))
        champions = []

        if debug_img is not None:
            cv2.line(debug_img, (0, row_center), (debug_img.shape[1] - 1, row_center), (80, 80, 255), 1)
            cv2.putText(
                debug_img,
                f"rank {rank} units {len(row_units)}",
                (8, max(12, row_center - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (80, 80, 255),
                1,
            )

        row_entries = []
        for col_idx, det in enumerate(row_units):
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            crop = _crop_champion_for_clip(img, (x, y, w, h), icon_mode=det.get("cls") == CLS_UNIT_ICON)
            candidates = (
                identify_champion_candidates(crop, top_k=5, conf_min=CLIP_CONF_MIN * 0.60)
                if crop.size > 0
                else []
            )
            row_entries.append({"det": det, "candidates": candidates})

        resolved_candidates = _resolve_row_champion_candidates([entry["candidates"] for entry in row_entries])

        for col_idx, entry in enumerate(row_entries):
            det = entry["det"]
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            resolved = resolved_candidates[col_idx] if col_idx < len(resolved_candidates) else {"id": "", "short_id": "", "score": 0.0}
            stem = resolved.get("id", "")
            score = float(resolved.get("score", 0.0))

            star = count_stars_from_pips(star_dets, (x, y, w, h))
            if star <= 0:
                sr = _crop_star_hsv_region(img, (x, y, w, h))
                star = detect_star_hsv(sr)

            item_names = _items_for_unit(img, item_dets, (x, y, w, h), search_above=True, debug_img=debug_img)
            short = resolved.get("short_id", "") or _make_short_id(stem)
            champions.append({
                "id": stem,
                "short_id": short,
                "name_en": short or f"unknown_{rank}_{col_idx}",
                "star": star,
                "cost": 0,
                "items": item_names,
                "position": None,
                "_order": col_idx,
                "_score": round(score, 3),
                "_box": [x, y, w, h],
                "_clip_candidates": entry["candidates"][:3],
            })

            if debug_img is not None:
                color = (0, 210, 0) if stem else ((0, 200, 255) if det.get("cls") == CLS_UNIT_ICON else (0, 140, 255))
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(
                    debug_img,
                    f"{short[:6] or '?'} {score:.2f}",
                    (x, max(0, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    color,
                    1,
                )

        known = [c for c in champions if c["id"]]
        traits = _try_calc_traits(known)
        players.append({
            "rank": rank,
            "team_size": len(known),
            "champions": champions,
            "traits": traits,
            "_raw_unit_count": len(row_units),
            "_row_center": row_center,
        })

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    my_team = players[0]["champions"] if players else []
    return {
        "team_size": len([c for c in my_team if c["id"]]),
        "champions": my_team,
        "traits": players[0]["traits"] if players else [],
        "players": players,
        "_source": "yolo_clip_global",
        "_elapsed_ms": int((time.time() - t0) * 1000),
        "_layout": "global",
    }

def _split_duel_boards(unit_dets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split detections into upper and lower boards for double-up screenshots."""
    if len(unit_dets) <= 1:
        return unit_dets, []

    sorted_dets = sorted(unit_dets, key=lambda d: d["cy"])
    median_h = float(np.median([d["h"] for d in sorted_dets])) if sorted_dets else 40.0

    best_gap = -1.0
    best_score = float("-inf")
    split_idx = None
    for i in range(len(sorted_dets) - 1):
        gap = sorted_dets[i + 1]["cy"] - sorted_dets[i]["cy"]
        top_count = i + 1
        bottom_count = len(sorted_dets) - top_count
        if top_count < 2 or bottom_count < 2:
            continue

        score = (gap / max(1.0, median_h)) * 3.0
        score -= abs(top_count - bottom_count) * 0.6
        score -= max(0, top_count - 10) * 3.5
        score -= max(0, bottom_count - 10) * 3.5
        if gap >= median_h * 0.75:
            score += 1.0

        if score > best_score:
            best_score = score
            best_gap = gap
            split_idx = top_count

    if split_idx is not None and best_gap >= median_h * 0.75:
        return sorted_dets[:split_idx], sorted_dets[split_idx:]

    med_y = float(np.median([d["cy"] for d in sorted_dets]))
    top = [d for d in sorted_dets if d["cy"] <= med_y]
    bottom = [d for d in sorted_dets if d["cy"] > med_y]
    return top, bottom

def recognize_duel(
    img: np.ndarray,
    debug: bool = False,
) -> Dict:
    """Recognize a double-up screenshot with two vertically separated boards."""
    t0 = time.time()
    dets = _run_yolo(img, conf_thr=YOLO_CONF_UNIT)

    unit_dets = _filter(dets, CLS_UNIT_BOX, YOLO_CONF_UNIT)
    item_dets = _filter(dets, CLS_ITEM, YOLO_CONF_ITEM)
    star_dets = _filter(dets, CLS_STAR, YOLO_CONF_STAR)

    if not unit_dets:
        return {
            "boards": [],
            "team_size": 0,
            "champions": [],
            "_source": "yolo_clip_duel",
            "_elapsed_ms": int((time.time() - t0) * 1000),
            "_layout": "duel",
        }

    top, bottom = _split_duel_boards(unit_dets)

    debug_img = img.copy() if debug else None
    boards = []

    for board_idx, group in enumerate([top, bottom]):
        if group:
            row_centers = _cluster_1d(sorted(set(d["cy"] for d in group)), gap_ratio=0.2)
        else:
            row_centers = []
        row_buckets: Dict[int, List[Dict]] = {i: [] for i in range(len(row_centers))}
        for det in group:
            if not row_centers:
                continue
            row_idx = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - det["cy"]))
            row_buckets[row_idx].append(det)

        positions = {}
        ordered_group: List[Dict] = []
        for row_idx in sorted(row_buckets):
            row_units = sorted(row_buckets[row_idx], key=lambda d: d["x"])
            for col_idx, det in enumerate(row_units):
                positions[id(det)] = (row_idx, col_idx)
                ordered_group.append(det)

        champions = []
        for det in ordered_group:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            crop = _crop_champion_for_clip(img, (x, y, w, h), icon_mode=det.get("cls") == CLS_UNIT_ICON)
            stem, score = identify_champion_clip(crop) if crop.size > 0 else ("", 0.0)
            row, col = positions.get(id(det), (0, 0))
            star = count_stars_from_pips(star_dets, (x, y, w, h))
            if star <= 0:
                sr = _crop_star_hsv_region(img, (x, y, w, h))
                star = detect_star_hsv(sr)
            item_names = _items_for_unit(img, item_dets, (x, y, w, h), debug_img=debug_img)
            short = _make_short_id(stem)
            champions.append({
                "id": stem,
                "short_id": short,
                "name_en": short or f"unknown_{x}_{y}",
                "star": star,
                "cost": 0,
                "items": item_names,
                "position": {"row": row, "col": col},
                "_score": round(score, 3),
                "_box": [x, y, w, h],
            })
            if debug_img is not None:
                color = (0, 255, 0) if stem else (0, 80, 255)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    debug_img,
                    f"{short or '?'} [{row},{col}] {score:.2f}",
                    (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (0, 255, 255),
                    1,
                )

        known = [c for c in champions if c["id"]]
        traits = _try_calc_traits(known)
        label = "opponent" if board_idx == 0 else "self"
        boards.append({
            "board_idx": board_idx,
            "label": label,
            "team_size": len(known),
            "champions": champions,
            "traits": traits,
        })

    if debug_img is not None:
        cv2.imwrite("tft_debug.png", debug_img)

    primary_board = boards[1] if len(boards) > 1 else (boards[0] if boards else {
        "board_idx": 0,
        "label": "self",
        "team_size": 0,
        "champions": [],
        "traits": [],
    })
    if len(boards) > 1 and boards[1]["team_size"] == 0 and boards[0]["team_size"] > 0:
        primary_board = boards[0]
    elif len(boards) > 1 and not (1 <= boards[1]["team_size"] <= 10) and (1 <= boards[0]["team_size"] <= 10):
        primary_board = boards[0]
    elif len(boards) > 1 and boards[0]["team_size"] >= boards[1]["team_size"] + 3:
        primary_board = boards[0]

    return {
        "boards": boards,
        "team_size": primary_board["team_size"],
        "champions": primary_board["champions"],
        "traits": primary_board["traits"],
        "_primary_board_idx": primary_board.get("board_idx", 0),
        "_source": "yolo_clip_duel",
        "_elapsed_ms": int((time.time() - t0) * 1000),
        "_layout": "duel",
    }

def _items_for_unit(
    img: np.ndarray,
    item_dets: List[Dict],
    unit_box: Tuple[int, int, int, int],
    search_above: bool = False,
    debug_img: Optional[np.ndarray] = None,
) -> List[str]:
    """Assign YOLO item detections to logical item slots, then classify each slot."""
    ux, uy, uw, uh = unit_box
    anchors = _item_slot_centers(unit_box, search_above=search_above)
    max_dist = max(12.0, max(uw, uh) * (0.58 if search_above else 0.45))

    best_by_slot: Dict[int, Dict] = {}
    for det in item_dets:
        if det["w"] > uw * 0.75 or det["h"] > uh * 0.75:
            continue

        best_slot = None
        best_dist = None
        for slot_idx, cx, cy in anchors:
            dist = float(np.hypot(det["cx"] - cx, det["cy"] - cy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_slot = slot_idx

        if best_slot is None or best_dist is None or best_dist > max_dist:
            continue

        current = best_by_slot.get(best_slot)
        if current is None or best_dist < current["dist"]:
            best_by_slot[best_slot] = {"det": det, "dist": best_dist}

    items = []
    for slot_idx in sorted(best_by_slot):
        det = best_by_slot[slot_idx]["det"]
        crop = _crop_item_for_clip(img, det)
        stem, score = ("", 0.0)
        if crop.size > 0:
            stem, score = identify_item_clip(crop)

        accepted = bool(stem and score >= CLIP_ITEM_CONF_MIN)
        if accepted:
            items.append(stem)

        if debug_img is not None:
            ix, iy, iw, ih = det["x"], det["y"], det["w"], det["h"]
            color = (255, 0, 255) if accepted else (0, 140, 255)
            label = _make_short_id(stem) if stem else "?"
            cv2.rectangle(debug_img, (ix, iy), (ix + iw, iy + ih), color, 1)
            cv2.putText(
                debug_img,
                f"I{slot_idx}:{label} {score:.2f}",
                (ix, min(debug_img.shape[0] - 4, iy + ih + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                color,
                1,
            )

    return items[:3]

def _make_short_id(stem: str) -> str:
    if not stem:
        return ""
    stem = re.sub(r"^TFT\d+_", "", stem)
    stem = re.sub(r"^TFT_", "", stem)
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



def recognize(
    source,
    debug:     bool = False,
    assets_dir: str = None,
    mode:       str = "auto",
) -> Dict:
    global ASSETS_DIR, CHAMP_DIR, ITEM_DIR, _embeddings_loaded, _asset_index_cache
    mode = (mode or "auto").strip().lower()
    if assets_dir:
        ASSETS_DIR = Path(assets_dir)
        CHAMP_DIR  = ASSETS_DIR / "champions"
        ITEM_DIR   = ASSETS_DIR / "items"
        _embeddings_loaded = False
        _asset_index_cache = None
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            return {"error": f"Failed to read image: {source}"}
    elif isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "failed to decode image bytes"}
    elif isinstance(source, np.ndarray):
        img = source
    else:
        return {"error": f"Unsupported input type: {type(source)}"}

    if mode == "auto":
        candidates = _auto_mode_candidates(img)
        print(f"[Auto] mode candidates: {candidates}")
        trial_results = []
        for candidate in candidates:
            if candidate == "lineup":
                result = recognize_lineup(img, debug)
            elif candidate == "global":
                result = recognize_global(img, debug)
            elif candidate == "duel":
                result = recognize_duel(img, debug)
            else:
                result = recognize_board(img, debug)
            score = _score_recognition_result(result)
            result["_auto_score"] = round(score, 3)
            trial_results.append((score, candidate, result))

        best_score, best_mode, best_result = max(trial_results, key=lambda x: x[0])
        best_result["_auto_candidates"] = [
            {"mode": candidate, "score": round(score, 3)}
            for score, candidate, _ in sorted(trial_results, key=lambda x: x[0], reverse=True)
        ]
        best_result["_auto_selected_mode"] = best_mode
        print(f"[Auto] selected mode: {best_mode} (score={best_score:.2f})")
        return best_result

    if mode == "lineup":
        return recognize_lineup(img, debug)
    if mode == "global":
        return recognize_global(img, debug)
    if mode == "duel":
        return recognize_duel(img, debug)
    return recognize_board(img, debug)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TFT screenshot recognition (YOLO + CLIP)"
    )
    ap.add_argument("image", nargs="?", help="screenshot path")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "board", "lineup", "global", "duel"])
    ap.add_argument("--debug", action="store_true", help="write tft_debug.png")
    ap.add_argument("--save", type=str, help="save result to JSON")
    ap.add_argument("--assets-dir", type=str, help="asset directory, default ./tft_assets")
    ap.add_argument("--device", type=str, help="CLIP device: cuda or cpu")
    args = ap.parse_args()

    if not args.image:
        print("Usage: python tft_screen_capture_yolo_clip.py screenshot.png [options]")
        print("  --mode auto|board|lineup|global|duel")
        print("  --debug")
        print("  --save out.json")
        print("  --device cuda")
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
        print(f"Saved result: {args.save}")
    else:
        layout = result.get("_layout", "")
        elapsed = result.get("_elapsed_ms", 0)

        if layout == "global" and result.get("players"):
            print(f"\n[Global] recognized {len(result['players'])} players ({elapsed}ms)")
            for p in result["players"]:
                names = [c["short_id"] for c in p["champions"] if c.get("id")]
                print(f"  rank {p['rank']}: {len(names)} champions -> {names}")

        elif layout == "duel" and result.get("boards"):
            print(f"\n[Duel] recognized two boards ({elapsed}ms)")
            for b in result["boards"]:
                names = [c["short_id"] for c in b.get("champions", []) if c.get("id")]
                label = "opponent" if b["board_idx"] == 0 else "self"
                print(f"  {label}: {len(names)} champions -> {names}")

        else:
            n = result.get("team_size", 0)
            if n > 0:
                print(f"\n[{layout}] recognized {n} champions ({elapsed}ms)")
                for c in sorted(result["champions"], key=lambda x: -x.get("_score", 0)):
                    if c.get("id"):
                        pos = c.get("position")
                        pos_str = f"pos={pos['row']},{pos['col']}" if pos else "(no position)"
                        print(f"  {c['short_id']:<20} score={c['_score']:.3f}  {c['star']}* {pos_str}  items={c['items']}")
            else:
                print(f"[{layout}] no champions recognized ({elapsed}ms)")
                if result.get("error"):
                    print(f"  error: {result['error']}")









