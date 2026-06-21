#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_converter.py

Normalize TFT inputs from Riot JSON, free-form text, and image recognition.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def strip_prefix(s: str) -> str:
    return re.sub(r"^(?:TFT(?:Set)?\d*_(?:Item_)?|TFT_)", "", s or "")


def normalize_champ_id(raw: str, set_num: int = 16) -> str:
    raw = (raw or "").strip()
    if re.match(rf"^TFT{set_num}_", raw):
        return raw
    clean = strip_prefix(raw)
    return f"TFT{set_num}_{clean}" if clean else raw


def normalize_item_id(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("TFT_Item_"):
        return raw
    clean = strip_prefix(raw)
    return f"TFT_Item_{clean}" if clean else raw


_champion_db: Optional[Dict[str, Dict[str, Any]]] = None
_trait_db: Optional[Dict[str, Dict[str, Any]]] = None
_item_db: Optional[Dict[str, Dict[str, Any]]] = None
_trait_dict: Optional[Dict[str, Dict[str, Any]]] = None
_champion_trait_map: Optional[Dict[str, List[str]]] = None


def _load_db() -> None:
    global _champion_db, _trait_db, _item_db, _trait_dict, _champion_trait_map
    if _champion_db is not None:
        return

    def _read(path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    _champion_db = _read("tft_champion_db.json")
    _trait_db = _read("tft_trait_db.json")
    _item_db = _read("tft_item_db.json")
    _trait_dict = _read("tft_trait_champion_dict.json")
    _champion_trait_map = _read("tft_champion_trait_map.json")


def _text_quality_score(text: str) -> float:
    text = text or ""
    cjk = sum(2 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(0.2 for ch in text if ch.isascii() and ch.isalpha())
    bad = (text.count("?") + text.count("�") + text.count("\ufffd")) * 3
    return cjk + latin - bad + len(text) * 0.01


def _fix_mojibake_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    candidates = [text]
    for enc in ("gbk", "gb18030"):
        try:
            repaired = text.encode(enc, errors="ignore").decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if repaired:
            candidates.append(repaired)
    deduped: List[str] = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return max(deduped, key=_text_quality_score)


def _clean_match_text(text: str) -> str:
    text = _fix_mojibake_text(text)
    return re.sub(r'[?�\ufffd]+$', "", text).strip()


def _normalize_lookup_text(text: str) -> str:
    text = _clean_match_text(text).lower()
    text = re.sub(r"[\s\-_'/·.]+", "", text)
    text = re.sub(r"[()（）\[\]{}<>《》,:：，、;；|]+", "", text)
    return text


def _effective_set_number(default: int = 16) -> int:
    meta_path = Path("tft_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            set_number = int(meta.get("set_number", 0) or 0)
            if set_number > 0:
                return set_number
        except Exception:
            pass

    _load_db()
    counts: Dict[int, int] = {}
    for api_name in (_champion_db or {}).keys():
        if not api_name.startswith("TFT") or "_" not in api_name:
            continue
        prefix = api_name.split("_", 1)[0]
        try:
            set_number = int(prefix[3:])
        except ValueError:
            continue
        counts[set_number] = counts.get(set_number, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else default


def _current_set_only(api_name: str, set_num: int) -> bool:
    if not api_name.startswith("TFT") or "_" not in api_name:
        return True
    prefix = api_name.split("_", 1)[0]
    try:
        return int(prefix[3:]) == set_num
    except ValueError:
        return True


def _item_extra_aliases(api_name: str, short_id: str) -> List[str]:
    key = short_id.lower()
    aliases = {
        "bluebuff": ["蓝buff", "蓝霸符"],
        "jeweledgauntlet": ["法爆"],
        "guinsoosrageblade": ["羊刀"],
        "bramblevest": ["反甲"],
        "warmogsarmor": ["狂徒"],
        "dragonsclaw": ["龙牙"],
        "hextechgunblade": ["科技枪"],
        "infinityedge": ["无尽"],
        "lastwhisper": ["轻语"],
        "bloodthirster": ["饮血"],
        "giantslayer": ["巨杀"],
        "handofjustice": ["正义"],
        "ionicspark": ["离子", "离子火花"],
        "sunfirecape": ["日炎"],
        "rabadonsdeathcap": ["帽子", "大帽"],
        "redemption": ["救赎"],
        "spearofshojin": ["青龙刀"],
        "statikkshiv": ["电刀"],
        "titansresolve": ["泰坦"],
        "edgeofnight": ["夜刃"],
        "archangelsstaff": ["大天使"],
        "crownguard": ["冕卫"],
        "evenshroud": ["薄暮"],
        "protectorsvow": ["圣盾誓约", "圣盾使的誓约"],
        "steraksgage": ["血手"],
        "quicksilver": ["水银"],
        "thiefsgloves": ["窃贼手套", "偷偷"],
        "morellonomicon": ["鬼书"],
    }
    return aliases.get(key, aliases.get(api_name.lower(), []))


def _build_champion_lookup_rows(set_num: int) -> List[Dict[str, Any]]:
    _load_db()
    rows: List[Dict[str, Any]] = []
    for api_name, info in (_champion_db or {}).items():
        if not _current_set_only(api_name, set_num):
            continue
        short_id = (info.get("short_id") or strip_prefix(api_name) or api_name).strip()
        display_name = _clean_match_text(
            info.get("name_cn") or info.get("name_zh") or info.get("name_en") or short_id
        ) or short_id
        aliases = {
            api_name,
            strip_prefix(api_name),
            short_id,
            info.get("short_id", ""),
            info.get("name_en", ""),
            info.get("name_cn", ""),
            info.get("name_zh", ""),
            display_name,
        }
        norm_aliases = sorted(
            {_normalize_lookup_text(alias) for alias in aliases if _normalize_lookup_text(alias)},
            key=len,
            reverse=True,
        )
        try:
            cost = int(info.get("cost", 0) or 0)
        except Exception:
            cost = 0
        rows.append({
            "id": api_name,
            "short_id": short_id,
            "display_name": display_name,
            "cost": cost,
            "aliases": norm_aliases,
        })
    return rows


def _build_item_lookup_rows() -> List[Dict[str, Any]]:
    _load_db()
    rows: List[Dict[str, Any]] = []
    for api_name, info in (_item_db or {}).items():
        if not api_name.startswith("TFT_Item_"):
            continue
        short_id = strip_prefix(api_name)
        display_name = _clean_match_text(
            info.get("name_cn") or info.get("name_zh") or info.get("name_en") or short_id
        ) or short_id
        aliases = {
            api_name,
            short_id,
            info.get("name_en", ""),
            info.get("name_cn", ""),
            info.get("name_zh", ""),
            display_name,
            *_item_extra_aliases(api_name, short_id),
        }
        norm_aliases = sorted(
            {_normalize_lookup_text(alias) for alias in aliases if _normalize_lookup_text(alias)},
            key=len,
            reverse=True,
        )
        rows.append({
            "id": api_name,
            "short_id": short_id,
            "display_name": display_name,
            "aliases": norm_aliases,
        })
    return rows


def _score_lookup_alias(query: str, alias: str) -> float:
    if not query or not alias:
        return 0.0
    if query == alias:
        return 1.0
    if query in alias or alias in query:
        return 0.92 + min(len(query), len(alias)) / max(len(query), len(alias), 1) * 0.06
    return 0.0


def _parse_star_value(text: str) -> Optional[int]:
    token = _normalize_lookup_text(text)
    if not token:
        return None
    patterns = (
        (3, ("3星", "3x", "3*", "3star", "star3", "三星", "★★★", "⭐⭐⭐")),
        (2, ("2星", "2x", "2*", "2star", "star2", "二星", "两星", "★★", "⭐⭐")),
        (1, ("1星", "1x", "1*", "1star", "star1", "一星", "★", "⭐")),
    )
    for value, aliases in patterns:
        if any(_normalize_lookup_text(alias) in token for alias in aliases):
            return value
    return None


def _strip_star_text(text: str) -> str:
    cleaned = text or ""
    for pattern in (
        r"[123]\s*(?:星|x|X|\*)",
        r"(?:star|Star)\s*[123]",
        r"[123]\s*(?:star|Star)",
        r"[一二三两]\s*星",
        r"[★☆⭐]+",
    ):
        cleaned = re.sub(pattern, " ", cleaned)
    return cleaned


def _best_fuzzy_row(token: str, rows: List[Dict[str, Any]], used_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
    used_ids = used_ids or set()
    query = _normalize_lookup_text(_strip_star_text(token))
    if not query:
        return None
    best = None
    best_score = 0.0
    for row in rows:
        if row["id"] in used_ids:
            continue
        score = max((_score_lookup_alias(query, alias) for alias in row["aliases"]), default=0.0)
        if score > best_score:
            best = row
            best_score = score
    threshold = 0.70 if re.search(r"[\u4e00-\u9fff]", token) else 0.80
    return best if best is not None and best_score >= threshold else None


def _find_alias_matches(text: str, rows: List[Dict[str, Any]], allow_duplicates: bool = False) -> List[Dict[str, Any]]:
    if not text:
        return []
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        for alias in row["aliases"]:
            if not alias:
                continue
            start = text.find(alias)
            while start != -1:
                candidates.append({
                    "row": row,
                    "start": start,
                    "end": start + len(alias),
                    "length": len(alias),
                })
                start = text.find(alias, start + 1)
    by_start: Dict[int, List[Dict[str, Any]]] = {}
    for match in candidates:
        by_start.setdefault(match["start"], []).append(match)

    selected: List[Dict[str, Any]] = []
    used_ids: set = set()
    pos = 0
    while pos < len(text):
        options = [m for m in by_start.get(pos, []) if allow_duplicates or m["row"]["id"] not in used_ids]
        if options:
            best = max(options, key=lambda item: (item["length"], len(item["row"]["aliases"])))
            selected.append(best)
            used_ids.add(best["row"]["id"])
            pos = best["end"]
        else:
            pos += 1
    return selected


def _extract_item_ids(block_text: str, item_rows: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    item_ids: List[str] = []
    for match in _find_alias_matches(block_text, item_rows, allow_duplicates=False):
        item_id = match["row"]["id"]
        if item_id in item_ids:
            continue
        item_ids.append(item_id)
        if len(item_ids) >= limit:
            break
    return item_ids


def calc_traits(champions: List[Dict[str, Any]], set_num: int = 16) -> List[Dict[str, Any]]:
    _load_db()
    trait_counts: Dict[str, int] = {}
    for champ in champions:
        champ_id = champ.get("id") or normalize_champ_id(champ.get("name_en", ""), set_num)
        db_entry = (_champion_db or {}).get(champ_id, {})
        traits = db_entry.get("traits", [])
        if not traits and isinstance(_champion_trait_map, dict):
            traits = _champion_trait_map.get(champ_id, [])
        for trait in traits:
            trait_counts[trait] = trait_counts.get(trait, 0) + 1

    activated: List[Dict[str, Any]] = []
    for short_id, count in trait_counts.items():
        trait_entry = None
        for key, value in (_trait_db or {}).items():
            if value.get("short_id") == short_id or key == short_id:
                trait_entry = value
                break
        if not trait_entry and _trait_dict:
            td = _trait_dict.get(short_id, {})
            if td:
                trait_entry = {
                    "id": short_id,
                    "short_id": short_id,
                    "name_en": td.get("name_en", short_id),
                    "levels": td.get("activation", {}).get("levels", []),
                }
        if not trait_entry:
            continue

        levels = sorted(int(v) for v in trait_entry.get("levels", []) if str(v).isdigit() or isinstance(v, int))
        active_level = 0
        for lvl in levels:
            if count >= lvl:
                active_level = lvl
        level_name = ""
        if active_level > 0:
            names = ["Bronze", "Silver", "Gold", "Prismatic"]
            level_name = names[min(levels.index(active_level), len(names) - 1)]
        activated.append({
            "id": trait_entry.get("id", short_id),
            "short_id": short_id,
            "name_en": trait_entry.get("name_en", short_id),
            "count": count,
            "level": active_level,
            "level_name": level_name,
            "thresholds": levels,
        })

    activated.sort(key=lambda item: (-item["level"], -item["count"], item["name_en"]))
    return activated


def build_summary(champions: List[Dict[str, Any]], traits: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    total = len(champions)
    has_position = any(c.get("position") for c in champions)
    if has_position:
        front_rows = sum(1 for c in champions if c.get("position", {}).get("row", 0) >= 3)
        front_ratio = f"{front_rows}/{total}"
    else:
        front_ratio = "默认合理 (无位置数据)"

    main_carry = ""
    max_cost = -1
    for champ in champions:
        cost = int(champ.get("cost", 0) or 0)
        if champ.get("items") and cost >= max_cost:
            max_cost = cost
            main_carry = champ.get("name_en") or champ.get("short_id") or champ.get("id", "")

    for champ in champions:
        if int(champ.get("cost", 0) or 0) >= 4 and not champ.get("items"):
            name = champ.get("name_en") or champ.get("id", "?")
            issues.append(f"{name}(费用≥4) 无装备")

    stackable = {"TFT_Item_TitansResolve", "TFT_Item_BlueBuff", "TFT_Item_Morellonomicon"}
    all_items: List[str] = []
    for champ in champions:
        all_items.extend(champ.get("items", []))
    for item_id, count in Counter(all_items).items():
        if count > 1 and item_id not in stackable:
            issues.append(f"{item_id} 重复装备 x{count}")

    summary = {
        "front_row_ratio": front_ratio,
        "main_carry": main_carry,
        "equipment_ok": len(issues) == 0,
        "total_items": len(all_items),
        "champion_count": total,
    }
    return summary, issues


def from_riot_json(data: Any, set_num: int = 16) -> Dict[str, Any]:
    units = data if isinstance(data, list) else data.get("units", data.get("champions", []))
    champions: List[Dict[str, Any]] = []
    for unit in units:
        raw_id = unit.get("character_id") or unit.get("champion_id") or unit.get("id") or unit.get("name") or ""
        champ_id = normalize_champ_id(raw_id, set_num)
        short_id = strip_prefix(champ_id)
        items = [normalize_item_id(item) for item in unit.get("itemNames", unit.get("items", [])) if item]
        champions.append({
            "id": champ_id,
            "short_id": short_id,
            "name_en": short_id,
            "star": int(unit.get("tier", unit.get("star", unit.get("rarity", 1))) or 1),
            "cost": 0,
            "items": items,
            "position": unit.get("position", {}),
        })

    _load_db()
    for champ in champions:
        champ["cost"] = (_champion_db or {}).get(champ["id"], {}).get("cost", 0)

    traits = calc_traits(champions, set_num)
    summary, issues = build_summary(champions, traits)
    return {
        "team_size": len(champions),
        "champions": champions,
        "traits": traits,
        "summary": summary,
        "equipment_issues": issues,
        "_source": "riot_json",
    }


def from_text(text: str, set_num: int = 16) -> Dict[str, Any]:
    _load_db()
    if set_num == 16:
        inferred_set = _effective_set_number(16)
        if inferred_set != 16:
            set_num = inferred_set

    champ_rows = _build_champion_lookup_rows(set_num)
    item_rows = _build_item_lookup_rows()
    raw_text = (text or "").strip()
    normalized = _normalize_lookup_text(raw_text)
    if not normalized:
        return {"error": "未能从文本中识别出英雄，请输入英雄名、星级、装备或 JSON"}

    champions: List[Dict[str, Any]] = []
    champion_matches = _find_alias_matches(normalized, champ_rows, allow_duplicates=False)
    if champion_matches:
        for idx, match in enumerate(champion_matches):
            block_end = champion_matches[idx + 1]["start"] if idx + 1 < len(champion_matches) else len(normalized)
            block_text = normalized[match["start"]:block_end]
            item_text = normalized[match["end"]:block_end]
            row = match["row"]
            champions.append({
                "id": row["id"],
                "short_id": row["short_id"],
                "name_en": row["display_name"] or row["short_id"],
                "star": _parse_star_value(block_text) or 1,
                "cost": row["cost"],
                "items": _extract_item_ids(item_text, item_rows, limit=3),
                "position": {},
            })
    else:
        tokens = [tok for tok in re.split(r"[\s,，、/|；;()（）\[\]{}<>《》\n\r\t]+", raw_text) if tok.strip()]
        used_ids: set = set()
        current: Optional[Dict[str, Any]] = None
        for token in tokens:
            champ_row = _best_fuzzy_row(token, champ_rows, used_ids)
            if champ_row is not None:
                if current is not None:
                    champions.append(current)
                    used_ids.add(current["id"])
                current = {
                    "id": champ_row["id"],
                    "short_id": champ_row["short_id"],
                    "name_en": champ_row["display_name"] or champ_row["short_id"],
                    "star": _parse_star_value(token) or 1,
                    "cost": champ_row["cost"],
                    "items": [],
                    "position": {},
                }
                continue
            if current is None:
                continue
            star = _parse_star_value(token)
            if star is not None:
                current["star"] = max(current["star"], star)
                continue
            item_row = _best_fuzzy_row(token, item_rows)
            if item_row is not None and item_row["id"] not in current["items"] and len(current["items"]) < 3:
                current["items"].append(item_row["id"])
        if current is not None:
            champions.append(current)

    if not champions:
        return {"error": "未能从文本中识别出英雄，请输入中文名、英文名、星级或装备名"}

    traits = calc_traits(champions, set_num)
    summary, issues = build_summary(champions, traits)
    return {
        "team_size": len(champions),
        "champions": champions,
        "traits": traits,
        "summary": summary,
        "equipment_issues": issues,
        "_source": "text",
    }


def from_image(image_bytes: bytes, assets_dir: str = "./tft_assets") -> Dict[str, Any]:
    try:
        try:
            from tft_screen_capture_yolo_clip import recognize  # type: ignore
            result = recognize(image_bytes)
        except ImportError:
            from tft_screen_capture import recognize  # type: ignore
            result = recognize(image_bytes, assets_dir=assets_dir)
        if not result.get("error"):
            _load_db()
            for champ in result.get("champions", []):
                champ["cost"] = (_champion_db or {}).get(champ.get("id", ""), {}).get("cost", 0)
        return result
    except ImportError:
        return {"error": "tft_screen_capture 模块未找到"}
    except Exception as exc:
        return {"error": str(exc)}


def convert(source: Any, assets_dir: str = "./tft_assets", set_num: int = 16) -> Dict[str, Any]:
    if isinstance(source, (bytes, bytearray)):
        return from_image(source, assets_dir)
    if isinstance(source, (dict, list)):
        return from_riot_json(source, set_num)
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.startswith(("[", "{")):
            try:
                return from_riot_json(json.loads(stripped), set_num)
            except json.JSONDecodeError:
                pass
        return from_text(stripped, set_num)
    return {"error": f"不支持的输入类型: {type(source)}"}


def save_analysis(analysis: Dict[str, Any], path: str = "tft_team_analysis.json") -> bool:
    try:
        Path(path).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        print(f"保存失败: {exc}")
        return False


def _calc_traits(champions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return calc_traits(champions)


def _build_summary(champions: List[Dict[str, Any]], traits: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    return build_summary(champions, traits)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python tft_converter.py <input.json|text|image.png>")
        raise SystemExit(0)

    src_arg = sys.argv[1]
    src_path = Path(src_arg)
    if src_path.exists():
        if src_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            result = convert(src_path.read_bytes())
        else:
            result = convert(src_path.read_text(encoding="utf-8", errors="replace"))
    else:
        result = convert(src_arg)

    print(json.dumps(result, ensure_ascii=False, indent=2))
