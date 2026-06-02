#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_converter.py
阵容格式转换 + 羁绊计算 + 摘要生成

支持输入格式：
  1. Riot 对局 JSON（participant.units 列表）
  2. 截图识别结果（tft_screen_capture 输出）
  3. 自由文本（英文 ID 列表，需配合 champion DB）
  4. 手动 JSON

输出格式（标准化 JSON）：
  {
    "team_size": int,
    "champions": [{ "id", "short_id", "name_en", "star", "cost", "items", "position" }],
    "traits":    [{ "id", "name_en", "count", "level", "level_name" }],
    "summary":   { "front_row_ratio", "main_carry", "equipment_ok" },
    "equipment_issues": [str],
    "_source": str
  }
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ──────────────────────────────────────────────────────────────
# 辅助：ID 规范化
# ──────────────────────────────────────────────────────────────
def strip_prefix(s: str) -> str:
    return re.sub(r"^(?:TFT(?:Set)?\d*_(?:Item_)?|TFT_)", "", s)


def normalize_champ_id(raw: str, set_num: int = 16) -> str:
    """将任意格式 ID 规范化为 TFT{set_num}_{Name}"""
    raw = raw.strip()
    if re.match(rf"^TFT{set_num}_", raw):
        return raw
    clean = strip_prefix(raw)
    return f"TFT{set_num}_{clean}"


def normalize_item_id(raw: str) -> str:
    """将任意格式装备 ID 规范化为 TFT_Item_{Name}"""
    raw = raw.strip()
    if raw.startswith("TFT_Item_"):
        return raw
    clean = strip_prefix(raw)
    return f"TFT_Item_{clean}"


# ──────────────────────────────────────────────────────────────
# 加载本地数据库（懒加载）
# ──────────────────────────────────────────────────────────────
_champion_db: Optional[Dict] = None
_trait_db:    Optional[Dict] = None
_item_db:     Optional[Dict] = None
_trait_dict:  Optional[Dict] = None


def _load_db():
    global _champion_db, _trait_db, _item_db, _trait_dict
    if _champion_db is not None:
        return

    def _read(path: str) -> Dict:
        p = Path(path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    _champion_db = _read("tft_champion_db.json")
    _trait_db    = _read("tft_trait_db.json")
    _item_db     = _read("tft_item_db.json")
    _trait_dict  = _read("tft_trait_champion_dict.json")


# ──────────────────────────────────────────────────────────────
# 羁绊计算
# ──────────────────────────────────────────────────────────────
def calc_traits(champions: List[Dict], set_num: int = 16) -> List[Dict]:
    """
    根据英雄列表计算激活的羁绊。
    需要 tft_champion_db.json 提供英雄 → 羁绊映射。
    """
    _load_db()

    # 统计每个羁绊的英雄数量
    trait_counts: Dict[str, int] = {}
    for champ in champions:
        champ_id = champ.get("id") or normalize_champ_id(champ.get("name_en", ""), set_num)
        db_entry = _champion_db.get(champ_id, {})
        traits   = db_entry.get("traits", [])
        for t in traits:
            trait_counts[t] = trait_counts.get(t, 0) + 1

    if not trait_counts:
        return []

    # 查找激活等级
    activated: List[Dict] = []
    for short_id, count in trait_counts.items():
        # 在 trait_db 中查找（apiName 或 short_id）
        trait_entry = None
        for key, val in (_trait_db or {}).items():
            if val.get("short_id") == short_id or key == short_id:
                trait_entry = val
                break
        # 也在 trait_dict 中查找
        if not trait_entry and _trait_dict:
            trait_entry_td = _trait_dict.get(short_id, {})
            if trait_entry_td:
                levels = trait_entry_td.get("activation", {}).get("levels", [])
                name_en = trait_entry_td.get("name_en", short_id)
                trait_entry = {"name_en": name_en, "levels": levels, "short_id": short_id}

        if not trait_entry:
            continue

        levels = trait_entry.get("levels", [])
        name_en = trait_entry.get("name_en", short_id)
        api_id  = trait_entry.get("id", short_id)

        # 确定激活等级
        active_level = 0
        level_name   = ""
        for lvl in sorted(levels):
            if count >= lvl:
                active_level = lvl
        if active_level > 0:
            level_idx = sorted(levels).index(active_level)
            level_names = ["Bronze", "Silver", "Gold", "Prismatic"]
            level_name  = level_names[min(level_idx, len(level_names)-1)]

        activated.append({
            "id"        : api_id,
            "short_id"  : short_id,
            "name_en"   : name_en,
            "count"     : count,
            "level"     : active_level,
            "level_name": level_name,
            "thresholds": levels,
        })

    # 按激活等级 + 人数排序
    activated.sort(key=lambda x: (-x["level"], -x["count"]))
    return activated


# ──────────────────────────────────────────────────────────────
# 摘要生成
# ──────────────────────────────────────────────────────────────
from typing import Dict, List, Tuple
from collections import Counter

def build_summary(champions: List[Dict], traits: List[Dict]) -> Tuple[Dict, List[str]]:
    """
    生成阵容摘要和装备问题列表。
    返回 (summary_dict, issues_list)
    """
    issues: List[str] = []
    total  = len(champions)

    # 🟢 核心修改：检查是否存在有效的位置信息
    # lineup / global 模式下 position 为 None，此时直接默认站位合理
    has_position = any(c.get("position") for c in champions)
    
    if has_position:
        front_rows = sum(1 for c in champions if c.get("position", {}).get("row", 0) >= 3)
        front_ratio = f"{front_rows}/{total}"
    else:
        # 无位置信息时（如 lineup），默认站位无问题，避免误报或显示无意义的 0/N
        front_ratio = "默认合理 (无位置数据)"

    # 主C：有装备的最高费英雄
    main_carry = ""
    max_cost = -1
    for c in champions:
        cost = c.get("cost", 0)
        if c.get("items") and cost >= max_cost:
            max_cost   = cost
            main_carry = c.get("name_en") or c.get("id", "")

    # 装备问题检查
    # 1. 高费英雄无装备
    for c in champions:
        if c.get("cost", 0) >= 4 and not c.get("items"):
            name = c.get("name_en") or c.get("id", "?")
            issues.append(f"{name}(费用≥4) 无装备")

    # 2. 同一装备重复（除了部分可叠加的）
    STACKABLE = {"TFT_Item_TitansResolve", "TFT_Item_BlueBuff", "TFT_Item_Morellonomicon"}
    all_items: List[str] = []
    for c in champions:
        all_items.extend(c.get("items", []))
        
    item_counter = Counter(all_items)
    for item, count in item_counter.items():
        if count > 1 and item not in STACKABLE:
            issues.append(f"{item} 重复装备 x{count}")

    equipment_ok = len(issues) == 0

    summary = {
        "front_row_ratio": front_ratio,
        "main_carry"     : main_carry,
        "equipment_ok"   : equipment_ok,
        "total_items"    : len(all_items),
        "champion_count" : total,
    }
    return summary, issues


# ──────────────────────────────────────────────────────────────
# 格式转换器
# ──────────────────────────────────────────────────────────────
def from_riot_json(data: Any, set_num: int = 16) -> Dict:
    """
    将 Riot 对局 API 返回的参与者 units 转换为标准格式。
    data: list of units 或包含 "units"/"champions" key 的 dict
    """
    units = (data if isinstance(data, list)
             else data.get("units", data.get("champions", [])))
    champions: List[Dict] = []
    for u in units:
        raw_id = (u.get("character_id") or u.get("champion_id")
                  or u.get("id") or u.get("name") or "")
        champ_id = normalize_champ_id(raw_id, set_num)
        short_id = strip_prefix(champ_id)

        items = []
        for i in u.get("itemNames", u.get("items", [])):
            if i:
                items.append(normalize_item_id(i))

        champions.append({
            "id"      : champ_id,
            "short_id": short_id,
            "name_en" : short_id,
            "star"    : int(u.get("tier", u.get("star", u.get("rarity", 1)))),
            "cost"    : 0,
            "items"   : items,
            "position": u.get("position", {}),
        })

    # 补全 cost
    _load_db()
    for c in champions:
        db_entry = (_champion_db or {}).get(c["id"], {})
        c["cost"] = db_entry.get("cost", 0)

    traits = calc_traits(champions, set_num)
    summary, issues = build_summary(champions, traits)

    return {
        "team_size"       : len(champions),
        "champions"       : champions,
        "traits"          : traits,
        "summary"         : summary,
        "equipment_issues": issues,
        "_source"         : "riot_json",
    }


def from_text(text: str, set_num: int = 16) -> Dict:
    """
    简单文本解析：尝试从文本中提取英雄 ID 列表。
    支持格式：
      - 逗号/空格分隔的英雄名（英文）
      - 每行一个英雄
    """
    _load_db()
    # 提取所有单词，与 champion_db 中的 short_id 比对
    words = re.findall(r"[A-Za-z][A-Za-z']+", text)
    champions: List[Dict] = []
    found_ids: set = set()

    for word in words:
        # 直接匹配或忽略大小写匹配
        for full_id, entry in (_champion_db or {}).items():
            sid = entry.get("short_id", "")
            if sid.lower() == word.lower() and full_id not in found_ids:
                champions.append({
                    "id"      : full_id,
                    "short_id": sid,
                    "name_en" : sid,
                    "star"    : 1,
                    "cost"    : entry.get("cost", 0),
                    "items"   : [],
                    "position": {},
                })
                found_ids.add(full_id)
                break

    if not champions:
        return {"error": "未能从文本中识别出英雄，请使用英雄英文 ID 或上传 JSON"}

    traits = calc_traits(champions, set_num)
    summary, issues = build_summary(champions, traits)
    return {
        "team_size"       : len(champions),
        "champions"       : champions,
        "traits"          : traits,
        "summary"         : summary,
        "equipment_issues": issues,
        "_source"         : "text",
    }


def from_image(image_bytes: bytes, assets_dir: str = "./tft_assets") -> Dict:
    """截图识别入口，委托给 tft_screen_capture"""
    try:
        from tft_screen_capture import recognize
        result = recognize(image_bytes, assets_dir=assets_dir)
        if not result.get("error"):
            # 补全 cost
            _load_db()
            for c in result.get("champions", []):
                db_entry = (_champion_db or {}).get(c.get("id", ""), {})
                c["cost"] = db_entry.get("cost", 0)
        return result
    except ImportError:
        return {"error": "tft_screen_capture.py 未找到"}
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────────────────────
def convert(source, assets_dir: str = "./tft_assets", set_num: int = 16) -> Dict:
    """
    自动检测输入类型并转换：
      bytes/bytearray → 截图识别
      dict/list       → Riot JSON
      str             → JSON 字符串 或 自由文本
    """
    if isinstance(source, (bytes, bytearray)):
        return from_image(source, assets_dir)

    if isinstance(source, (dict, list)):
        return from_riot_json(source, set_num)

    if isinstance(source, str):
        s = source.strip()
        if s.startswith(("[", "{")):
            try:
                return from_riot_json(json.loads(s), set_num)
            except json.JSONDecodeError:
                pass
        return from_text(s, set_num)

    return {"error": f"不支持的输入类型: {type(source)}"}


def save_analysis(analysis: Dict, path: str = "tft_team_analysis.json") -> bool:
    try:
        Path(path).write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# 兼容旧版调用（tft_screen_capture 中使用的函数名）
# ──────────────────────────────────────────────────────────────
def _calc_traits(champions: List[Dict]) -> List[Dict]:
    return calc_traits(champions)

def _build_summary(champions: List[Dict], traits: List[Dict]) -> Tuple[Dict, List[str]]:
    return build_summary(champions, traits)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python tft_converter.py <input.json|text|image.png>")
        sys.exit(0)

    src_arg = sys.argv[1]
    src_path = Path(src_arg)
    if src_path.exists():
        if src_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            data = src_path.read_bytes()
        else:
            data = src_path.read_text(encoding="utf-8")
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
    else:
        data = src_arg

    result = convert(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
