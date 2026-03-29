#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_data_manager.py
从 CommunityDragon / DDragon 拉取赛季数据，生成本地 JSON 词典。

输出文件（全英文 ID）：
  tft_champion_db.json          英雄完整数据（费用/羁绊/属性）
  tft_trait_db.json             羁绊激活阈值
  tft_item_db.json              装备数据
  tft_champion_trait_map.json   apiName → [short_trait_id]
  tft_trait_champion_dict.json  short_trait_id → {champions, activation, name_en}
  tft_meta.json                 版本元信息

用法:
  python tft_data_manager.py           # 拉取当前 Set 16
  python tft_data_manager.py --set 17  # 指定赛季
  python tft_data_manager.py --verify  # 验证已有文件
"""

import json, time, re, sys, argparse, requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

HEADERS    = {"User-Agent": "Mozilla/5.0 TFT-DataManager/2.0"}
DELAY      = 0.4
OUTPUT_DIR = Path(".")


# ──────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────
def get_ddragon_version() -> str:
    try:
        r = requests.get(
            "https://ddragon.leagueoflegends.com/api/versions.json",
            headers=HEADERS, timeout=15,
        )
        v = r.json()[0]
        print(f"  DDragon 版本: {v}")
        return v
    except Exception as e:
        print(f"  DDragon 版本获取失败: {e}")
        return "15.12.1"


def _strip(s: str, n: int) -> str:
    s = re.sub(rf"^TFT{n}_", "", s)
    s = re.sub(r"^TFTSet\d+_", "", s)
    s = re.sub(r"^Set\d+_", "", s)
    s = re.sub(r"^TFT_", "", s)
    return s


# ──────────────────────────────────────────────────────────────
# 1. CommunityDragon（首选）
# ──────────────────────────────────────────────────────────────
def fetch_cdragon(set_number: int) -> Optional[Dict]:
    patches = ["latest", "16.12", "16.10", "16.8", "16.6", "16.5", "16.3", "16.1", "pbe"]
    for patch in patches:
        url = f"https://raw.communitydragon.org/{patch}/cdragon/tft/en_us.json"
        print(f"  CDragon {patch}...", end=" ", flush=True)
        try:
            time.sleep(DELAY)
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}")
                continue
            data = r.json()

            # 尝试 sets dict
            sets = data.get("sets", {})
            sd   = (sets.get(str(set_number))
                    or sets.get(f"set{set_number}")
                    or sets.get(set_number))

            # 顶层 fallback
            if not sd and data.get("champions") and data.get("traits"):
                sd = data

            if sd:
                c, t = sd.get("champions", []), sd.get("traits", [])
                if c and t:
                    print(f"✓ ({len(c)} 英雄, {len(t)} 羁绊)")
                    return {
                        "source"    : "cdragon",
                        "patch"     : patch,
                        "set_number": set_number,
                        "champions" : c,
                        "traits"    : t,
                        "items"     : data.get("items", []),
                    }
                print("空数据")
            else:
                avail = list(sets.keys())[:6]
                print(f"未找到 set{set_number} (现有: {avail})")
        except Exception as e:
            print(f"错误: {e}")
    return None


# ──────────────────────────────────────────────────────────────
# 2. DDragon（备用）
# ──────────────────────────────────────────────────────────────
def fetch_ddragon(set_number: int, version: str) -> Optional[Dict]:
    base   = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US"
    prefix = f"TFT{set_number}_"
    result: Dict = {}
    for ep, key in [
        ("tft-champion.json", "champions"),
        ("tft-trait.json",    "traits"),
        ("tft-item.json",     "items"),
    ]:
        try:
            time.sleep(DELAY)
            r = requests.get(f"{base}/{ep}", headers=HEADERS, timeout=20)
            raw = r.json().get("data", {}) if r.status_code == 200 else {}
            filtered = {k: v for k, v in raw.items()
                        if k.startswith(prefix)
                        or (isinstance(v, dict) and v.get("id", "").startswith(prefix))}
            result[key] = filtered
            print(f"  DDragon {ep}: {len(filtered)} 条")
        except Exception as e:
            print(f"  DDragon {ep} 失败: {e}")
            result[key] = {}

    if any(result.values()):
        result.update({"source": "ddragon", "version": version, "set_number": set_number})
        return result
    return None


# ──────────────────────────────────────────────────────────────
# 3. 解析 CommunityDragon 列表格式
# ──────────────────────────────────────────────────────────────
def parse_cdragon(raw: Dict):
    n = raw["set_number"]
    champ_db, trait_db, item_db, cmap = {}, {}, {}, {}

    # ── 先解析羁绊，建立 name_en → short_id 的反查表 ────────────
    # CommunityDragon 英雄数据中 c['traits'] 存的是显示名（如 "Quickstriker"），
    # 而羁绊的 short_id 是 strip(apiName)（如 "Rapidfire"）。
    # 必须先建立反查表，才能将英雄 traits 列表统一为 short_id，
    # 保证 save_all 里的 `if sid in traits` 能正确匹配。
    name_to_short: Dict[str, str] = {}   # "Quickstriker" -> "Rapidfire"

    for t in raw.get("traits", []):
        if not isinstance(t, dict):
            continue
        api = t.get("apiName", "")
        if not api:
            continue
        short  = _strip(api, n)
        name   = t.get("name", short)
        levels = sorted({e.get("minUnits", 0) for e in t.get("effects", []) if e.get("minUnits", 0) > 0})
        trait_db[api] = {
            "id"      : api,
            "short_id": short,
            "name_en" : name,
            "levels"  : levels,
            "effects" : t.get("effects", []),
        }
        # 双向注册：short_id 和 name_en 都能查到 short_id
        name_to_short[short] = short   # 已是 short_id 的情况（直接命中）
        name_to_short[name]  = short   # 显示名 -> short_id

    # ── 解析英雄，traits 统一转为 short_id ─────────────────────
    for c in raw.get("champions", []):
        if not isinstance(c, dict):
            continue
        api = c.get("apiName", "")
        if not api:
            continue
        short = _strip(api, n)

        # c['traits'] 可能是 short_id（如 "Freljord"）也可能是 display name（如 "Quickstriker"）
        # 通过 name_to_short 统一转换为 short_id
        raw_traits = c.get("traits", [])
        traits = []
        for t in raw_traits:
            t_stripped = _strip(t, n)  # 去掉 TFT16_ 前缀（通常已无前缀）
            # 优先用反查表，找不到则直接用 stripped 值
            traits.append(name_to_short.get(t_stripped, name_to_short.get(t, t_stripped)))

        champ_db[api] = {
            "id"      : api,
            "short_id": short,
            "name_en" : c.get("name", short),
            "cost"    : c.get("cost", 0),
            "traits"  : traits,
            "stats"   : c.get("stats", {}),
        }
        cmap[api] = traits

    for item in raw.get("items", []):
        if not isinstance(item, dict):
            continue
        api = item.get("apiName", item.get("id", ""))
        if not api:
            continue
        item_db[api] = {
            "id"         : api,
            "name_en"    : item.get("name", api),
            "desc"       : item.get("desc", ""),
            "unique"     : item.get("unique", False),
            "composition": item.get("composition", []),
        }

    return champ_db, trait_db, item_db, cmap


# ──────────────────────────────────────────────────────────────
# 4. 解析 DDragon 字典格式
# ──────────────────────────────────────────────────────────────
def parse_ddragon(raw: Dict):
    n = raw["set_number"]
    champ_db, trait_db, item_db, cmap = {}, {}, {}, {}

    # 先解析羁绊，建立 name_en → short_id 反查表（与 parse_cdragon 逻辑一致）
    name_to_short: Dict[str, str] = {}

    for key, t in raw.get("traits", {}).items():
        if not isinstance(t, dict):
            continue
        api    = t.get("id", key)
        short  = _strip(api, n)
        name   = t.get("name", short)
        levels = sorted({s.get("min", 0) for s in t.get("sets", []) if s.get("min", 0) > 0})
        trait_db[api] = {
            "id"      : api,
            "short_id": short,
            "name_en" : name,
            "levels"  : levels,
            "effects" : t.get("sets", []),
        }
        name_to_short[short] = short
        name_to_short[name]  = short

    for key, c in raw.get("champions", {}).items():
        if not isinstance(c, dict):
            continue
        api   = c.get("id", key)
        short = _strip(api, n)

        raw_traits = c.get("traits", [])
        traits = []
        for t in raw_traits:
            t_stripped = _strip(t, n)
            traits.append(name_to_short.get(t_stripped, name_to_short.get(t, t_stripped)))

        champ_db[api] = {
            "id"      : api,
            "short_id": short,
            "name_en" : c.get("name", short),
            "cost"    : c.get("tier", c.get("cost", 0)),
            "traits"  : traits,
            "stats"   : c.get("stats", {}),
        }
        cmap[api] = traits

    for key, item in raw.get("items", {}).items():
        if not isinstance(item, dict):
            continue
        api = item.get("id", key)
        item_db[api] = {
            "id"         : api,
            "name_en"    : item.get("name", api),
            "desc"       : item.get("desc", ""),
            "unique"     : item.get("unique", False),
            "composition": item.get("from", []),
        }

    return champ_db, trait_db, item_db, cmap


# ──────────────────────────────────────────────────────────────
# 5. 写入输出文件
# ──────────────────────────────────────────────────────────────
def save_all(champ_db, trait_db, item_db, cmap, set_number, source_info):
    def write(name: str, data: Dict):
        p = OUTPUT_DIR / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {name}  ({len(data)} 条)")

    print("\n写入文件:")
    write("tft_champion_db.json",       champ_db)
    write("tft_trait_db.json",          trait_db)
    write("tft_item_db.json",           item_db)
    write("tft_champion_trait_map.json", cmap)

    # 兼容旧格式：short_id → {champions, activation}
    # 同时写入 name_en 作为别名 key，使得用显示名查询也能命中
    legacy: Dict = {}
    for api, trait in trait_db.items():
        sid     = trait["short_id"]
        name_en = trait["name_en"]
        members = [cn for cn, traits in cmap.items() if sid in traits]
        entry = {
            "api_name"  : api,
            "short_id"  : sid,
            "name_en"   : name_en,
            "champions" : members,
            "activation": {"levels": trait["levels"]},
        }
        legacy[sid] = entry          # 用内部 short_id 查（如 "Rapidfire"）
        if name_en != sid:
            legacy[name_en] = entry  # 用显示名查（如 "Quickstriker"）也能命中
    write("tft_trait_champion_dict.json", legacy)

    meta = {
        "set_number"     : set_number,
        **source_info,
        "champion_count" : len(champ_db),
        "trait_count"    : len(trait_db),
        "item_count"     : len(item_db),
        "fetched_at"     : datetime.now().isoformat(),
    }
    write("tft_meta.json", meta)
    print(f"\n✅ 完成！输出目录: {OUTPUT_DIR.resolve()}")


# ──────────────────────────────────────────────────────────────
# 6. 验证已有文件
# ──────────────────────────────────────────────────────────────
def verify():
    print("\n🔍 验证本地数据文件:")
    files = {
        "tft_champion_db.json"      : "英雄 DB",
        "tft_trait_db.json"         : "羁绊 DB",
        "tft_item_db.json"          : "装备 DB",
        "tft_champion_trait_map.json": "英雄→羁绊映射",
        "tft_trait_champion_dict.json": "羁绊词典",
        "tft_meta.json"             : "版本元信息",
    }
    all_ok = True
    for fname, label in files.items():
        p = OUTPUT_DIR / fname
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                print(f"  ✓ {label:<20} ({len(data)} 条)")
            except Exception as e:
                print(f"  ✗ {label} — 解析失败: {e}")
                all_ok = False
        else:
            print(f"  ✗ {label:<20} — 文件缺失")
            all_ok = False

    if all_ok:
        print("\n✅ 所有文件完整")
    else:
        print("\n⚠  部分文件缺失，请运行: python tft_data_manager.py")


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def fetch_and_build(set_number: int = 16) -> bool:
    print(f"\n{'='*52}")
    print(f"  TFT Data Manager — Set {set_number}")
    print(f"{'='*52}")

    print("\n[1/2] 从 CommunityDragon 拉取数据...")
    raw = fetch_cdragon(set_number)

    if raw:
        champ_db, trait_db, item_db, cmap = parse_cdragon(raw)
        src = {"source": "cdragon", "patch": raw["patch"]}
    else:
        print("\n[2/2] CDragon 失败，切换到 DDragon...")
        ver = get_ddragon_version()
        raw = fetch_ddragon(set_number, ver)
        if raw:
            champ_db, trait_db, item_db, cmap = parse_ddragon(raw)
            src = {"source": "ddragon", "version": ver}
        else:
            print("❌ 所有数据源失败")
            return False

    if not champ_db:
        print("❌ 未获取到英雄数据")
        return False

    print(f"\n解析结果: {len(champ_db)} 英雄 / {len(trait_db)} 羁绊 / {len(item_db)} 装备")

    # 费用分布
    from collections import Counter
    cost_dist = Counter(c["cost"] for c in champ_db.values())
    print(f"费用分布: {dict(sorted(cost_dist.items()))}")

    save_all(champ_db, trait_db, item_db, cmap, set_number, src)
    return True


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TFT 赛季数据管理器")
    ap.add_argument("--set",        type=int, default=16, help="赛季编号（默认 16）")
    ap.add_argument("--output-dir", type=str, default=".", help="输出目录")
    ap.add_argument("--verify",     action="store_true",  help="验证已有文件")
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify()
        sys.exit(0)

    sys.exit(0 if fetch_and_build(args.set) else 1)
