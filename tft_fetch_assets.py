#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_fetch_assets.py
下载英雄头像和装备图标（模板匹配用）。

【数据驱动】
  英雄列表和装备列表均从 tft_data_manager.py 生成的 JSON 文件读取，
  无需手工维护字典。需先运行：
      python tft_data_manager.py --set 16

下载策略（按优先级）：
  英雄:
    1. DDragon  /img/tft-champion/{apiName}.png
    2. CommunityDragon  /game/assets/characters/{lc_id}/hud/{lc_id}_square.*.png
  装备:
    1. DDragon  /img/tft-item/{apiName}.png   （apiName 即 item_db.json 的 key，
       已含正确前缀，如 TFT5_Item_DeathbladeRadiant、TFT4_Item_OrnnDeathsDefiance）
    2. CommunityDragon  /game/assets/items/icons2d/{lc_id}.png

文件命名规范：
  英雄: {apiName}.png          例: TFT16_Draven.png
  装备: {apiName}.png          例: TFT_Item_Deathblade.png
                                    TFT16_Item_Bilgewater_JollyRoger.png
                                    TFT5_Item_DeathbladeRadiant.png
                                    TFT4_Item_OrnnDeathsDefiance.png

用法:
  python tft_fetch_assets.py               # 下载全部（需先有 JSON 数据库）
  python tft_fetch_assets.py --champs      # 仅英雄
  python tft_fetch_assets.py --items       # 仅装备
  python tft_fetch_assets.py --verify      # 检查完整性
  python tft_fetch_assets.py --list-missing
  python tft_fetch_assets.py --db-dir ./data  # 指定 JSON 数据库目录
"""

import re
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# ──────────────────────────────────────────────────────────────
# 基础配置
# ──────────────────────────────────────────────────────────────
VERSION    = "16.10.1"       # DDragon 版本备用值（自动获取失败时使用）
ASSETS_DIR = Path("./tft_assets")
DB_DIR     = Path(".")        # JSON 数据库目录，可由 --db-dir 覆盖
FORCE      = False            # 由 --force 覆盖，强制重新下载已有文件
ICON_SIZE  = 64               # 英雄头像目标尺寸
ITEM_SIZE  = 36               # 装备图标目标尺寸
DELAY      = 0.2
HEADERS    = {"User-Agent": "Mozilla/5.0 Chrome/124.0 Safari/537.36"}

# CommunityDragon 根 URL
_CDN_CHARS = "https://raw.communitydragon.org/latest/game/assets/characters"
_CDN_ITEMS = "https://raw.communitydragon.org/latest/game/assets/items/icons2d"


# ──────────────────────────────────────────────────────────────
# 装备过滤：只保留在实际游戏中会出现的装备
# ──────────────────────────────────────────────────────────────
# 以下 ID 或包含以下关键词的装备属于内部/教程/旧赛季专属，跳过
_SKIP_RE = re.compile(
    r"Tutorial|Consumable|Assist|Debug|Explorer|Generic|DoubleUp|"
    r"ChampionItem|StatBonus|FirstFree|Tier\d|_Golem_|UnusableSlot|"
    r"EmptyBag|Spatula|SentinelSwarm|ForceOfNature|FryingPan|"
    r"Moonstone|Leviathan|NightHarvester|RadiantVirtue|Shroud|"
    r"SpectralGauntlet|SteraksGage|SupportKnightsVow|"
    r"TacticiansR|TacticiansS|UnstableTreasure|"
    r"AdaptiveHelm|AegisOfTheLegion|BansheesVeil|"
    r"ChonccsChalice|ChonccsCrown|ChonccsSpork|EternalFlame|"
    r"GuinsoosRageblade|LocketOf|Grant|Random|Augment|"
    r"CypherArmory|SetMechanic|RoboRanger|MonsterTrainer|"
    r"CrystalRose|BloodFury|KaynBlue|KaynRed|TreasureDragon|"
    r"GoldenItemRemover|MasterworkUpgrade|GrowingUp|SkipOption|"
    r"TFT7_|TFT11_|TFT14_|TFT15_|_HR$"
)

def _item_is_valid(api_name: str) -> bool:
    """判断一个 item apiName 是否属于需要下载的实际游戏装备。"""
    if _SKIP_RE.search(api_name):
        return False
    # TFT4_* 只保留 Ornn 神器
    if api_name.startswith("TFT4_") and "Ornn" not in api_name:
        return False
    # TFT9_* 只保留 Ornn 神器
    if api_name.startswith("TFT9_") and "Ornn" not in api_name:
        return False
    # 保留的前缀
    return api_name.startswith((
        "TFT_Item_",       # 标准合成/基础装备
        "TFT5_Item_",      # 辐光装备
        "TFT16_Item_",     # Set16 专属（比尔吉沃特/皮尔特沃夫/徽章）
        "TFT16_TheDarkin", # 达肯武器
        "TFT4_Item_Ornn",  # 奥恩神器（Set4 版）
        "TFT9_Item_Ornn",  # 奥恩神器（Set9 版）
    ))


# ──────────────────────────────────────────────────────────────
# 读取 JSON 数据库
# ──────────────────────────────────────────────────────────────
def load_champion_db() -> dict:
    """
    读取 tft_champion_db.json。
    返回 {apiName: {id, short_id, cost, ...}} 字典。
    apiName 格式如 'TFT16_Draven'。
    """
    p = DB_DIR / "tft_champion_db.json"
    if not p.exists():
        print(f"  ✗ 找不到 {p}，请先运行: python tft_data_manager.py")
        return {}
    try:
        db = json.loads(p.read_text(encoding="utf-8"))
        print(f"  英雄数据库: {len(db)} 个英雄  (来自 {p})")
        return db
    except Exception as e:
        print(f"  ✗ 读取 {p} 失败: {e}")
        return {}


def load_item_db() -> dict:
    """
    读取 tft_item_db.json。
    返回 {apiName: {id, name_en, ...}} 字典。
    apiName 已含正确前缀，如 'TFT_Item_Deathblade'、
    'TFT5_Item_DeathbladeRadiant'、'TFT16_Item_Bilgewater_JollyRoger'。
    """
    p = DB_DIR / "tft_item_db.json"
    if not p.exists():
        print(f"  ✗ 找不到 {p}，请先运行: python tft_data_manager.py")
        return {}
    try:
        db = json.loads(p.read_text(encoding="utf-8"))
        # 过滤：只保留实际游戏装备
        filtered = {k: v for k, v in db.items() if _item_is_valid(k)}
        print(f"  装备数据库: {len(filtered)} 个装备  (来自 {p}，原始 {len(db)} 条，已过滤)")
        return filtered
    except Exception as e:
        print(f"  ✗ 读取 {p} 失败: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# HTTP 工具
# ──────────────────────────────────────────────────────────────
def dl(url: str) -> Optional[bytes]:
    import requests as _req
    time.sleep(DELAY)
    try:
        r = _req.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200 and len(r.content) > 300:
            return r.content
    except Exception:
        pass
    return None


def dl_with_url(url: str) -> Tuple[Optional[bytes], str]:
    """同 dl()，但同时返回成功的 URL（用于调试）。"""
    data = dl(url)
    return data, url


def resize_image(data: bytes, size: int) -> bytes:
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data)).convert("RGBA").resize(
            (size, size), Image.LANCZOS
        )
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        return data


def get_ddragon_version() -> str:
    import requests as _req
    try:
        r = _req.get(
            "https://ddragon.leagueoflegends.com/api/versions.json",
            headers=HEADERS, timeout=10,
        )
        return r.json()[0]
    except Exception:
        return VERSION


# ──────────────────────────────────────────────────────────────
# 下载英雄（数据驱动）
# ──────────────────────────────────────────────────────────────
def fetch_champions(ver: str, failed_log: list):
    """
    从 tft_champion_db.json 读取所有英雄 apiName，逐一下载头像。
    apiName 本身就是 DDragon URL 中的文件名茎（TFT16_Draven），
    CDragon 回退则用小写版本推算路径。
    """
    champ_db = load_champion_db()
    if not champ_db:
        return

    out = ASSETS_DIR / "champions"
    out.mkdir(parents=True, exist_ok=True)
    cdn = f"https://ddragon.leagueoflegends.com/cdn/{ver}"
    ok = fail = skip = 0

    print(f"\n🖼  英雄头像 ({len(champ_db)} 个) → {out}")
    for api_name, entry in champ_db.items():
        dest = out / f"{api_name}.png"
        if not FORCE and dest.exists() and dest.stat().st_size > 500:
            skip += 1
            continue

        # short_id: 去掉前缀后的名字，如 'Draven'
        short_id = entry.get("short_id", api_name)

        data = None
        used_url = ""
        # ── DDragon：只尝试 TFT 专属端点（/img/tft-champion/） ─
        # 绝对不回退到 /img/champion/！
        # LoL 英雄图是动作立绘，与游戏内 TFT 六边形头像风格完全不同，
        # 如果用错了模板，NCC 会变成负数，导致识别完全失败。
        for dd_name in _ddragon_champ_candidates(api_name, short_id):
            url = f"{cdn}/img/tft-champion/{dd_name}.png"
            data, used_url = dl_with_url(url)
            if data:
                break

        # ── CommunityDragon 回退 ────────────────────────────────
        # CDragon 提供正确的 TFT 内游戏头像，是最可靠的备用来源
        # 注意：_cdragon_champ_urls 会最后才尝试 _square.png（原皮），
        # 只有当所有 tft_set* 后缀均 404 时才会命中。
        if not data:
            for url in _cdragon_champ_urls(api_name, short_id):
                data, used_url = dl_with_url(url)
                if data:
                    break

        if data:
            # 如果最终命中的是原皮路径（_square.png 无 set 后缀），发出警告
            is_base_skin = used_url.endswith("_square.png") and "tft_set" not in used_url
            warn = "  ⚠ 原皮兜底" if is_base_skin else ""
            dest.write_bytes(resize_image(data, ICON_SIZE))
            print(f"  ✓ {api_name}{warn}")
            ok += 1
        else:
            print(f"  ✗ {api_name}  ← 需手动下载")
            fail += 1
            failed_log.append(("champion", api_name, str(dest)))

    print(f"  结果: ✓{ok}  ✗{fail}  跳过{skip}")


def _ddragon_champ_candidates(api_name: str, short_id: str) -> List[str]:
    """
    生成 DDragon /img/tft-champion/ 端点的候选文件名（不含 .png）。
    只生成 TFT 专属端点的候选，不包含 /img/champion/ 路径。
    DDragon 的文件名有多种历史格式，按可能性从高到低排列。
    """
    # 已知大小写差异（DDragon tft-champion 端点有时不遵循驼峰）
    _CASE_FIXES: dict = {
        "chogath"    : ["ChoGath", "Chogath"],
        "kaisa"      : ["KaiSa",   "Kaisa"],
        "leblanc"    : ["LeBlanc", "Leblanc"],
        "monkeyking" : ["MonkeyKing", "Wukong", "monkeyking"],
        "kogmaw"     : ["KogMaw",  "Kogmaw"],
        "drmundo"    : ["DrMundo", "Drmundo"],
        "jarvaniv"   : ["JarvanIV","Jarvaniv"],
        "masteryi"   : ["MasterYi","Masteryi"],
        "missfortune": ["MissFortune", "Missfortune"],
        "reksai"     : ["RekSai",  "Reksai"],
        "tahmkench"  : ["TahmKench","Tahmkench"],
        "twistedfate": ["TwistedFate","Twistedfate"],
        "xinzhao"    : ["XinZhao", "Xinzhao"],
    }

    lc = short_id.lower()
    # 先用修正后的名字，再用原始 short_id，最后用 apiName 本身
    name_variants = [short_id]
    if lc in _CASE_FIXES:
        name_variants = _CASE_FIXES[lc] + name_variants

    # 每种名字变体，都尝试 TFT16_X、TFT_X 两种前缀
    candidates = []
    for name in name_variants:
        candidates.append(f"TFT16_{name}")
        candidates.append(f"TFT_{name}")

    # apiName 本身最终兜底（如 TFT16_Draven 已在上面，但避免重复也无妨）
    if api_name not in candidates:
        candidates.insert(0, api_name)

    # 去重，保持顺序
    seen: set = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _cdragon_champ_urls(api_name: str, short_id: str) -> List[str]:
    """
    生成 CommunityDragon 的英雄头像备用 URL 列表。
    CDragon 路径格式：
      /game/assets/characters/{lc_id}/hud/{lc_id}_square.{suffix}.png

    关键规则：
      - 优先使用 tft_set16（赛季皮肤）后缀，**绝不**在有 set 后缀可选时降级到
        _square.png（原皮）—— 原皮与 TFT 六边形头像外观完全不同，会破坏模板匹配。
      - 尝试顺序：
          1. tft16_{name} 路径 + tft_set17 / tft_set16 后缀（Set16 专属英雄）
          2. {name} 普通路径 + tft_set17 / tft_set16 后缀（借用 LoL 英雄）
          3. 仅在以上全部失败时，才最后尝试 _square.png（原皮）作为
             最终兜底（宁可有头像也好过完全缺失）
    """
    lc_tft  = f"tft16_{short_id.lower()}"   # Set16 专属角色路径
    lc_base = short_id.lower()               # 普通英雄路径

    # TFT 赛季皮肤后缀，按新到旧排列（tft_set17 为未来兼容预留）
    tft_suffixes = ["tft_set17", "tft_set16", "tft_set15"]

    urls = []
    # 第一优先：tft16_ 专属路径 × 所有 TFT 后缀
    for suffix in tft_suffixes:
        urls.append(f"{_CDN_CHARS}/{lc_tft}/hud/{lc_tft}_square.{suffix}.png")

    # 第二优先：普通英雄路径 × 所有 TFT 后缀（借用 LoL 英雄，如 Draven、Jinx）
    for suffix in tft_suffixes:
        urls.append(f"{_CDN_CHARS}/{lc_base}/hud/{lc_base}_square.{suffix}.png")

    # 最终兜底：原皮图（_square.png），宁可有图也好过完全缺失，但优先级最低
    # 注意：此图为原皮立绘风格，若被识别系统使用可能影响 NCC 精度
    urls.append(f"{_CDN_CHARS}/{lc_tft}/hud/{lc_tft}_square.png")
    urls.append(f"{_CDN_CHARS}/{lc_base}/hud/{lc_base}_square.png")

    return urls


# ──────────────────────────────────────────────────────────────
# 下载装备（数据驱动）
# ──────────────────────────────────────────────────────────────
def fetch_items(ver: str, failed_log: list):
    """
    从 tft_item_db.json 读取所有装备 apiName，逐一下载图标。

    关键设计：item_db 的 key 就是正确的 DDragon 文件名茎，
    因此无需任何手工映射表：
      TFT_Item_Deathblade          -> /tft-item/TFT_Item_Deathblade.png
      TFT5_Item_DeathbladeRadiant  -> /tft-item/TFT5_Item_DeathbladeRadiant.png
      TFT4_Item_OrnnDeathsDefiance -> /tft-item/TFT4_Item_OrnnDeathsDefiance.png
      TFT16_Item_Bilgewater_*      -> /tft-item/TFT16_Item_Bilgewater_*.png
    CDragon 回退：将 apiName 转小写后拼 URL。
    """
    item_db = load_item_db()
    if not item_db:
        return

    out = ASSETS_DIR / "items"
    out.mkdir(parents=True, exist_ok=True)
    cdn = f"https://ddragon.leagueoflegends.com/cdn/{ver}"
    ok = fail = skip = 0

    print(f"\n🗡  装备图标 ({len(item_db)} 个) → {out}")
    for api_name in item_db:
        dest = out / f"{api_name}.png"
        if not FORCE and dest.exists() and dest.stat().st_size > 200:
            skip += 1
            continue

        data = None
        # ── DDragon：直接用 apiName 作文件名茎 ──────────────────
        data = dl(f"{cdn}/img/tft-item/{api_name}.png")

        # ── CommunityDragon 回退 ────────────────────────────────
        if not data:
            lc = api_name.lower()
            for url in [
                f"{_CDN_ITEMS}/{lc}.png",
                # 部分装备在 CDragon 的 tft/tftitems 子目录
                f"https://raw.communitydragon.org/latest/game/assets/tft/tftitems/icons2d/{lc}.png",
            ]:
                data = dl(url)
                if data:
                    break

        if data:
            dest.write_bytes(resize_image(data, ITEM_SIZE))
            print(f"  ✓ {api_name}")
            ok += 1
        else:
            print(f"  ✗ {api_name}")
            fail += 1
            failed_log.append(("item", api_name, str(dest)))

    print(f"  结果: ✓{ok}  ✗{fail}  跳过{skip}")


# ──────────────────────────────────────────────────────────────
# 验证 / 列出缺失
# ──────────────────────────────────────────────────────────────
def verify():
    print("\n🔍 资产完整性检查")
    for sub in ["champions", "items"]:
        d = ASSETS_DIR / sub
        if not d.exists():
            print(f"  ✗ {sub}/ 目录缺失")
            continue
        files  = list(d.glob("*.png"))
        kb     = sum(f.stat().st_size for f in files) / 1024
        tiny   = [f.name for f in files if f.stat().st_size < 300]
        status = f"  ⚠ 可能损坏: {tiny[:3]}" if tiny else "  ✓"
        print(f"  {sub:<12}: {len(files)} 文件  {kb:.0f} KB{status}")

    import importlib.util
    for lib, pkg in [("PIL", "pillow"), ("cv2", "opencv-python")]:
        if importlib.util.find_spec(lib):
            print(f"  ✓ {pkg}")
        else:
            print(f"  ✗ {pkg} 缺失 → pip install {pkg}")


def list_missing():
    print("\n🔍 缺失/损坏文件:")
    found_any = False

    champ_db = load_champion_db()
    for api_name in champ_db:
        p = ASSETS_DIR / "champions" / f"{api_name}.png"
        if not p.exists() or p.stat().st_size < 500:
            print(f"  MISSING champion: {api_name}.png")
            found_any = True

    item_db = load_item_db()
    for api_name in item_db:
        p = ASSETS_DIR / "items" / f"{api_name}.png"
        if not p.exists() or p.stat().st_size < 200:
            print(f"  MISSING item: {api_name}.png")
            found_any = True

    if not found_any:
        print("  所有文件完整 ✓")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import requests  # 确保依赖存在，否则早失败

    ap = argparse.ArgumentParser(
        description="TFT 模板资源下载器（数据驱动，从 tft_champion_db.json / tft_item_db.json 读取列表）"
    )
    ap.add_argument("--champs",       action="store_true", help="仅下载英雄头像")
    ap.add_argument("--items",        action="store_true", help="仅下载装备图标")
    ap.add_argument("--verify",       action="store_true", help="检查完整性")
    ap.add_argument("--list-missing", action="store_true", help="列出缺失文件")
    ap.add_argument("--force",        action="store_true", help="强制重新下载已有文件（修复之前下载的原皮图）")
    ap.add_argument("--version",      default=None,        help="指定 DDragon 版本（默认自动获取）")
    ap.add_argument("--db-dir",       default=".",         help="JSON 数据库目录（默认当前目录）")
    ap.add_argument("--assets-dir",   default=None,        help="模板输出目录（默认 ./tft_assets）")
    args = ap.parse_args()

    # 更新全局路径 / 开关
    DB_DIR = Path(args.db_dir)
    if args.assets_dir:
        ASSETS_DIR = Path(args.assets_dir)
    if args.force:
        FORCE = True

    if args.verify:
        verify()
        sys.exit(0)
    if args.list_missing:
        list_missing()
        sys.exit(0)

    # 检查数据库是否存在
    if not (DB_DIR / "tft_champion_db.json").exists() and \
       not (DB_DIR / "tft_item_db.json").exists():
        print("⚠  未找到数据库文件，请先运行:")
        print("     python tft_data_manager.py --set 16")
        sys.exit(1)

    ver = args.version or get_ddragon_version()
    print(f"DDragon 版本: {ver}")
    print(f"数据库目录: {DB_DIR.resolve()}")
    print(f"输出目录:   {ASSETS_DIR.resolve()}")

    failed: list = []
    do_all = not args.champs and not args.items
    if do_all or args.champs:
        fetch_champions(ver, failed)
    if do_all or args.items:
        fetch_items(ver, failed)

    print(f"\n{'='*52}")
    print(f"✅ 完成！输出: {ASSETS_DIR.resolve()}")

    if failed:
        print(f"\n⚠  {len(failed)} 个文件下载失败，需手动处理:")
        print("   英雄: https://raw.communitydragon.org/latest/game/assets/characters/")
        print("         → 搜索英雄名（小写）→ hud/ → *_square*.png")
        print("   装备: https://raw.communitydragon.org/latest/game/assets/items/icons2d/")
        print()
        for typ, api_name, dest in failed:
            print(f"   [{typ}]  {api_name}")
        print(f"\n   或运行: python tft_fetch_assets.py --list-missing")

    print(f"\n下一步: python tft_screen_capture.py screenshot.png")
