#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch TFT champion and item template assets.

Champion portraits prefer seasonal TFT assets. Base-skin fallbacks are disabled by
default because they break template matching. Use --allow-base-skin only if you
explicitly want incomplete coverage over correctness.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.exceptions import ProxyError, RequestException, SSLError

VERSION = "16.10.1"
ASSETS_DIR = Path("./tft_assets")
DB_DIR = Path(".")
ASSET_INDEX_NAME = "asset_index.json"
FORCE = False
ALLOW_BASE_SKIN = False
ICON_SIZE = 64
ITEM_SIZE = 36
DELAY = 0.2
HEADERS = {"User-Agent": "Mozilla/5.0 TFT-AssetFetcher/3.0"}
NETWORK_MODE = "auto"

CDRAGON_BASE = "https://raw.communitydragon.org/latest/game/assets"
DDRGON_BASE = "https://ddragon.leagueoflegends.com"

CASE_FIXES: Dict[str, List[str]] = {
    "chogath": ["ChoGath", "Chogath"],
    "drmundo": ["DrMundo", "DrMundo"],
    "jarvaniv": ["JarvanIV", "Jarvaniv"],
    "kaisa": ["KaiSa", "Kaisa"],
    "kogmaw": ["KogMaw", "KogMaw"],
    "leblanc": ["LeBlanc", "Leblanc"],
    "masteryi": ["MasterYi", "Masteryi"],
    "missfortune": ["MissFortune", "Missfortune"],
    "monkeyking": ["MonkeyKing", "Wukong", "Monkeyking"],
    "reksai": ["RekSai", "Reksai"],
    "tahmkench": ["TahmKench", "Tahmkench"],
    "twistedfate": ["TwistedFate", "Twistedfate"],
    "xinzhao": ["XinZhao", "Xinzhao"],
}

NON_PLAYABLE_RE = re.compile(
    r"BlueGolem|TrainingDummy|ArmoryKey|Scuttle|Crab|Dummy|Golem|Krug|Raptor|"
    r"Wolf|Herald|Turret|Minion|Camp|Portal|KeyCompleted|Sentinel|VoidGate|"
    r"DragonTreasure|Spawn|Loot|Test|Debug",
    re.IGNORECASE,
)

SKIP_ITEM_RE = re.compile(
    r"Tutorial|Consumable|Assist|Debug|Explorer|Generic|DoubleUp|ChampionItem|"
    r"StatBonus|FirstFree|Tier\d|_Golem_|UnusableSlot|EmptyBag|Spatula|"
    r"SentinelSwarm|ForceOfNature|FryingPan|Moonstone|Leviathan|NightHarvester|"
    r"RadiantVirtue|Shroud|SpectralGauntlet|SteraksGage|SupportKnightsVow|"
    r"TacticiansR|TacticiansS|UnstableTreasure|AdaptiveHelm|AegisOfTheLegion|"
    r"BansheesVeil|ChonccsChalice|ChonccsCrown|ChonccsSpork|EternalFlame|"
    r"GuinsoosRageblade|LocketOf|Grant|Random|Augment|CypherArmory|SetMechanic|"
    r"RoboRanger|MonsterTrainer|CrystalRose|BloodFury|KaynBlue|KaynRed|"
    r"TreasureDragon|GoldenItemRemover|MasterworkUpgrade|GrowingUp|SkipOption|"
    r"TFT7_|TFT11_|TFT14_|TFT15_|_HR$"
)

INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SPACE_RE = re.compile(r"\s+")
DISPLAY_NAME_KEYS = ("name_cn", "name_zh", "name", "display_name", "name_en")


def asset_index_path() -> Path:
    return ASSETS_DIR / ASSET_INDEX_NAME


def has_proxy_env() -> bool:
    return any(os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))


def build_session(trust_env: bool) -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.trust_env = trust_env
    return session


def http_get(url: str, timeout: int, label: str) -> requests.Response:
    modes = {
        "auto": [True, False] if has_proxy_env() else [False],
        "env": [True],
        "direct": [False],
    }[NETWORK_MODE]

    last_error: Optional[Exception] = None
    for idx, trust_env in enumerate(modes):
        try:
            return build_session(trust_env).get(url, timeout=timeout)
        except (ProxyError, SSLError) as exc:
            last_error = exc
            if trust_env and idx < len(modes) - 1:
                print(f"  {label}: proxy failed, retrying direct...")
                continue
            raise
        except RequestException as exc:
            last_error = exc
            if trust_env and idx < len(modes) - 1 and any(token in str(exc).lower() for token in ("proxy", "socks")):
                print(f"  {label}: proxy request failed, retrying direct...")
                continue
            raise

    raise RuntimeError(f"{label} failed: {last_error or url}")


def dl(url: str) -> Optional[bytes]:
    time.sleep(DELAY)
    try:
        response = http_get(url, timeout=15, label="asset")
    except Exception:
        return None
    if response.status_code == 200 and len(response.content) > 300:
        return response.content
    return None


def dl_with_url(url: str) -> Tuple[Optional[bytes], str]:
    return dl(url), url


def resize_image(data: bytes, size: int) -> bytes:
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGBA")
        image = image.resize((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()
    except Exception:
        return data


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_asset_index() -> dict:
    path = asset_index_path()
    if not path.exists():
        return {"version": 1, "champions": {}, "items": {}}
    try:
        data = load_json(path)
    except Exception:
        return {"version": 1, "champions": {}, "items": {}}
    if not isinstance(data, dict):
        return {"version": 1, "champions": {}, "items": {}}
    data.setdefault("version", 1)
    data.setdefault("champions", {})
    data.setdefault("items", {})
    return data


def save_asset_index(index: dict) -> None:
    path = asset_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def load_champion_db() -> dict:
    path = DB_DIR / "tft_champion_db.json"
    if not path.exists():
        print(f"Missing champion DB: {path}")
        return {}
    data = load_json(path)
    print(f"Champion DB: {len(data)} entries from {path}")
    return data


def item_is_valid(api_name: str) -> bool:
    if SKIP_ITEM_RE.search(api_name):
        return False
    if api_name.startswith("TFT4_") and "Ornn" not in api_name:
        return False
    if api_name.startswith("TFT9_") and "Ornn" not in api_name:
        return False
    if api_name.startswith("TFT_Item_"):
        return True
    if re.match(r"^TFT\d+_Item_", api_name):
        return True
    if re.match(r"^TFT\d+_TheDarkin", api_name):
        return True
    return api_name.startswith(("TFT4_Item_Ornn", "TFT9_Item_Ornn"))


def load_item_db() -> dict:
    path = DB_DIR / "tft_item_db.json"
    if not path.exists():
        print(f"Missing item DB: {path}")
        return {}
    raw = load_json(path)
    filtered = {key: value for key, value in raw.items() if item_is_valid(key)}
    print(f"Item DB: {len(filtered)} usable entries from {path} ({len(raw)} raw)")
    return filtered


def detect_set_number(champ_db: dict) -> Optional[int]:
    meta_path = DB_DIR / "tft_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            if isinstance(meta.get("set_number"), int):
                return meta["set_number"]
        except Exception:
            pass

    counter: Counter[int] = Counter()
    for api_name in champ_db:
        match = re.match(r"^TFT(\d+)_", api_name)
        if match:
            counter[int(match.group(1))] += 1
    return counter.most_common(1)[0][0] if counter else None


def champion_is_playable(api_name: str, entry: dict) -> bool:
    if NON_PLAYABLE_RE.search(api_name):
        return False
    if NON_PLAYABLE_RE.search(str(entry.get("name_en", ""))):
        return False
    return True


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def asset_token(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", text.lower().replace("'", "").replace(" ", ""))


def seasonal_suffixes(set_number: Optional[int]) -> List[str]:
    if not set_number:
        return []
    suffixes = [f"tft_set{set_number}"]
    if set_number > 1:
        suffixes.append(f"tft_set{set_number - 1}")
    return unique(suffixes)


def ddragon_champ_candidates(api_name: str, short_id: str, set_number: Optional[int]) -> List[str]:
    variants = [short_id]
    fixed = CASE_FIXES.get(asset_token(short_id))
    if fixed:
        variants = fixed + variants

    candidates = [api_name]
    if set_number:
        candidates.extend(f"TFT{set_number}_{name}" for name in variants)
    candidates.extend(f"TFT_{name}" for name in variants)
    candidates.extend(variants)
    return unique(candidates)


def cdragon_champ_urls(api_name: str, short_id: str, set_number: Optional[int]) -> List[str]:
    base_token = asset_token(short_id)
    api_token = asset_token(api_name)
    dir_tokens = [base_token]
    if set_number:
        dir_tokens.insert(0, f"tft{set_number}_{base_token}")
    if api_token and "_" in api_name:
        dir_tokens.insert(0, api_name.lower())
    dir_tokens = unique(dir_tokens)

    urls: List[str] = []
    for directory in dir_tokens:
        stem = directory
        for suffix in seasonal_suffixes(set_number):
            urls.append(f"{CDRAGON_BASE}/characters/{directory}/hud/{stem}_square.{suffix}.png")
    if ALLOW_BASE_SKIN:
        for directory in dir_tokens:
            urls.append(f"{CDRAGON_BASE}/characters/{directory}/hud/{directory}_square.png")
    return unique(urls)


def is_base_skin_url(url: str) -> bool:
    return url.endswith("_square.png") and ".tft_set" not in url


def get_ddragon_version() -> str:
    try:
        response = http_get(f"{DDRGON_BASE}/api/versions.json", timeout=10, label="versions")
        return response.json()[0]
    except Exception:
        return VERSION


def _first_display_name(api_name: str, entry: Optional[dict], fallback: str) -> str:
    if isinstance(entry, dict):
        for key in DISPLAY_NAME_KEYS:
            value = entry.get(key)
            if isinstance(value, str):
                value = SPACE_RE.sub(" ", value).strip()
                if value and value != api_name:
                    return value
    return fallback


def safe_filename_part(text: str, fallback: str) -> str:
    value = SPACE_RE.sub(" ", str(text or fallback)).strip()
    value = INVALID_FILENAME_RE.sub("_", value)
    value = value.rstrip(" .")
    value = re.sub(r"_+", "_", value)
    return value or fallback


def champion_asset_stem(api_name: str, entry: dict) -> str:
    short_id = str(entry.get("short_id") or api_name)
    display_name = safe_filename_part(_first_display_name(api_name, entry, short_id), short_id)
    prefix = api_name
    if short_id and api_name.endswith(short_id):
        prefix = api_name[: -len(short_id)].rstrip("_")
    elif "_" in api_name:
        prefix = api_name.rsplit("_", 1)[0]
    return f"{prefix}_{display_name}" if prefix else display_name


def item_asset_stem(api_name: str, entry: dict) -> str:
    fallback = api_name.rsplit("_", 1)[-1] if "_" in api_name else api_name
    display_name = safe_filename_part(_first_display_name(api_name, entry, fallback), fallback)
    prefix = api_name.rsplit("_", 1)[0] if "_" in api_name else ""
    return f"{prefix}_{display_name}" if prefix else display_name


def build_asset_record(api_name: str, stem: str, entry: dict) -> dict:
    record = {
        "id": api_name,
        "filename": f"{stem}.png",
        "name": _first_display_name(api_name, entry, api_name),
    }
    if isinstance(entry, dict) and entry.get("short_id"):
        record["short_id"] = entry["short_id"]
    return record


def maybe_copy_legacy_asset(dest: Path, legacy_path: Path, min_size: int) -> bool:
    if FORCE:
        return False
    if dest.exists() and dest.stat().st_size > min_size:
        return True
    if legacy_path != dest and legacy_path.exists() and legacy_path.stat().st_size > min_size:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(legacy_path.read_bytes())
        return True
    return False


def fetch_champions(version: str, failed_log: list) -> Dict[str, dict]:
    champ_db = load_champion_db()
    if not champ_db:
        return {}

    set_number = detect_set_number(champ_db)
    playable = {k: v for k, v in champ_db.items() if champion_is_playable(k, v)}
    skipped = len(champ_db) - len(playable)
    out_dir = ASSETS_DIR / "champions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cdn = f"{DDRGON_BASE}/cdn/{version}"

    ok = fail = cache_skip = 0
    asset_map: Dict[str, dict] = {}
    print(f"\nChampions: {len(playable)} playable entries -> {out_dir}")
    if set_number:
        print(f"Detected set: {set_number}")
    if skipped:
        print(f"Skipped non-playable entries: {skipped}")

    for api_name, entry in playable.items():
        stem = champion_asset_stem(api_name, entry)
        dest = out_dir / f"{stem}.png"
        legacy_path = out_dir / f"{api_name}.png"
        asset_map[stem] = build_asset_record(api_name, stem, entry)

        if maybe_copy_legacy_asset(dest, legacy_path, 500):
            cache_skip += 1
            continue

        short_id = str(entry.get("short_id") or api_name)
        data = None
        used_url = ""

        for candidate in ddragon_champ_candidates(api_name, short_id, set_number):
            data, used_url = dl_with_url(f"{cdn}/img/tft-champion/{candidate}.png")
            if data:
                break

        if not data:
            for url in cdragon_champ_urls(api_name, short_id, set_number):
                data, used_url = dl_with_url(url)
                if data:
                    break

        if data and not is_base_skin_url(used_url):
            dest.write_bytes(resize_image(data, ICON_SIZE))
            print(f"  OK  {api_name} -> {dest.name}")
            ok += 1
            continue

        if data and is_base_skin_url(used_url):
            print(f"  BAD {api_name} -> base skin fallback blocked")
        else:
            print(f"  MISS {api_name} -> no seasonal portrait found")
        fail += 1
        failed_log.append(("champion", api_name, str(dest)))

    print(f"Result: ok={ok} failed={fail} skipped={cache_skip}")
    return asset_map


def fetch_items(version: str, failed_log: list) -> Dict[str, dict]:
    item_db = load_item_db()
    if not item_db:
        return {}

    out_dir = ASSETS_DIR / "items"
    out_dir.mkdir(parents=True, exist_ok=True)
    cdn = f"{DDRGON_BASE}/cdn/{version}"
    ok = fail = cache_skip = 0
    asset_map: Dict[str, dict] = {}

    print(f"\nItems: {len(item_db)} entries -> {out_dir}")
    for api_name, entry in item_db.items():
        stem = item_asset_stem(api_name, entry)
        dest = out_dir / f"{stem}.png"
        legacy_path = out_dir / f"{api_name}.png"
        asset_map[stem] = build_asset_record(api_name, stem, entry)

        if maybe_copy_legacy_asset(dest, legacy_path, 200):
            cache_skip += 1
            continue

        data = dl(f"{cdn}/img/tft-item/{api_name}.png")
        if not data:
            lc = api_name.lower()
            for url in (
                f"{CDRAGON_BASE}/items/icons2d/{lc}.png",
                f"{CDRAGON_BASE}/tft/tftitems/icons2d/{lc}.png",
            ):
                data = dl(url)
                if data:
                    break

        if data:
            dest.write_bytes(resize_image(data, ITEM_SIZE))
            print(f"  OK  {api_name} -> {dest.name}")
            ok += 1
        else:
            print(f"  MISS {api_name}")
            fail += 1
            failed_log.append(("item", api_name, str(dest)))

    print(f"Result: ok={ok} failed={fail} skipped={cache_skip}")
    return asset_map


def verify() -> None:
    print("\nAsset verification")
    index = load_asset_index()
    index_path = asset_index_path()
    if index_path.exists():
        print(f"  asset_index count: champions={len(index.get('champions', {}))} items={len(index.get('items', {}))}")
    for subdir in ("champions", "items"):
        directory = ASSETS_DIR / subdir
        if not directory.exists():
            print(f"  MISSING {directory}")
            continue
        files = list(directory.glob("*.png"))
        total_kb = sum(path.stat().st_size for path in files) / 1024
        tiny = [path.name for path in files if path.stat().st_size < 300]
        status = "possible corrupt files: " + ", ".join(tiny[:3]) if tiny else "looks good"
        print(f"  {subdir:<10} count={len(files):<4} size={total_kb:8.1f}KB  {status}")


def list_missing() -> None:
    found_any = False
    champ_db = load_champion_db()
    for api_name, entry in champ_db.items():
        if not champion_is_playable(api_name, entry):
            continue
        path = ASSETS_DIR / "champions" / f"{champion_asset_stem(api_name, entry)}.png"
        if not path.exists() or path.stat().st_size < 500:
            print(f"MISSING champion: {path.name}")
            found_any = True

    item_db = load_item_db()
    for api_name, entry in item_db.items():
        path = ASSETS_DIR / "items" / f"{item_asset_stem(api_name, entry)}.png"
        if not path.exists() or path.stat().st_size < 200:
            print(f"MISSING item: {path.name}")
            found_any = True

    if not found_any:
        print("All expected asset files are present.")


def main() -> int:
    global ASSETS_DIR, DB_DIR, FORCE, ALLOW_BASE_SKIN, NETWORK_MODE

    parser = argparse.ArgumentParser(description="Fetch TFT champion and item assets")
    parser.add_argument("--champs", action="store_true", help="download champion portraits only")
    parser.add_argument("--items", action="store_true", help="download item icons only")
    parser.add_argument("--verify", action="store_true", help="verify downloaded assets")
    parser.add_argument("--list-missing", action="store_true", help="list missing assets")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--allow-base-skin", action="store_true", help="allow plain _square.png fallback for champions")
    parser.add_argument("--version", default=None, help="pin a specific DDragon version")
    parser.add_argument("--db-dir", default=".", help="directory containing TFT JSON DB files")
    parser.add_argument("--assets-dir", default=None, help="output directory for downloaded assets")
    parser.add_argument("--network", choices=["auto", "env", "direct"], default="auto", help="network mode")
    args = parser.parse_args()

    DB_DIR = Path(args.db_dir)
    if args.assets_dir:
        ASSETS_DIR = Path(args.assets_dir)
    FORCE = args.force
    ALLOW_BASE_SKIN = args.allow_base_skin
    NETWORK_MODE = args.network

    if args.verify:
        verify()
        return 0
    if args.list_missing:
        list_missing()
        return 0

    if not (DB_DIR / "tft_champion_db.json").exists() and not (DB_DIR / "tft_item_db.json").exists():
        print("Missing DB files. Run: python tft_data_manager.py --set 17")
        return 1

    version = args.version or get_ddragon_version()
    print(f"DDragon version: {version}")
    print(f"DB dir:   {DB_DIR.resolve()}")
    print(f"Out dir:  {ASSETS_DIR.resolve()}")
    print(f"Network:  {NETWORK_MODE}")
    print(f"Base skin fallback: {'enabled' if ALLOW_BASE_SKIN else 'disabled'}")

    failed: List[Tuple[str, str, str]] = []
    index = load_asset_index()
    do_all = not args.champs and not args.items
    if do_all or args.champs:
        index["champions"] = fetch_champions(version, failed)
    if do_all or args.items:
        index["items"] = fetch_items(version, failed)
    save_asset_index(index)

    print("\n" + "=" * 52)
    print(f"Done. Output: {ASSETS_DIR.resolve()}")
    print(f"Asset index: {asset_index_path().resolve()}")
    if failed:
        print(f"Missing or blocked assets: {len(failed)}")
        for asset_type, api_name, _ in failed[:20]:
            print(f"  [{asset_type}] {api_name}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        print("Run: python tft_fetch_assets.py --list-missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
