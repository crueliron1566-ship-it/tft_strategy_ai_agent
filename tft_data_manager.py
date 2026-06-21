#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFT data manager.

Downloads TFT data from CommunityDragon or DDragon and builds local JSON files.
Internal ids stay in English for matching logic. Download locale defaults to Chinese.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 TFT-DataManager/4.0"}
DELAY = 0.4
OUTPUT_DIR = Path(".")
DEFAULT_CDRAGON_LOCALE = "zh_cn"
DEFAULT_DDRAGON_LOCALE = "zh_CN"
CDRAGON_PATCH_CANDIDATES = ["latest", "pbe", "16.12", "16.10", "16.8", "16.6", "16.5", "16.3", "16.1"]
FALLBACK_DDRAGON_VERSION = "15.12.1"
TIMEOUT = 15


def _request_json(url: str, timeout: int = TIMEOUT) -> Any:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    # CommunityDragon / DDragon occasionally omit or misreport charset headers.
    # Decode JSON bytes as UTF-8 directly so Chinese names are preserved.
    try:
        return json.loads(response.content.decode("utf-8"))
    except UnicodeDecodeError:
        response.encoding = "utf-8"
        return response.json()


def get_ddragon_version() -> str:
    try:
        versions = _request_json("https://ddragon.leagueoflegends.com/api/versions.json")
        if isinstance(versions, list) and versions:
            version = str(versions[0])
            print(f"  DDragon latest version: {version}")
            return version
    except Exception as exc:
        print(f"  DDragon version lookup failed: {exc}")
    return FALLBACK_DDRAGON_VERSION


def _strip(value: str, set_number: int) -> str:
    value = re.sub(rf"^TFT{set_number}_", "", value)
    value = re.sub(rf"^TFTSet{set_number}_", "", value)
    value = re.sub(rf"^Set{set_number}_", "", value)
    value = re.sub(r"^TFT_", "", value)
    return value


def _is_target_set_id(value: str, set_number: int) -> bool:
    if not value:
        return False
    return bool(
        re.match(rf"^TFT{set_number}_", value)
        or re.match(rf"^TFTSet{set_number}_", value)
        or re.match(rf"^Set{set_number}_", value)
    )


def _is_generic_item_id(value: str) -> bool:
    if not value:
        return False
    return value.startswith("TFT_Item_") or value.startswith("Item_")


def _item_in_scope(value: str, set_number: int) -> bool:
    return _is_generic_item_id(value) or _is_target_set_id(value, set_number)


def _is_playable_champion(api: str, cost: int, set_number: int) -> bool:
    if not _is_target_set_id(api, set_number):
        return False
    if cost < 1 or cost > 5:
        return False
    blocked_tokens = ("PVE_", "Summon", "Minion", "Core")
    return not any(token in api for token in blocked_tokens)

def _foreign_champion_ids(champ_db: Dict[str, Any], set_number: int) -> list[str]:
    foreign: list[str] = []
    for champ_id in champ_db:
        if not _is_target_set_id(champ_id, set_number):
            foreign.append(champ_id)
    return sorted(foreign)


def _normalize_cdragon_payload(payload: Any, set_number: int, patch: str, locale: str) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    top_champions = payload.get("champions")
    top_traits = payload.get("traits")
    if isinstance(top_champions, list) and isinstance(top_traits, list):
        return {
            "set_number": set_number,
            "patch": patch,
            "locale": locale,
            "champions": top_champions,
            "traits": top_traits,
            "items": payload.get("items", []),
            "item_ids": [],
        }

    for key in ("sets", "setData", "data"):
        blocks = payload.get(key)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_set = block.get("setNumber") or block.get("number") or block.get("set")
            if block_set == set_number and any(k in block for k in ("champions", "traits", "items")):
                return {
                    "set_number": set_number,
                    "patch": patch,
                    "locale": locale,
                    "champions": block.get("champions", []),
                    "traits": block.get("traits", []),
                    "items": payload.get("items", []),
                    "item_ids": [item for item in block.get("items", []) if isinstance(item, str)],
                }

    return None


def fetch_cdragon(set_number: int, locale: str = DEFAULT_CDRAGON_LOCALE) -> Optional[Dict[str, Any]]:
    for patch in CDRAGON_PATCH_CANDIDATES:
        url = f"https://raw.communitydragon.org/{patch}/cdragon/tft/{locale}.json"
        try:
            print(f"  CDragon {patch}...")
            payload = _request_json(url)
            raw = _normalize_cdragon_payload(payload, set_number, patch, locale)
            if raw and raw.get("champions"):
                return raw
            print("    skipped: payload shape does not match TFT data")
        except Exception as exc:
            print(f"    error: {exc}")
        time.sleep(DELAY)
    return None


def fetch_ddragon(
    set_number: int,
    version: str,
    locale: str = DEFAULT_DDRAGON_LOCALE,
) -> Optional[Dict[str, Any]]:
    base = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}"
    endpoints = {
        "champions": "tft-champion.json",
        "traits": "tft-trait.json",
        "items": "tft-item.json",
    }

    payloads: Dict[str, Any] = {}
    for key, filename in endpoints.items():
        url = f"{base}/{filename}"
        try:
            print(f"  DDragon {filename}...")
            payloads[key] = _request_json(url)
        except Exception as exc:
            print(f"  DDragon {filename} failed: {exc}")
            return None
        time.sleep(DELAY)

    return {
        "set_number": set_number,
        "version": version,
        "locale": locale,
        "champions": payloads["champions"].get("data", payloads["champions"]),
        "traits": payloads["traits"].get("data", payloads["traits"]),
        "items": payloads["items"].get("data", payloads["items"]),
    }


def _finalize_trait_db(
    all_traits: Dict[str, Any],
    referenced_short_ids: Iterable[str],
) -> Dict[str, Any]:
    referenced = {trait_id for trait_id in referenced_short_ids if trait_id}
    return {
        api: trait
        for api, trait in all_traits.items()
        if trait.get("short_id") in referenced
    }


def _has_trait_mapping(champ_db: Dict[str, Any], cmap: Dict[str, Any], trait_db: Dict[str, Any]) -> bool:
    if trait_db:
        return True
    if any(champ.get("traits") for champ in champ_db.values() if isinstance(champ, dict)):
        return True
    if any(traits for traits in cmap.values() if isinstance(traits, list)):
        return True
    return False


def parse_cdragon(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    set_number = raw["set_number"]
    all_traits: Dict[str, Any] = {}
    champ_db: Dict[str, Any] = {}
    item_db: Dict[str, Any] = {}
    cmap: Dict[str, Any] = {}
    name_to_short: Dict[str, str] = {}
    referenced_short_ids: set[str] = set()

    for trait in raw.get("traits", []):
        if not isinstance(trait, dict):
            continue
        api = trait.get("apiName") or trait.get("id") or ""
        if not api:
            continue
        short_id = _strip(api, set_number)
        name = trait.get("name") or short_id
        effects = trait.get("effects") or trait.get("sets") or []
        levels = sorted(
            {
                int(effect.get("minUnits") or effect.get("min") or 0)
                for effect in effects
                if isinstance(effect, dict) and int(effect.get("minUnits") or effect.get("min") or 0) > 0
            }
        )
        all_traits[api] = {
            "id": api,
            "short_id": short_id,
            "name_en": name,
            "levels": levels,
            "effects": effects,
        }
        name_to_short[short_id] = short_id
        name_to_short[name] = short_id

    for champ in raw.get("champions", []):
        if not isinstance(champ, dict):
            continue
        api = champ.get("apiName") or champ.get("id") or ""
        if not _is_target_set_id(api, set_number):
            continue

        short_id = _strip(api, set_number)
        traits: list[str] = []
        for trait in champ.get("traits", []):
            if not isinstance(trait, str):
                continue
            stripped = _strip(trait, set_number)
            normalized = name_to_short.get(stripped, name_to_short.get(trait, stripped))
            if normalized:
                traits.append(normalized)
                referenced_short_ids.add(normalized)

        cost = int(champ.get("cost") or champ.get("tier") or 0)
        if not _is_playable_champion(api, cost, set_number):
            continue

        champ_db[api] = {
            "id": api,
            "short_id": short_id,
            "name_en": champ.get("name") or short_id,
            "cost": cost,
            "traits": traits,
            "stats": champ.get("stats", {}),
        }
        cmap[api] = traits

    item_scope_ids = set(raw.get("item_ids") or [])
    for item in raw.get("items", []):
        if not isinstance(item, dict):
            continue
        api = item.get("apiName") or item.get("id") or ""
        if item_scope_ids:
            if api not in item_scope_ids:
                continue
        elif not _item_in_scope(api, set_number):
            continue
        item_db[api] = {
            "id": api,
            "name_en": item.get("name") or api,
            "desc": item.get("desc") or item.get("description") or "",
            "unique": bool(item.get("unique", False)),
            "composition": item.get("composition") or item.get("from") or [],
        }

    trait_db = _finalize_trait_db(all_traits, referenced_short_ids)
    return champ_db, trait_db, item_db, cmap


def parse_ddragon(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    set_number = raw["set_number"]
    all_traits: Dict[str, Any] = {}
    champ_db: Dict[str, Any] = {}
    item_db: Dict[str, Any] = {}
    cmap: Dict[str, Any] = {}
    name_to_short: Dict[str, str] = {}
    referenced_short_ids: set[str] = set()

    for key, trait in raw.get("traits", {}).items():
        if not isinstance(trait, dict):
            continue
        api = trait.get("id") or key
        short_id = _strip(api, set_number)
        name = trait.get("name") or short_id
        effects = trait.get("sets") or trait.get("effects") or []
        levels = sorted(
            {
                int(effect.get("min") or effect.get("minUnits") or 0)
                for effect in effects
                if isinstance(effect, dict) and int(effect.get("min") or effect.get("minUnits") or 0) > 0
            }
        )
        all_traits[api] = {
            "id": api,
            "short_id": short_id,
            "name_en": name,
            "levels": levels,
            "effects": effects,
        }
        name_to_short[short_id] = short_id
        name_to_short[name] = short_id

    for key, champ in raw.get("champions", {}).items():
        if not isinstance(champ, dict):
            continue
        api = champ.get("id") or key
        if not _is_target_set_id(api, set_number):
            continue

        short_id = _strip(api, set_number)
        traits: list[str] = []
        for trait in champ.get("traits", []):
            if not isinstance(trait, str):
                continue
            stripped = _strip(trait, set_number)
            normalized = name_to_short.get(stripped, name_to_short.get(trait, stripped))
            if normalized:
                traits.append(normalized)
                referenced_short_ids.add(normalized)

        cost = int(champ.get("tier") or champ.get("cost") or 0)
        if not _is_playable_champion(api, cost, set_number):
            continue

        champ_db[api] = {
            "id": api,
            "short_id": short_id,
            "name_en": champ.get("name") or short_id,
            "cost": cost,
            "traits": traits,
            "stats": champ.get("stats", {}),
        }
        cmap[api] = traits

    for key, item in raw.get("items", {}).items():
        if not isinstance(item, dict):
            continue
        api = item.get("id") or key
        if not _item_in_scope(api, set_number):
            continue
        item_db[api] = {
            "id": api,
            "name_en": item.get("name") or api,
            "desc": item.get("desc") or item.get("description") or "",
            "unique": bool(item.get("unique", False)),
            "composition": item.get("from") or item.get("composition") or [],
        }

    trait_db = _finalize_trait_db(all_traits, referenced_short_ids)
    return champ_db, trait_db, item_db, cmap


def save_all(
    champ_db: Dict[str, Any],
    trait_db: Dict[str, Any],
    item_db: Dict[str, Any],
    cmap: Dict[str, Any],
    set_number: int,
    source_info: Dict[str, Any],
) -> None:
    def write(name: str, data: Any) -> None:
        path = OUTPUT_DIR / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        size = len(data) if hasattr(data, "__len__") else 0
        print(f"  saved {name} ({size})")

    print("\nSaving files...")
    write("tft_champion_db.json", champ_db)
    write("tft_trait_db.json", trait_db)
    write("tft_item_db.json", item_db)
    write("tft_champion_trait_map.json", cmap)

    legacy: Dict[str, Any] = {}
    for api, trait in trait_db.items():
        short_id = trait["short_id"]
        name_en = trait["name_en"]
        members = [champ_id for champ_id, traits in cmap.items() if short_id in traits]
        entry = {
            "api_name": api,
            "short_id": short_id,
            "name_en": name_en,
            "champions": members,
            "activation": {"levels": trait.get("levels", [])},
        }
        legacy[short_id] = entry
        if name_en != short_id:
            legacy[name_en] = entry
    write("tft_trait_champion_dict.json", legacy)

    meta = {
        "set_number": set_number,
        **source_info,
        "champion_count": len(champ_db),
        "trait_count": len(trait_db),
        "item_count": len(item_db),
        "fetched_at": datetime.now().isoformat(),
    }
    write("tft_meta.json", meta)
    print(f"\nOutput dir: {OUTPUT_DIR.resolve()}")


def verify() -> bool:
    print("\nVerifying data files...")
    files = {
        "tft_champion_db.json": "champions",
        "tft_trait_db.json": "traits",
        "tft_item_db.json": "items",
        "tft_champion_trait_map.json": "champion-trait map",
        "tft_trait_champion_dict.json": "trait dictionary",
        "tft_meta.json": "meta",
    }

    all_ok = True
    loaded: Dict[str, Any] = {}
    for filename, label in files.items():
        path = OUTPUT_DIR / filename
        if not path.exists():
            print(f"  missing: {label} -> {filename}")
            all_ok = False
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded[filename] = data
            size = len(data) if hasattr(data, "__len__") else 0
            print(f"  ok: {label:<18} ({size})")
        except Exception as exc:
            print(f"  broken: {label} -> {exc}")
            all_ok = False

    meta = loaded.get("tft_meta.json")
    champ_db = loaded.get("tft_champion_db.json")
    trait_db = loaded.get("tft_trait_db.json")
    cmap = loaded.get("tft_champion_trait_map.json")
    if isinstance(meta, dict) and isinstance(champ_db, dict):
        set_number = int(meta.get("set_number", 0) or 0)
        if set_number > 0:
            foreign = _foreign_champion_ids(champ_db, set_number)
            if foreign:
                all_ok = False
                preview = ", ".join(foreign[:8])
                print(f"  mixed-set warning: found {len(foreign)} off-set champions, e.g. {preview}")

        if not _has_trait_mapping(
            champ_db,
            cmap if isinstance(cmap, dict) else {},
            trait_db if isinstance(trait_db, dict) else {},
        ):
            all_ok = False
            print("  trait mapping missing: champion/trait relationship data is empty")

    if all_ok:
        print("\nVerification passed.")
    else:
        print("\nVerification failed. Rebuild with: python tft_data_manager.py --set <set>")
    return all_ok


def fetch_and_build(
    set_number: int = 16,
    cdragon_locale: str = DEFAULT_CDRAGON_LOCALE,
    ddragon_locale: str = DEFAULT_DDRAGON_LOCALE,
) -> bool:
    print(f"\n{'=' * 52}")
    print(f"  TFT Data Manager - Set {set_number}")
    print(f"{'=' * 52}")

    print("\n[1/2] Fetching from CommunityDragon...")
    raw = fetch_cdragon(set_number, locale=cdragon_locale)

    if raw:
        champ_db, trait_db, item_db, cmap = parse_cdragon(raw)
        source_info = {
            "source": "cdragon",
            "patch": raw["patch"],
            "locale": raw.get("locale", cdragon_locale),
        }
    else:
        print("\n[2/2] CDragon failed, switching to DDragon...")
        version = get_ddragon_version()
        raw = fetch_ddragon(set_number, version, locale=ddragon_locale)
        if not raw:
            print("All data sources failed.")
            return False
        champ_db, trait_db, item_db, cmap = parse_ddragon(raw)
        source_info = {
            "source": "ddragon",
            "version": version,
            "locale": raw.get("locale", ddragon_locale),
        }

    if not champ_db:
        print("No in-set champions were found. Nothing was written.")
        return False

    if not _has_trait_mapping(champ_db, cmap, trait_db):
        source_name = source_info.get("source", "unknown")
        print(
            "Trait extraction failed: the selected data source did not provide usable champion-to-trait mapping. "
            f"Current source: {source_name}."
        )
        if source_name == "ddragon":
            print("DDragon no longer exposes full TFT trait membership data. Please use CommunityDragon instead.")
        return False

    foreign = _foreign_champion_ids(champ_db, set_number)
    if foreign:
        print(f"Refusing to write mixed-set data; found off-set ids like: {', '.join(foreign[:5])}")
        return False

    print(
        f"\nParsed: {len(champ_db)} champions / {len(trait_db)} traits / {len(item_db)} items"
    )
    cost_dist = Counter(int(champ.get("cost", 0)) for champ in champ_db.values())
    print(f"Cost distribution: {dict(sorted(cost_dist.items()))}")

    save_all(champ_db, trait_db, item_db, cmap, set_number, source_info)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local TFT data files")
    parser.add_argument("--set", type=int, default=16, help="target set number")
    parser.add_argument("--output-dir", type=str, default=".", help="output directory")
    parser.add_argument("--verify", action="store_true", help="verify existing JSON files")
    parser.add_argument(
        "--cdragon-locale",
        type=str,
        default=DEFAULT_CDRAGON_LOCALE,
        help="CommunityDragon locale (default: zh_cn)",
    )
    parser.add_argument(
        "--ddragon-locale",
        type=str,
        default=DEFAULT_DDRAGON_LOCALE,
        help="DDragon locale (default: zh_CN)",
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        sys.exit(0 if verify() else 1)

    sys.exit(0 if fetch_and_build(args.set, args.cdragon_locale, args.ddragon_locale) else 1)



