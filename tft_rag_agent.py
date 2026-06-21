#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_rag_agent.py
TFT 闃靛椤鹃棶 鈥?RAG Agent 鏍稿績妯″潡

鏋舵瀯:
  - LLMClient        : LLM 璋冪敤锛圓nthropic / OpenRouter锛?
  - TFTCrawler       : Riot API 楂樼灞€鏁版嵁閲囬泦锛堜粎浣跨敤 tft-* 绯诲垪 API锛?
  - JSONKnowledgeBase: BM25 鐭ヨ瘑搴擄紙鏃犲閮ㄤ緷璧栵級
  - LocalDataLoader  : 鏈湴闃靛鏁版嵁鍔犺浇
  - 涓変釜瀛?Agent     : EconomyAgent / PowerAgent / PositionAgent
  - TFTRagAgent      : 涓诲崗璋冨櫒锛屾暣鍚堟墍鏈夊瓙 Agent 杈撳嚭

鐢ㄦ硶:
  python tft_rag_agent.py                       # 浜や簰妯″紡
  python tft_rag_agent.py --question "濡備綍杩囨浮"  # 鍗曟鎻愰棶
"""

import json, os, re, time, math, hashlib, logging, sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from functools import lru_cache
from openai import OpenAI

import requests


# 鏃ュ織

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TFT-RAG")


# 鍏ㄥ眬閰嶇疆

CFG: Dict[str, Any] = {
    # 璧涘
    "current_set"           : 17,
    # 鏂囦欢璺緞
    "data_dir"              : "./tft_rag_data",
    "analysis_file"         : "tft_team_analysis.json",
    "champion_db_file"      : "tft_champion_db.json",
    "trait_db_file"         : "tft_trait_db.json",
    "item_db_file"          : "tft_item_db.json",
    "trait_dict_file"       : "tft_trait_champion_dict.json",
    # LLM
    "llm_provider"          : os.getenv("LLM_PROVIDER", "sophnet"),
    "sophnet_api_key"       : os.getenv("SOPHNET_API_KEY", "MiSqrLARooni-MPndBAJGM-BHmeI9ocg4wkZLFCHf6Q3PeBC6iGLJgL4FP2Xr25zhp9i8TTuz0k_W_Xb1c1VpA"),
    "sophnet_base_url"      : "https://www.sophnet.com/api/open-apis/v1",
    "sophnet_model"         : os.getenv("SOPHNET_MODEL", "DeepSeek-V4-Flash"),
    "max_tokens"            : 2048,
    # Riot API锛堜粎 TFT 绯诲垪鎺ュ彛锛?
    "riot_api_key"          : os.getenv("RIOT_API_KEY", "RGAPI-83325a18-9840-4e1e-86dc-7f7ed9ed7bd1"),
    "riot_region_platform"  : "kr",
    "riot_region_regional"  : "asia",
    "riot_tiers"            : ("challenger", "grandmaster"),
    "riot_max_players"      : 10,
    "riot_matches_per_player": 10,
    "cache_ttl_hours"       : 12,
    # RAG
    "top_k"                 : 6,
    "chunk_size"            : 600,
    "chunk_overlap"         : 80,
    "background_crawl"      : True,
}
DATA_DIR = Path(CFG["data_dir"])
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TFT-Advisor/2.0",
}


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
    return int(CFG.get("current_set", default) or default)

CFG["current_set"] = _effective_set_number(CFG.get("current_set", 17))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clean_display_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return ""
    return text


def _load_json_if_possible(path_value: Any) -> Any:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _strip_trait_label_suffix(name: str) -> str:
    text = _clean_display_name(name)
    for suffix in ("纹章", "Emblem", " Emblem", "模式", " Mode"):
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()
    return text


_INTERNAL_TRAIT_TOKENS = {
    "ADMIN", "APTrait", "ASTrait", "AnimaSquad", "AssassinTrait", "Astronaut",
    "DRX", "DarkStar", "Fateweaver", "FlexTrait", "HPTank", "ManaTrait",
    "Mecha", "MeleeTrait", "Primordian", "PsyOps", "RangedTrait",
    "ResistTank", "ShieldTank", "SpaceGroove", "SummonTrait", "Timebreaker",
}


@lru_cache(maxsize=1)
def _trait_alias_map() -> Dict[str, str]:
    alias: Dict[str, str] = {}
    set_num = _effective_set_number()

    def _register(display_name: Any, *keys: Any):
        display = _strip_trait_label_suffix(display_name)
        if not display or _looks_internal_trait_name(display):
            return
        for raw_key in keys:
            key = _clean_display_name(raw_key)
            if not key:
                continue
            clean_key = re.sub(r"^TFT(?:Set)?\d+_", "", key)
            alias.setdefault(key, display)
            alias.setdefault(clean_key, display)

    trait_dict = _load_json_if_possible(CFG.get("trait_dict_file"))
    if isinstance(trait_dict, dict):
        for trait_name, entry in trait_dict.items():
            if not isinstance(entry, dict):
                continue
            _register(
                entry.get("name_zh") or entry.get("name_cn") or entry.get("name_en") or entry.get("name") or trait_name,
                trait_name,
                entry.get("short_id"),
                entry.get("api_name"),
                entry.get("apiName"),
                entry.get("id"),
            )

    trait_db = _load_json_if_possible(CFG.get("trait_db_file"))
    if isinstance(trait_db, dict):
        for trait_id, entry in trait_db.items():
            if not isinstance(entry, dict):
                continue
            _register(
                entry.get("name_zh") or entry.get("name_cn") or entry.get("name_en") or entry.get("name") or trait_id,
                trait_id,
                entry.get("short_id"),
                entry.get("api_name"),
                entry.get("apiName"),
                entry.get("id"),
            )

    item_db = _load_json_if_possible(CFG.get("item_db_file"))
    pattern = re.compile(rf"^TFT{set_num}_Item_([A-Za-z0-9_]+)EmblemItem$")
    if isinstance(item_db, dict):
        for item_id, item in item_db.items():
            if not isinstance(item, dict):
                continue
            match = pattern.match(item_id)
            if not match:
                continue
            _register(
                item.get("name_zh") or item.get("name_cn") or item.get("name_en") or item.get("name"),
                match.group(1),
            )

    if "Stargazer" in alias:
        for variant in (
            "Stargazer_Fountain",
            "Stargazer_Huntress",
            "Stargazer_Medallion",
            "Stargazer_Mountain",
            "Stargazer_Serpent",
            "Stargazer_Shield",
            "Stargazer_Wolf",
        ):
            alias.setdefault(variant, alias["Stargazer"])
    return alias
@lru_cache(maxsize=1)
def _unit_alias_map() -> Dict[str, str]:
    alias: Dict[str, str] = {
        "IvernMinion": "小木灵",
        "Summon": "召唤物",
        "DarkStar_FakeUnit": "迷你黑洞",
        "MissFortune_TraitClone": "厄运小姐",
    }

    sources = [
        CFG.get("analysis_file"),
        "tft_team_analysis.json",
        "t1_duel_debug.json",
        "t2_global_debug.json",
        CFG.get("champion_db_file"),
    ]

    def walk(obj: Any):
        if isinstance(obj, dict):
            display = _clean_display_name(
                obj.get("name_zh")
                or obj.get("name_cn")
                or obj.get("name_en")
                or obj.get("name")
            )
            short_id = _clean_display_name(obj.get("short_id"))
            unit_id = _clean_display_name(obj.get("id"))
            if display and not display.lower().startswith("unknown_"):
                for key in (short_id, unit_id):
                    if not key:
                        continue
                    clean_key = re.sub(r"^TFT(?:Set)?\d+_", "", key)
                    alias.setdefault(key, display)
                    alias.setdefault(clean_key, display)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    for source in sources:
        data = _load_json_if_possible(source)
        if data is not None:
            walk(data)
    return alias


def _looks_internal_trait_name(name: Any) -> bool:
    raw = _clean_display_name(name)
    if not raw:
        return False
    return (
        raw in _INTERNAL_TRAIT_TOKENS
        or raw.startswith("Stargazer_")
        or raw.endswith("Trait")
        or raw.endswith("Tank")
        or raw.endswith("UniqueTrait")
    )


def _canonical_trait_name(name: Any) -> str:
    raw = _clean_display_name(name)
    if not raw:
        return ""
    clean = re.sub(r"^TFT(?:Set)?\d+_", "", raw)
    alias = _trait_alias_map()
    if clean in alias:
        return alias[clean]
    if raw in alias:
        return alias[raw]
    if clean.startswith("Stargazer_"):
        return alias.get("Stargazer", "观星者")
    if _looks_internal_trait_name(clean):
        return alias.get(clean, clean)
    return clean


def _canonical_unit_name(name: Any) -> str:
    raw = _clean_display_name(name)
    if not raw:
        return ""
    alias = _unit_alias_map()
    if raw in alias:
        return alias[raw]
    clean = re.sub(r"^TFT(?:Set)?\d+_", "", raw)
    clean = clean.replace("_TraitClone", "")
    if clean.startswith("Enemy_"):
        clean = clean.split("_", 1)[1]
    if clean in alias:
        return alias[clean]
    return clean


def _is_placeholder_unit_name(name: Any) -> bool:
    text = _canonical_unit_name(name)
    raw = _clean_display_name(name)
    return (
        not text
        or text.lower().startswith("unknown_")
        or raw in {"Summon", "TFT17_Summon"}
        or text in {"召唤物"}
    )


def _unit_name(unit: Dict[str, Any]) -> str:
    return _canonical_unit_name(
        unit.get("name_zh")
        or unit.get("name_cn")
        or unit.get("name_en")
        or unit.get("name")
        or unit.get("short_id")
        or unit.get("id")
        or "unknown"
    ) or "unknown"


def _trait_name(trait: Dict[str, Any]) -> str:
    return _canonical_trait_name(
        trait.get("name_zh")
        or trait.get("name_cn")
        or trait.get("name_en")
        or trait.get("name")
        or trait.get("short_id")
        or trait.get("id")
        or ""
    )


def _doc_needs_trait_cleanup(text: Any) -> bool:
    content = str(text or "")
    return bool(re.search(r"\b(?:ADMIN|APTrait|ASTrait|Astronaut|DRX|Fateweaver|FlexTrait|HPTank|ManaTrait|MeleeTrait|RangedTrait|ResistTank|ShieldTank|SpaceGroove|SummonTrait|Timebreaker|Stargazer_[A-Za-z]+)\b", content))


def _sanitize_riot_doc_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    title = str(payload.get("title", "") or "")
    content = str(payload.get("content", "") or "")
    tags = [str(tag) for tag in payload.get("tags", []) if tag]

    title_body = title.split("]", 1)[-1].strip() if "]" in title else title.strip()
    raw_traits = [part.strip() for part in title_body.split("+") if part.strip()]
    trait_names: List[str] = []
    for raw_trait in raw_traits:
        display = _canonical_trait_name(raw_trait)
        if display and display not in trait_names:
            trait_names.append(display)

    lines = content.splitlines()
    header = lines[0] if lines else ""
    sample_match = re.search(r"样本[:：]?\s*(\d+)|Sample[:：]?\s*(\d+)", header)
    avg_match = re.search(r"(?:均名|AvgPlacement)[:：]?\s*(\d+(?:\.\d+)?)", header)
    top4_match = re.search(r"Top4[:：]?\s*(\d+(?:\.\d+)?)%", header)
    win_match = re.search(r"(?:吃鸡|Win)[:：]?\s*(\d+(?:\.\d+)?)%", header)
    sample_size = _safe_int((sample_match.group(1) or sample_match.group(2)) if sample_match else 0, 0)
    avg_placement = _safe_float(avg_match.group(1), 0.0) if avg_match else 0.0
    top4_rate = _safe_float(top4_match.group(1), 0.0) if top4_match else 0.0
    win_rate = _safe_float(win_match.group(1), 0.0) if win_match else 0.0

    core_units: List[str] = []
    if len(lines) >= 2:
        for raw_unit in lines[1].split(":", 1)[-1].split(","):
            display = _canonical_unit_name(raw_unit)
            if display and not _is_placeholder_unit_name(display) and display not in core_units:
                core_units.append(display)

    augments: List[str] = []
    if len(lines) >= 3:
        for raw_aug in lines[2].split(":", 1)[-1].split(","):
            aug = _clean_display_name(raw_aug)
            if aug and aug != "无" and aug not in augments:
                augments.append(aug)

    if trait_names:
        archetype = " + ".join(trait_names[:2])
    elif core_units:
        archetype = "核心英雄: " + " / ".join(core_units[:3])
    else:
        archetype = title_body or "当前高分阵容"

    clean_tags: List[str] = []
    for tag in tags:
        if tag.lower().startswith("set"):
            if tag not in clean_tags:
                clean_tags.append(tag)
            continue
        if _looks_internal_trait_name(tag):
            continue
        display = _canonical_trait_name(tag)
        if display:
            if display not in clean_tags:
                clean_tags.append(display)
            continue
        unit = _canonical_unit_name(tag)
        if unit and not _is_placeholder_unit_name(unit) and unit not in clean_tags:
            clean_tags.append(unit)
    for display in trait_names + core_units[:3]:
        if display and display not in clean_tags:
            clean_tags.append(display)

    clean_payload = dict(payload)
    clean_payload["title"] = f"[KR高端局] {archetype}"
    clean_payload["content"] = (
        f"阵容: {archetype} | 样本: {sample_size} | 均名: {avg_placement:.2f} | Top4: {top4_rate:.0f}% | 吃鸡: {win_rate:.0f}%\n"
        f"核心英雄: {', '.join(core_units) or '无'}\n"
        f"常见海克斯: {', '.join(augments) or '无'}"
    )
    clean_payload["tags"] = clean_tags
    return clean_payload


@dataclass
class MetaReference:
    archetype: str
    traits: List[str]
    core_units: List[str]
    sample_size: int
    avg_placement: float
    top4_rate: float
    win_rate: float
    augments: List[str] = field(default_factory=list)
    title: str = ""
    source: str = "riot_api"


def _parse_meta_reference(doc: Dict[str, Any]) -> Optional[MetaReference]:
    title = str(doc.get("title", "") or "")
    content = str(doc.get("content", "") or "")
    tags = [str(t) for t in doc.get("tags", []) if t]
    archetype = title.split("]", 1)[-1].strip() if "]" in title else title.strip()
    if not archetype:
        return None

    header = content.splitlines()[0] if content else ""
    parts = [part.strip() for part in header.split("|") if part.strip()]
    sample_size = 0
    avg_placement = 0.0
    top4_rate = 0.0
    win_rate = 0.0
    if len(parts) >= 5:
        sample_match = re.search(r"(\d+)", parts[1])
        avg_match = re.search(r"(\d+(?:\.\d+)?)", parts[2])
        top4_match = re.search(r"(\d+(?:\.\d+)?)%", parts[3])
        win_match = re.search(r"(\d+(?:\.\d+)?)%", parts[4])
        sample_size = _safe_int(sample_match.group(1), 0) if sample_match else 0
        avg_placement = _safe_float(avg_match.group(1), 0.0) if avg_match else 0.0
        top4_rate = _safe_float(top4_match.group(1), 0.0) / 100.0 if top4_match else 0.0
        win_rate = _safe_float(win_match.group(1), 0.0) / 100.0 if win_match else 0.0

    lines = content.splitlines()
    core_units: List[str] = []
    augments: List[str] = []
    if len(lines) >= 2:
        for raw_unit in lines[1].split(":", 1)[-1].split(","):
            name = _canonical_unit_name(raw_unit)
            if name and not _is_placeholder_unit_name(name) and name not in core_units:
                core_units.append(name)
    if len(lines) >= 3:
        augments = [a.strip() for a in lines[2].split(":", 1)[-1].split(",") if a.strip() and a.strip() != "无"]

    traits = []
    for part in archetype.split("+"):
        name = _canonical_trait_name(part)
        if name and name not in traits:
            traits.append(name)
    if not traits:
        for tag in tags:
            if tag.lower().startswith("set"):
                continue
            name = _canonical_trait_name(tag)
            if name and name not in traits:
                traits.append(name)
    return MetaReference(
        archetype=archetype,
        traits=traits,
        core_units=core_units,
        sample_size=sample_size,
        avg_placement=avg_placement,
        top4_rate=top4_rate,
        win_rate=win_rate,
        augments=augments,
        title=title,
        source=str(doc.get("source", "riot_api") or "riot_api"),
    )


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鏁版嵁缁撴瀯
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
@dataclass
class Doc:
    doc_id    : str
    source    : str
    title     : str
    content   : str
    url       : str
    fetched_at: str
    tags      : List[str] = field(default_factory=list)

    def chunks(self) -> List[Dict]:
        text  = f"[{self.source.upper()}] {self.title}\n{self.content}"
        size  = CFG["chunk_size"]
        step  = size - CFG["chunk_overlap"]
        parts = []
        i = 0
        while i < len(text):
            parts.append({
                "chunk_id": f"{self.doc_id}_{len(parts)}",
                "doc_id"  : self.doc_id,
                "source"  : self.source,
                "title"   : self.title,
                "url"     : self.url,
                "text"    : text[i:i + size],
                "tags"    : self.tags,
            })
            i += step
        return parts


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# BM25 鐭ヨ瘑搴?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
class JSONKnowledgeBase:
    KB_CHUNKS = DATA_DIR / "kb_chunks.json"
    IDF_FILE  = DATA_DIR / "kb_idf.json"

    def __init__(self):
        self.chunks: List[Dict] = []
        self.idf: Dict[str, float] = {}
        self._load()

    def _load(self):
        if self.KB_CHUNKS.exists():
            try:
                self.chunks = json.loads(self.KB_CHUNKS.read_text(encoding="utf-8"))
                logger.info(f"知识库加载: {len(self.chunks)} chunks")
            except Exception:
                pass
        if self.IDF_FILE.exists():
            try:
                self.idf = json.loads(self.IDF_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self):
        self.KB_CHUNKS.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.IDF_FILE.write_text(
            json.dumps(self.idf, ensure_ascii=False), encoding="utf-8"
        )

    def add_docs(self, docs: List[Doc]):
        existing = {c["chunk_id"] for c in self.chunks}
        new_chunks = []
        for doc in docs:
            for chunk in doc.chunks():
                if chunk["chunk_id"] not in existing:
                    new_chunks.append(chunk)
                    existing.add(chunk["chunk_id"])
        if not new_chunks:
            logger.info("知识库无新增（缓存命中）")
            return
        self.chunks.extend(new_chunks)
        self._rebuild_idf()
        self._save()
        logger.info(f"知识库新增 {len(new_chunks)} chunks，总计 {len(self.chunks)}")

    def _tokenize(self, text: str) -> List[str]:
        en = re.findall(r"[a-zA-Z]{2,}", text.lower())
        zh = re.sub(r"[^\u4e00-\u9fff]", "", text)
        bigrams = [zh[i:i+2] for i in range(len(zh) - 1)]
        return en + bigrams + list(zh)

    def _rebuild_idf(self):
        df: Dict[str, int] = defaultdict(int)
        N = len(self.chunks)
        for chunk in self.chunks:
            for t in set(self._tokenize(chunk["text"])):
                df[t] += 1
        self.idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        if not self.chunks:
            return []
        q_tokens = self._tokenize(query)
        k1, b = 1.5, 0.75
        avg_dl = sum(len(c["text"]) for c in self.chunks) / len(self.chunks) or 1
        scored = []
        for chunk in self.chunks:
            dl = len(chunk["text"])
            tf_map: Dict[str, int] = defaultdict(int)
            for t in self._tokenize(chunk["text"]):
                tf_map[t] += 1
            score = 0.0
            for t in q_tokens:
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                idf = self.idf.get(t, 1.0)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            for tag in chunk.get("tags", []):
                if tag.lower() in query.lower():
                    score += 2.0
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]

    def clear(self):
        self.chunks = []
        self.idf = {}
        for f in [self.KB_CHUNKS, self.IDF_FILE]:
            if f.exists():
                f.unlink()

    def stats(self) -> str:
        sources: Dict[str, int] = defaultdict(int)
        for c in self.chunks:
            sources[c["source"]] += 1
        detail = " | ".join(f"{s}:{n}" for s, n in sources.items())
        return f"{len(self.chunks)} chunks [{detail}]"


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Riot API 鐖櫕锛堜粎 TFT 鎺ュ彛锛?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
class TFTCrawler:
    """
    浣跨敤浠ヤ笅 TFT 涓撳睘 API锛堜笉娑夊強浠讳綍 LoL 鎺ュ彛锛夛細
      tft-league-v1    鈫?楂樼灞€鎺掕姒?
      tft-summoner-v1  鈫?summonerId 鈫?puuid
      tft-match-v1     鈫?瀵瑰眬 ID 鍒楄〃 + 瀵瑰眬璇︽儏
    """
    CACHE_FILE = DATA_DIR / "riot_cache.json"
    # Development Key 闄愰€燂細20 req/s (鐭湡) / 100 req/2min (闀挎湡)

    RATE_DELAY = 0.5

    def __init__(self):
        self.api_key = CFG.get("riot_api_key", "")
        self._cache: Dict = self._load_cache()


    def _load_cache(self) -> Dict:
        if self.CACHE_FILE.exists():
            try:
                return json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_FILE.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts = datetime.fromisoformat(self._cache[key]["ts"])
        return (datetime.now() - ts) < timedelta(hours=CFG["cache_ttl_hours"])

    def clear_cache(self):
        self._cache = {}
        if self.CACHE_FILE.exists():
            self.CACHE_FILE.unlink()
        logger.info("Riot API 缓存已清空")

    def sanitize_cache_bucket(self, key: str) -> bool:
        bucket = self._cache.get(key)
        if not isinstance(bucket, dict):
            return False
        docs = bucket.get("docs", [])
        if not isinstance(docs, list):
            return False

        changed = False
        clean_docs = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            clean_doc = _sanitize_riot_doc_payload(doc)
            clean_docs.append(clean_doc)
            if clean_doc != doc:
                changed = True

        if changed:
            bucket["docs"] = clean_docs
            self._save_cache()
        return changed

    def prune_non_current_cache_buckets(self, current_set: int) -> bool:
        keep_key = f"riot_kb_{current_set}"
        removable = [key for key in list(self._cache.keys()) if key.startswith("riot_kb_") and key != keep_key]
        if not removable:
            return False
        for key in removable:
            self._cache.pop(key, None)
        self._save_cache()
        return True


    def _get(self, url: str, params: Dict = None) -> Optional[Any]:
        if not self.api_key:
            return None
        headers = {"X-Riot-Token": self.api_key}
        for attempt in range(3):
            try:
                time.sleep(self.RATE_DELAY)
                r = requests.get(url, headers=headers, params=params, timeout=20)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 10))
                    logger.warning(f"Riot API rate limited, waiting {wait}s")
                    time.sleep(wait + 1)
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                logger.warning(f"HTTP {e.response.status_code}: {url[-60:]}")
                return None
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        return None


    def _get_top_players(self) -> List[str]:
        """Read top ladder players directly from league entries and return puuid tokens."""













        region = CFG["riot_region_platform"]
        base   = f"https://{region}.api.riotgames.com"
        puuids: List[str] = []
        sids:   List[str] = []
        limit  = CFG["riot_max_players"]

        tiers = list(CFG.get("riot_tiers", ("challenger", "grandmaster")))
        if "master" not in tiers:
            tiers.append("master")

        for tier in tiers:
            url  = f"{base}/tft/league/v1/{tier}"
            data = self._get(url)
            if not data:
                logger.warning(f"  {tier} leaderboard fetch failed (HTTP error or timeout)")
                continue

            entries = data.get("entries", [])
            if not entries:
                logger.warning(f"  {tier} leaderboard entries empty")
                continue

            entries.sort(key=lambda x: x.get("leaguePoints", 0), reverse=True)

            taken = 0
            for e in entries[:limit]:

                puuid = e.get("puuid")
                if puuid and puuid not in puuids:
                    puuids.append(puuid)
                    taken += 1
                else:

                    sid = e.get("summonerId")
                    if sid and sid not in sids:
                        sids.append(sid)
                        taken += 1

            logger.info(f"  {tier}: {len(entries)} players, collected {taken} ids")


        if not puuids and not sids:
            logger.warning("Failed to fetch any player IDs from the ladder. Check Development Key, region config, and network proxy.")









        result = puuids + [f"SID:{sid}" for sid in sids]
        return result


    def _get_puuid(self, summoner_id: str) -> Optional[str]:
        region = CFG["riot_region_platform"]
        url  = f"https://{region}.api.riotgames.com/tft/summoner/v1/summoners/{summoner_id}"
        data = self._get(url)
        return data.get("puuid") if data else None


    def _get_match_ids(self, puuid: str) -> List[str]:
        region = CFG["riot_region_regional"]
        url    = f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids"
        count  = CFG["riot_matches_per_player"]
        data   = self._get(url, params={"count": count, "type": "ranked"})
        return data if isinstance(data, list) else []


    def _get_match(self, match_id: str) -> Optional[Dict]:
        region = CFG["riot_region_regional"]
        url    = f"https://{region}.api.riotgames.com/tft/match/v1/matches/{match_id}"
        return self._get(url)


    @staticmethod
    def _parse_participant(p: Dict) -> Optional[Dict]:
        placement = p.get("placement")
        if not isinstance(placement, int) or not (1 <= placement <= 8):
            return None

        def strip(s: str) -> str:
            return re.sub(r"^(?:TFT(?:Set)?\d*_|Set\d+_)", "", s)

        def strip_aug(s: str) -> str:
            return re.sub(r"^TFT\w*?Augment_", "", s)

        traits = []
        for t in p.get("traits", []):
            if t.get("tier_current", 0) > 0:
                name = _canonical_trait_name(t.get("name_en") or t.get("name") or strip(t.get("name", "")))
                if not name:
                    continue
                traits.append({
                    "name" : name,
                    "count": t.get("num_units", 0),
                    "tier" : t.get("tier_current", 0),
                })
        traits.sort(key=lambda x: (-x["tier"], -x["count"]))

        units = [
            {
                "id"   : _canonical_unit_name(strip(u.get("character_id", ""))),
                "star" : u.get("tier", 1),
                "items": [strip(i) for i in u.get("itemNames", u.get("items", []))],
            }
            for u in p.get("units", [])
        ]

        augments = [strip_aug(a) for a in p.get("augments", [])]

        return {
            "placement": placement,
            "traits"   : traits,
            "units"    : units,
            "augments" : augments,
            "top4"     : placement <= 4,
            "win"      : placement == 1,
        }


    @staticmethod
    def _aggregate(participants: List[Dict]) -> List[Doc]:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for rec in participants:
            if not rec["traits"]:
                continue
            top2 = sorted(rec["traits"], key=lambda x: (-x["tier"], -x["count"]))[:2]
            key  = " + ".join(t["name"] for t in top2)
            groups[key].append(rec)

        docs = []
        for comp_key, records in sorted(groups.items(), key=lambda x: -len(x[1])):
            n = len(records)
            if n < 5:
                continue
            avg_pl = sum(r["placement"] for r in records) / n
            top4 = sum(1 for r in records if r["top4"]) / n
            wins = sum(1 for r in records if r["win"]) / n

            unit_ctr: Counter = Counter()
            for r in records:
                for u in r["units"]:
                    unit_ctr[u["id"]] += 1
            core = [uid for uid, cnt in unit_ctr.most_common(8) if cnt / n >= 0.5]

            aug_ctr: Counter = Counter()
            for r in records:
                for a in r["augments"]:
                    aug_ctr[a] += 1
            top_augs = [a for a, _ in aug_ctr.most_common(4) if a]

            content = (
                f"Comp: {comp_key} | Sample: {n} | AvgPlacement: {avg_pl:.2f} | Top4: {top4:.0%} | Win: {wins:.0%}\n"
                f"CoreUnits: {', '.join(core) or 'none'}\n"
                f"CommonAugments: {', '.join(top_augs) or 'none'}"
            )
            doc_id = hashlib.md5(comp_key.encode()).hexdigest()[:10]
            docs.append(Doc(
                doc_id=f"riot_{doc_id}",
                source="riot_api",
                title=f"[KR Ladder] {comp_key}",
                content=content,
                url="https://developer.riotgames.com/apis#tft-match-v1",
                fetched_at=datetime.now().isoformat(),
                tags=[f"set{_effective_set_number()}", *comp_key.split(" + "), *core[:3]],
            ))
        return docs


    def crawl(self) -> List[Doc]:
        if not self.api_key:
            logger.warning("RIOT_API_KEY is not set; skipping Riot API crawl")
            return []

        cache_key = f"riot_kb_{_effective_set_number()}"
        if self._cache_valid(cache_key):
            logger.info("Riot API cache is valid; reusing cached docs")
            cached_docs = self._cache[cache_key].get("docs", [])
            return [Doc(**d) for d in cached_docs]

        logger.info("=== Riot API crawl started ===")


        logger.info("Step 1: fetch high-rank ladder players")
        player_tokens = self._get_top_players()
        if not player_tokens:
            logger.warning(
                "Failed to fetch any player IDs from the Riot ladder.\n"
                "  Common causes:\n"
                "  1. The Development Key expired; generate a new one at https://developer.riotgames.com\n"
                "  2. Current riot_tiers config is " + str(CFG.get("riot_tiers")) + "\n"
                "  3. Network issue (VPN / firewall / proxy)"
            )
            return []





        logger.info(f"Step 2: resolve {len(player_tokens)} player tokens into PUUIDs")
        puuids: List[str] = []
        sid_count = 0
        for i, token in enumerate(player_tokens):
            if token.startswith("SID:"):
                sid = token[4:]
                sid_count += 1
                puuid = self._get_puuid(sid)
                if puuid:
                    puuids.append(puuid)
            else:
                puuids.append(token)
            if (i + 1) % 10 == 0:
                logger.info(f"  PUUID progress: {i+1}/{len(player_tokens)}")

        direct = len(player_tokens) - sid_count
        logger.info(f"  direct puuid: {direct}, via summonerId: {sid_count}, resolved: {len(puuids)}")


        
        logger.info(f"Step 3: crawl matches for {len(puuids)} players")
        seen_matches: set = set()
        all_participants: List[Dict] = []
        for i, puuid in enumerate(puuids):
            for mid in self._get_match_ids(puuid):
                if mid in seen_matches:
                    continue
                seen_matches.add(mid)
                match = self._get_match(mid)
                if not match:
                    continue
                info = match.get("info", {})
                if info.get("tft_game_type") not in ("standard", None, ""):
                    continue
                for p in info.get("participants", []):
                    rec = self._parse_participant(p)
                    if rec:
                        all_participants.append(rec)
            if (i + 1) % 5 == 0:
                logger.info(f"  player {i+1}/{len(puuids)}, matches {len(seen_matches)}, records {len(all_participants)}")


        logger.info(f"crawl complete: {len(seen_matches)} matches / {len(all_participants)} records")

        # Step 5: 鑱氬悎
        docs = self._aggregate(all_participants)
        logger.info(f"生成 {len(docs)} 个阵容文档")

        # 鍐欑紦瀛?
        self._cache[cache_key] = {
            "ts"  : datetime.now().isoformat(),
            "docs": [asdict(d) for d in docs],
        }
        self._save_cache()
        return docs


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鏈湴鏁版嵁鍔犺浇鍣?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

class LocalDataLoader:
    def __init__(self):
        self.analysis: Dict[str, Any] = {}
        self.champion_db: Dict[str, Any] = {}
        self.trait_db: Dict[str, Any] = {}
        self.item_db: Dict[str, Any] = {}
        self.trait_dict: Dict[str, Any] = {}
        self._champion_fact_index: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self):
        self.analysis = {}
        self.champion_db = {}
        self.trait_db = {}
        self.item_db = {}
        self.trait_dict = {}
        self._champion_fact_index = {}

        for path in [CFG["analysis_file"], "tft_team_analysis.json"]:
            p = Path(path)
            if not p.exists():
                continue
            try:
                self.analysis = json.loads(p.read_text(encoding="utf-8"))
                logger.info(f"阵容分析已加载: {path}")
                break
            except Exception as e:
                logger.warning(f"加载 {path} 失败: {e}")

        for attr, key in [
            ("champion_db", "champion_db_file"),
            ("trait_db", "trait_db_file"),
            ("item_db", "item_db_file"),
            ("trait_dict", "trait_dict_file"),
        ]:
            p = Path(CFG[key])
            if not p.exists():
                continue
            try:
                setattr(self, attr, json.loads(p.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"加载 {p} 失败: {e}")

        self._build_champion_fact_index()

    def reload(self):
        self._load_all()

    @staticmethod
    def _display_unit(unit: Dict[str, Any]) -> str:
        name = _unit_name(unit)
        if name.endswith("_TraitClone"):
            name = name.replace("_TraitClone", "")
        if name.lower().startswith("unknown"):
            return "未知"
        return name

    def _build_champion_fact_index(self):
        facts: Dict[str, Dict[str, Any]] = {}
        champion_traits: Dict[str, List[str]] = defaultdict(list)

        for trait_name, entry in self.trait_dict.items():
            if not isinstance(entry, dict):
                continue
            display_trait = _canonical_trait_name(
                entry.get("name_zh")
                or entry.get("name_cn")
                or entry.get("name_en")
                or entry.get("name")
                or trait_name
            )
            if not display_trait:
                continue
            for champ in entry.get("champions", []):
                champ_name = _canonical_unit_name(champ)
                if not champ_name or _is_placeholder_unit_name(champ_name):
                    continue
                if display_trait not in champion_traits[champ_name]:
                    champion_traits[champ_name].append(display_trait)

        for champ_id, entry in self.champion_db.items():
            if not isinstance(entry, dict):
                continue
            name = _canonical_unit_name(
                entry.get("name_zh")
                or entry.get("name_cn")
                or entry.get("name_en")
                or entry.get("name")
                or entry.get("short_id")
                or champ_id
            )
            if not name or _is_placeholder_unit_name(name):
                continue

            traits: List[str] = []
            raw_traits = entry.get("traits", []) if isinstance(entry.get("traits"), list) else []
            for raw_trait in raw_traits:
                trait_name = _canonical_trait_name(raw_trait)
                if trait_name and trait_name not in traits:
                    traits.append(trait_name)
            for trait_name in champion_traits.get(name, []):
                if trait_name not in traits:
                    traits.append(trait_name)

            fact = {
                "id": champ_id,
                "name": name,
                "cost": _safe_int(entry.get("cost", 0), 0),
                "traits": traits,
            }
            keys = {
                name,
                _clean_display_name(entry.get("short_id")),
                _clean_display_name(entry.get("id")),
                _clean_display_name(champ_id),
            }
            for key in keys:
                if not key:
                    continue
                facts[key] = fact
                facts[_canonical_unit_name(key)] = fact

        self._champion_fact_index = facts

    def champion_fact(self, name: Any) -> Optional[Dict[str, Any]]:
        key = _canonical_unit_name(name)
        if not key:
            return None
        return self._champion_fact_index.get(key)

    def champion_fact_lines(self, names: List[str], limit: int = 18) -> List[str]:
        lines: List[str] = []
        seen: set[str] = set()
        for raw_name in names:
            fact = self.champion_fact(raw_name)
            if not fact:
                continue
            name = fact["name"]
            if name in seen:
                continue
            seen.add(name)
            traits = ", ".join(fact.get("traits", [])[:4]) or "未知"
            cost = fact.get("cost", 0)
            cost_text = f"{cost}费" if cost else "费用未知"
            lines.append(f"{name}: {cost_text} | 羁绊: {traits}")
            if len(lines) >= limit:
                break
        return lines

    def trait_fact_lines(self, names: List[str], limit: int = 12) -> List[str]:
        lines: List[str] = []
        seen: set[str] = set()
        index: Dict[str, str] = {}

        for trait_name, entry in self.trait_dict.items():
            if not isinstance(entry, dict):
                continue
            display_name = _canonical_trait_name(
                entry.get("name_zh")
                or entry.get("name_cn")
                or entry.get("name_en")
                or entry.get("name")
                or trait_name
            )
            if not display_name:
                continue
            champs: List[str] = []
            for champ in entry.get("champions", []):
                cname = _canonical_unit_name(champ)
                if cname and not _is_placeholder_unit_name(cname) and cname not in champs:
                    champs.append(cname)
            levels = entry.get("activation", {}).get("levels", [])
            line = f"{display_name}: 阈值={levels or []} | 棋子: {', '.join(champs[:8]) or '未知'}"
            keys = {
                display_name,
                _canonical_trait_name(trait_name),
                _clean_display_name(trait_name),
                _clean_display_name(entry.get("short_id")),
                _clean_display_name(entry.get("api_name")),
                _clean_display_name(entry.get("apiName")),
                _clean_display_name(entry.get("id")),
            }
            for key in keys:
                if key:
                    index[key] = line

        for raw_name in names:
            key = _canonical_trait_name(raw_name) or _clean_display_name(raw_name)
            if not key or key in seen:
                continue
            line = index.get(key)
            if not line:
                continue
            seen.add(key)
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def to_docs(self) -> List[Doc]:
        docs: List[Doc] = []

        if self.analysis:
            docs.append(Doc(
                doc_id="local_analysis",
                source="local",
                title="当前阵容分析",
                content=self._fmt_analysis(),
                url="local://tft_team_analysis.json",
                fetched_at=datetime.now().isoformat(),
                tags=self.analysis_tags(),
            ))

        if self.trait_dict:
            lines: List[str] = []
            for tname, tdata in self.trait_dict.items():
                if not isinstance(tdata, dict):
                    continue
                display_name = _canonical_trait_name(tname)
                if not display_name:
                    continue
                champs = ", ".join(_canonical_unit_name(champ) for champ in tdata.get("champions", []) if _canonical_unit_name(champ))
                levels = tdata.get("activation", {}).get("levels", [])
                lines.append(f"[{display_name}] champions: {champs} | levels: {levels}")
            docs.append(Doc(
                doc_id="local_traits",
                source="local",
                title="羁绊数据",
                content="\n".join(lines[:80]),
                url="local://tft_trait_champion_dict.json",
                fetched_at=datetime.now().isoformat(),
                tags=["trait", "activation", f"set{_effective_set_number()}"] ,
            ))

        if self.champion_db:
            lines: List[str] = []
            added: set[str] = set()
            for fact in self._champion_fact_index.values():
                name = fact.get("name", "")
                if not name or name in added:
                    continue
                added.add(name)
                traits = ", ".join(fact.get("traits", [])[:4]) or "未知"
                cost = fact.get("cost", 0)
                cost_text = f"{cost}费" if cost else "费用未知"
                lines.append(f"[{name}] cost: {cost_text} | traits: {traits}")
            docs.append(Doc(
                doc_id="local_champions",
                source="local",
                title="英雄费用与羁绊数据",
                content="\n".join(sorted(lines)[:120]),
                url="local://tft_champion_db.json",
                fetched_at=datetime.now().isoformat(),
                tags=["champion", "cost", "trait", f"set{_effective_set_number()}"],
            ))

        return docs

    def _fmt_single_analysis(self, analysis: Dict[str, Any]) -> str:
        parts = [f"team_size: {analysis.get('team_size', len(analysis.get('champions', [])))}"]
        champs = [c for c in analysis.get("champions", []) if isinstance(c, dict)]
        if champs:
            cstrs = []
            for champ in champs:
                name = self._display_unit(champ)
                star = _safe_int(champ.get("star", 1), 1)
                items = ", ".join(champ.get("items", [])) or "none"
                fact = self.champion_fact(name)
                cost = fact.get("cost", 0) if fact else 0
                cost_text = f"{cost}费 " if cost else ""
                cstrs.append(f"{name} {cost_text}{star}* ({items})")
            parts.append("champions: " + " / ".join(cstrs))
        traits = [t for t in analysis.get("traits", []) if isinstance(t, dict)]
        if traits:
            tstrs = []
            for trait in traits:
                name = _trait_name(trait) or "unknown_trait"
                count = _safe_int(trait.get("count", 0), 0)
                lvl = trait.get("level_name") or trait.get("level") or trait.get("tier") or ""
                tstrs.append(f"{name}({count}{'/' + str(lvl) if lvl else ''})")
            parts.append("traits: " + ", ".join(tstrs))
        summary = analysis.get("summary", {}) if isinstance(analysis.get("summary"), dict) else {}
        if summary.get("main_carry"):
            parts.append(f"main_carry: {summary['main_carry']}")
        if summary.get("front_row_ratio"):
            parts.append(f"front_row_ratio: {summary['front_row_ratio']}")
        if analysis.get("equipment_issues"):
            parts.append("equipment_issues: " + "; ".join(map(str, analysis["equipment_issues"])))
        return "\n".join(parts)

    def _fmt_analysis(self) -> str:
        analysis = self.analysis or {}
        players = analysis.get("players") if isinstance(analysis.get("players"), list) else []
        if players:
            lines = ["mode: global", f"players: {len(players)}"]
            for idx, player in enumerate(players[:8], 1):
                champs = [self._display_unit(c) for c in player.get("champions", []) if isinstance(c, dict)]
                traits = [_trait_name(t) for t in player.get("traits", []) if isinstance(t, dict) and _trait_name(t)]
                lines.append(
                    f"rank {idx}: champions={', '.join(champs[:9]) or 'none'} | traits={', '.join(traits[:4]) or 'none'}"
                )
            return "\n".join(lines)
        return self._fmt_single_analysis(analysis)

    def analysis_tags(self) -> List[str]:
        tags: List[str] = []
        players = self.analysis.get("players") if isinstance(self.analysis.get("players"), list) else []
        if players:
            for player in players[:8]:
                for trait in player.get("traits", []):
                    if isinstance(trait, dict):
                        name = _trait_name(trait)
                        if name:
                            tags.append(name)
                for champ in player.get("champions", []):
                    if isinstance(champ, dict):
                        name = self._display_unit(champ)
                        if name and name != "未知":
                            tags.append(name)
            return [t for t in tags if t][:24]

        for trait in self.analysis.get("traits", []):
            if isinstance(trait, dict):
                name = _trait_name(trait)
                if name:
                    tags.append(name)
        for champ in self.analysis.get("champions", []):
            if isinstance(champ, dict):
                name = self._display_unit(champ)
                if name and name != "未知":
                    tags.append(name)
        return [t for t in tags if t]

    def team_summary(self) -> str:
        return self._fmt_analysis() if self.analysis else "(未检测到阵容数据)"

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._mailboxes: Dict[str, List[AgentMessage]] = defaultdict(list)
        self._handlers: Dict[str, List[Any]] = defaultdict(list)

    def subscribe(self, agent_name: str, msg_type: str, handler):
        with self._lock:
            self._handlers[msg_type].append((agent_name, handler))

    def publish(self, msg: AgentMessage):
        with self._lock:
            if msg.receiver == "*":
                for name in self._mailboxes:
                    if name != msg.sender:
                        self._mailboxes[name].append(msg)
            else:
                self._mailboxes[msg.receiver].append(msg)
            handlers = list(self._handlers.get(msg.msg_type, []))

        for agent_name, handler in handlers:
            if agent_name == msg.sender:
                continue
            try:
                handler(msg)
            except Exception as e:
                logger.warning(f"[AgentBus] handler error ({agent_name}): {e}")

    def get_messages(self, agent_name: str) -> List[AgentMessage]:
        with self._lock:
            msgs = list(self._mailboxes.get(agent_name, []))
            self._mailboxes[agent_name] = []
            return msgs


class BaseAgent:
    def __init__(self, name: str, bus: AgentBus):
        self.name = name
        self.bus = bus
        self.state: Dict[str, Any] = {}
        self._result: Optional[str] = None

    def emit(self, msg_type: str, payload: Dict[str, Any], receiver: str = "*"):
        self.bus.publish(AgentMessage(
            sender=self.name,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
        ))

    def run(self, analysis: Dict[str, Any]) -> str:
        raise NotImplementedError


class EconomyAgent(BaseAgent):
    def __init__(self, bus: AgentBus):
        super().__init__("EconomyAgent", bus)

    def run(self, analysis: Dict[str, Any]) -> str:
        team_size = _safe_int(analysis.get("team_size", 0), 0)
        champs = [c for c in analysis.get("champions", []) if isinstance(c, dict)]
        avg_star = sum(_safe_int(c.get("star", 1), 1) for c in champs) / max(len(champs), 1)

        if team_size <= 5:
            phase, advice = "前期", "优先保经济和连胜/连败节奏，不急于硬D。"
        elif team_size <= 7:
            phase, advice = "中期", "根据当前质量决定是补两星还是准备升人口。"
        else:
            phase, advice = "后期", "以保血和成型为主，金币应服务于核心位和关键挂件。"

        parts = [f"阶段判断: {phase}", f"运营建议: {advice}"]
        if avg_star < 1.6:
            parts.append("星级偏低，说明场面质量还没站稳，需要补两星或缩小阵容方向。")
        elif avg_star >= 2.1:
            parts.append("星级基础不错，可以更关注人口、上限和针对性补强。")

        self.emit("phase_result", {
            "phase": phase,
            "avg_star": round(avg_star, 2),
            "team_size": team_size,
        })
        self.state.update({"phase": phase, "avg_star": avg_star})
        self._result = "\n".join(parts)
        return self._result


class PowerAgent(BaseAgent):
    def __init__(self, bus: AgentBus, trait_dict: Dict[str, Any]):
        super().__init__("PowerAgent", bus)
        self.trait_dict = trait_dict
        self._phase = "未知"
        bus.subscribe(self.name, "phase_result", self._on_phase)

    def _on_phase(self, msg: AgentMessage):
        self._phase = str(msg.payload.get("phase", "未知"))
        self.state["phase"] = self._phase

    def run(self, analysis: Dict[str, Any]) -> str:
        traits = [t for t in analysis.get("traits", []) if isinstance(t, dict)]
        champs = [c for c in analysis.get("champions", []) if isinstance(c, dict)]
        active_traits = [t for t in traits if _safe_int(t.get("count", 0), 0) > 0 or t.get("level") or t.get("tier")]
        trait_names = [_trait_name(t) for t in active_traits if _trait_name(t)]

        parts: List[str] = []
        if trait_names:
            parts.append("核心羁绊: " + ", ".join(trait_names[:4]))
        else:
            parts.append("羁绊信息不足，当前更像原始识别结果而不是完整成型阵容。")

        suggestions: List[str] = []
        for trait in active_traits[:4]:
            tname = _trait_name(trait)
            if not tname:
                continue
            count = _safe_int(trait.get("count", 0), 0)
            levels = self.trait_dict.get(tname, {}).get("activation", {}).get("levels", [])
            for lvl in levels:
                if count < lvl <= count + 2:
                    suggestions.append(f"{tname} 距离下一档还差 {lvl - count} 个")
                    break
        if suggestions:
            parts.append("羁绊补强: " + " | ".join(suggestions[:3]))

        all_items: List[str] = []
        no_item_carries: List[str] = []
        for champ in champs:
            items = champ.get("items", []) if isinstance(champ.get("items"), list) else []
            all_items.extend(items)
            cost_thr = 3 if self._phase == "后期" else 4
            if not items and _safe_int(champ.get("cost", 1), 1) >= cost_thr:
                name = _unit_name(champ)
                if name and not name.lower().startswith("unknown"):
                    no_item_carries.append(name)

        if no_item_carries:
            parts.append("高费空装位: " + ", ".join(no_item_carries[:4]))
        parts.append(f"已识别装备数: {len(all_items)}")

        self.emit("power_result", {
            "key_traits": trait_names[:4],
            "item_count": len(all_items),
            "no_item_carries": no_item_carries,
        })
        self._result = "\n".join(parts)
        return self._result

class PositionAgent(BaseAgent):
    def __init__(self, bus: AgentBus):
        super().__init__("PositionAgent", bus)
        self._no_item_carries: List[str] = []
        bus.subscribe(self.name, "power_result", self._on_power)

    def _on_power(self, msg: AgentMessage):
        self._no_item_carries = list(msg.payload.get("no_item_carries", []))
        self.state["no_item_carries"] = self._no_item_carries

    def run(self, analysis: Dict[str, Any]) -> str:
        parts: List[str] = []
        champs = [c for c in analysis.get("champions", []) if isinstance(c, dict)]
        summary = analysis.get("summary", {}) if isinstance(analysis.get("summary"), dict) else {}

        front_ratio = summary.get("front_row_ratio")
        if front_ratio:
            parts.append(f"前后排比例: {front_ratio}")

        positions = [c.get("position") for c in champs if isinstance(c.get("position"), dict)]
        if positions:
            rows = [_safe_int(pos.get("row", 0), 0) for pos in positions]
            front_row = sum(1 for row in rows if row >= 3)
            back_row = len(rows) - front_row
            parts.append(f"站位统计: 前排 {front_row} / 后排 {back_row}")
            if front_row < 2:
                parts.append("前排偏少，容易被直接穿透。")
            elif back_row < 2:
                parts.append("后排偏少，输出位可能不足。")
        else:
            parts.append("没有可靠站位坐标，无法做精细对位判断。")

        main_carry = summary.get("main_carry")
        if main_carry:
            parts.append(f"当前主C: {main_carry}")
        if self._no_item_carries:
            parts.append("关键输出位仍有空装，站位上应优先缩角保护。")
        if analysis.get("equipment_issues"):
            parts.append("装备异常: " + "; ".join(map(str, analysis.get("equipment_issues", [])[:3])))

        self._result = "\n".join(parts) if parts else "站位数据不足"
        return self._result


def _normalize_profile_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text or text.lower().startswith("unknown"):
        return ""
    return text.replace("_TraitClone", "")


def _extract_units(analysis: Dict[str, Any]) -> List[str]:
    units: List[str] = []
    for champ in analysis.get("champions", []):
        if not isinstance(champ, dict):
            continue
        name = _normalize_profile_name(_unit_name(champ))
        if name:
            units.append(name)
    return units


def _extract_traits(analysis: Dict[str, Any]) -> List[str]:
    traits: List[str] = []
    for trait in analysis.get("traits", []):
        if not isinstance(trait, dict):
            continue
        if not (_safe_int(trait.get("count", 0), 0) > 0 or trait.get("level") or trait.get("tier") or trait.get("style")):
            continue
        name = _normalize_profile_name(_trait_name(trait))
        if name:
            traits.append(name)
    return traits


def _build_profile(analysis: Dict[str, Any], label: str = "", rank: Optional[int] = None) -> Dict[str, Any]:
    champs = [c for c in analysis.get("champions", []) if isinstance(c, dict)]
    units = _extract_units(analysis)
    traits = _extract_traits(analysis)
    item_count = sum(len(c.get("items", [])) for c in champs if isinstance(c.get("items"), list))
    two_star = sum(1 for c in champs if _safe_int(c.get("star", 1), 1) >= 2)
    three_star = sum(1 for c in champs if _safe_int(c.get("star", 1), 1) >= 3)
    carry = max(
        champs,
        key=lambda c: (
            len(c.get("items", [])) if isinstance(c.get("items"), list) else 0,
            _safe_int(c.get("star", 1), 1),
            _safe_float(c.get("_score", 0.0), 0.0),
        ),
        default=None,
    )
    carry_name = _normalize_profile_name(_unit_name(carry)) if carry else ""
    return {
        "label": label or analysis.get("label") or analysis.get("player_name") or analysis.get("name") or "当前阵容",
        "rank": rank,
        "team_size": _safe_int(analysis.get("team_size", len(champs)), len(champs)),
        "units": units,
        "traits": traits,
        "item_count": item_count,
        "two_star": two_star,
        "three_star": three_star,
        "carry": carry_name,
        "carry_star": _safe_int(carry.get("star", 1), 1) if carry else 1,
        "front_row_ratio": (analysis.get("summary") or {}).get("front_row_ratio", ""),
        "main_carry": (analysis.get("summary") or {}).get("main_carry", ""),
    }


def _meta_match_details(profile: Dict[str, Any], ref: MetaReference) -> Dict[str, Any]:
    unit_overlap = [u for u in ref.core_units if u in profile["units"]]
    trait_overlap = [t for t in ref.traits if t in profile["traits"]]
    missing_units = [u for u in ref.core_units if u not in profile["units"]]
    missing_traits = [t for t in ref.traits if t not in profile["traits"]]
    unit_ratio = len(unit_overlap) / max(1, min(len(ref.core_units), 6))
    trait_ratio = len(trait_overlap) / max(1, len(ref.traits))
    score = trait_ratio * 2.2 + unit_ratio * 1.8
    if profile.get("carry") and profile["carry"] in ref.core_units:
        score += 0.8
    return {
        "ref": ref,
        "unit_overlap": unit_overlap,
        "trait_overlap": trait_overlap,
        "missing_units": missing_units,
        "missing_traits": missing_traits,
        "unit_ratio": unit_ratio,
        "trait_ratio": trait_ratio,
        "score": score,
    }

class CompetitiveAgent(BaseAgent):
    def __init__(self, bus: AgentBus):
        super().__init__("CompetitiveAgent", bus)

    def run(self, analysis: Dict[str, Any]) -> str:
        profile = analysis.get("_profile") if isinstance(analysis.get("_profile"), dict) else _build_profile(analysis)
        raw_refs = analysis.get("_meta_refs", [])
        refs: List[MetaReference] = []
        for entry in raw_refs:
            if isinstance(entry, MetaReference):
                refs.append(entry)
            elif isinstance(entry, dict):
                try:
                    refs.append(MetaReference(**entry))
                except Exception:
                    continue

        if not refs:
            self._result = "竞赛参照不足，当前只能基于识别到的阵容结构做保守判断。"
            return self._result

        details = [_meta_match_details(profile, ref) for ref in refs]
        best = max(details, key=lambda item: item["score"])
        ref = best["ref"]
        completion = (best["unit_ratio"] + best["trait_ratio"]) / 2.0
        if completion >= 0.7:
            completion_text = "接近成型"
        elif completion >= 0.4:
            completion_text = "半成型"
        else:
            completion_text = "偏离主流模板"

        lines = [
            f"最接近竞赛模板: {ref.archetype}",
            f"竞赛表现: 样本 {ref.sample_size} | 平均名次 {ref.avg_placement:.2f} | Top4 {ref.top4_rate:.0%} | 吃鸡 {ref.win_rate:.0%}",
            f"重合点: 羁绊 {', '.join(best['trait_overlap']) or '无'} | 核心棋子 {', '.join(best['unit_overlap']) or '无'}",
            f"缺口: 羁绊 {', '.join(best['missing_traits'][:4]) or '无'} | 核心棋子 {', '.join(best['missing_units'][:5]) or '无'}",
            f"成型度判断: {completion_text}",
        ]
        self._result = "\n".join(lines)
        return self._result


class AgentOrchestrator:
    def __init__(self, agents: List[BaseAgent], bus: AgentBus):
        self.agents = agents
        self.bus = bus
        self._results: Dict[str, str] = {}

    def run_parallel(self, analysis: Dict[str, Any], timeout: float = 10.0) -> Dict[str, str]:
        self._results = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.agents)), thread_name_prefix="tft_agent") as executor:
            future_map = {executor.submit(agent.run, analysis): agent.name for agent in self.agents}
            for future in as_completed(future_map, timeout=timeout):
                name = future_map[future]
                try:
                    self._results[name] = future.result()
                except Exception as e:
                    self._results[name] = f"{name} 分析失败: {e}"
                    logger.warning(f"[Orchestrator] {name} error: {e}")
        return self._results

    def synthesize(self) -> str:
        order = ["EconomyAgent", "PowerAgent", "PositionAgent", "CompetitiveAgent"]
        labels = {
            "EconomyAgent": "[运营 Agent]",
            "PowerAgent": "[战力 Agent]",
            "PositionAgent": "[站位 Agent]",
            "CompetitiveAgent": "[竞赛 Agent]",
        }
        lines: List[str] = []
        for name in order:
            if name not in self._results:
                continue
            lines.append(labels[name])
            lines.append(self._results[name])
            lines.append("")
        return "\n".join(lines).strip()


class LLMClient:
    def __init__(self):
        self.api_key = self._resolve_key()
        self.base_url = CFG["sophnet_base_url"]
        self._client: Optional[OpenAI] = None

    @property
    def provider(self) -> str:
        return str(CFG.get("llm_provider", "sophnet"))

    def _resolve_key(self) -> str:
        key = CFG.get("sophnet_api_key", "")
        if key:
            return key
        if not sys.stdin.isatty():
            logger.warning("LLM API Key 未配置(provider=sophnet)")
            return ""
        print("\n未配置 SOPHNET_API_KEY")
        print("获取地址: https://www.sophnet.com/")
        choice = input("输入 1 手动填写，或直接回车跳过: ").strip()
        if choice == "1":
            import getpass
            key = getpass.getpass("SOPHNET_API_KEY: ").strip()
            if key:
                os.environ["SOPHNET_API_KEY"] = key
                CFG["sophnet_api_key"] = key
                return key
        return ""

    @property
    def model(self) -> str:
        return CFG["sophnet_model"]

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, system: str, user: str) -> str:
        if not self.api_key:
            return "LLM 未配置，请设置 SOPHNET_API_KEY。"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=CFG["max_tokens"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return self._handle_err(e)

    @staticmethod
    def _handle_err(e: Exception) -> str:
        from openai import APIStatusError, APIConnectionError, RateLimitError, AuthenticationError

        if isinstance(e, AuthenticationError):
            return "API Key 无效，请检查配置。"
        if isinstance(e, RateLimitError):
            return "请求频率过高或额度不足，请稍后再试。"
        if isinstance(e, APIConnectionError):
            return f"连接失败: {e}"
        if isinstance(e, APIStatusError):
            code = e.status_code
            return {402: "余额不足，请检查账户状态。"}.get(code, f"HTTP {code}: {str(e)[:200]}")
        return f"请求失败: {e}"


SYSTEM_PROMPT = """你是专业的云顶之弈战术顾问，只能基于输入给出的数据、子 Agent 报告和竞赛参考来回答。

要求：
1. 当前赛季是 Set{set_num}，禁止引用旧赛季、杜撰羁绊名、杜撰棋子名、杜撰装备。
2. 优先使用用户输入数据、本地英雄数据（尤其是费用/羁绊）和竞赛参考做对照分析，不要把检索到的竞赛模板直接当成当前阵容。
3. 如果是 global 模式，要逐个比较每位玩家的阵容完成度、核心棋子、装备完成度、与竞赛模板的接近程度，再给出谁更强、谁更有上限、谁更像锁前二或容易掉线。
4. 对棋子费用、正式羁绊等事实，若本地英雄数据已给出，就必须以本地数据为准，不允许自行猜测。
5. 严禁把一个正式羁绊名擅自改写成另一个羁绊名；如果本地数据写的是“海魔人”，就不能改成“海克斯”，如果写的是“挑战者”，就不能改成“狂战士”。
6. 如果本地数据与竞赛文档或你的常识冲突，只能说明“数据存在冲突，建议复核”，不能代替用户改名或纠错。
7. “海克斯”在竞赛文本里通常指强化符文/augment，不可把它当作羁绊别名。
8. 如果识别信息明显不足，要明确说“不足以确定”，而不是编造。
9. 用中文回答，结论要可执行，少说空话。

输出格式：
## 阵容评价
## 竞赛对照
## 关键差距
## 建议
"""

class TFTRagAgent:
    def __init__(self):
        logger.info("初始化 TFT RAG Agent")
        self.local = LocalDataLoader()
        self.kb = JSONKnowledgeBase()
        self.crawler = TFTCrawler()
        self.llm = LLMClient()

        self.bus = AgentBus()
        self.economy_agent = EconomyAgent(self.bus)
        self.power_agent = PowerAgent(self.bus, self.local.trait_dict)
        self.position_agent = PositionAgent(self.bus)
        self.competitive_agent = CompetitiveAgent(self.bus)
        self.orchestrator = AgentOrchestrator(
            agents=[self.economy_agent, self.power_agent, self.position_agent, self.competitive_agent],
            bus=self.bus,
        )

        if not self.crawler.api_key:
            key = CFG.get("riot_api_key") or os.getenv("RIOT_API_KEY", "")
            if key:
                self.crawler.api_key = key

    def _set_number(self) -> int:
        return _effective_set_number(CFG.get("current_set", 17))

    def _kb_needs_cleanup(self) -> bool:
        for chunk in self.kb.chunks:
            if not isinstance(chunk, dict):
                continue
            blob = " ".join([
                str(chunk.get("title", "") or ""),
                str(chunk.get("text", "") or ""),
                " ".join(str(tag) for tag in chunk.get("tags", []) if tag),
            ])
            if _doc_needs_trait_cleanup(blob):
                return True
        return False

    def _rebuild_kb_from_cache(self):
        set_num = self._set_number()
        cache_key = f"riot_kb_{set_num}"
        cached_docs = self.crawler._cache.get(cache_key, {}).get("docs", [])
        riot_docs = [Doc(**doc) for doc in cached_docs if isinstance(doc, dict)]
        local_docs = self.local.to_docs()
        self.kb.clear()
        if riot_docs or local_docs:
            self.kb.add_docs(riot_docs + local_docs)

    def build_kb(self, force: bool = False, background: Optional[bool] = None):
        if background is None:
            background = bool(CFG.get("background_crawl", True))

        if force:
            self.crawler.clear_cache()
            self.kb.clear()

        set_num = self._set_number()
        cache_key = f"riot_kb_{set_num}"
        if self.crawler.prune_non_current_cache_buckets(set_num):
            logger.info("已移除非当前赛季的 Riot 缓存桶，避免旧赛季数据混入")
        if self.crawler.sanitize_cache_bucket(cache_key):
            logger.info("检测到旧版 Riot 文档命名，已自动清洗为当前赛季正式名称")
        if self._kb_needs_cleanup():
            logger.info("检测到旧版知识库条目，正在基于清洗后的缓存重建本地索引")
            self._rebuild_kb_from_cache()
        has_kb = bool(self.kb.chunks)
        riot_cache_valid = self.crawler._cache_valid(cache_key)

        if riot_cache_valid and not has_kb:
            logger.info("从 Riot 缓存快速加载知识库...")
            cached_docs = self.crawler._cache.get(cache_key, {}).get("docs", [])
            riot_docs = [Doc(**doc) for doc in cached_docs]
            local_docs = self.local.to_docs()
            self.kb.add_docs(riot_docs + local_docs)
            self._print_kb_stats(riot_docs, local_docs)
            return

        if not has_kb and not riot_cache_valid:
            print("\n" + "=" * 52)
            print("首次构建知识库，正在抓取 Riot 竞赛数据...")
            print(f"预计规模: {CFG['riot_max_players']} 名玩家 x {CFG['riot_matches_per_player']} 局")
            print("=" * 52)
            self._do_crawl_and_build()
            return

        if background and riot_cache_valid:
            logger.info(f"知识库已就绪({self.kb.stats()})，缓存仍有效，跳过刷新")
            local_docs = self.local.to_docs()
            if local_docs:
                self.kb.add_docs(local_docs)
            return

        if background:
            logger.info(f"知识库已就绪({self.kb.stats()})，后台刷新 Riot 数据...")
            thread = threading.Thread(target=self._do_crawl_and_build, daemon=True)
            thread.start()
        else:
            self._do_crawl_and_build()

    def _do_crawl_and_build(self):
        riot_docs = self.crawler.crawl()
        local_docs = self.local.to_docs()
        if riot_docs or local_docs:
            self.kb.add_docs(riot_docs + local_docs)
        self._print_kb_stats(riot_docs, local_docs)

    def _print_kb_stats(self, riot_docs: List[Doc], local_docs: List[Doc]):
        source_counter = Counter(doc.source for doc in riot_docs + local_docs)
        print(f"  Riot API : {source_counter.get('riot_api', 0)} 个文档")
        print(f"  本地数据 : {source_counter.get('local', 0)} 个文档")
        print(f"  知识库总计: {self.kb.stats()}")

    def _cached_riot_docs(self) -> List[Dict[str, Any]]:
        set_num = self._set_number()
        cache_key = f"riot_kb_{set_num}"
        bucket = self.crawler._cache.get(cache_key, {})
        docs = bucket.get("docs", []) if isinstance(bucket, dict) else []
        if docs:
            return docs
        return []

    def _current_set_meta_refs(self) -> List[MetaReference]:
        refs: List[MetaReference] = []
        seen: set[str] = set()
        for doc in self._cached_riot_docs():
            ref = _parse_meta_reference(doc)
            if not ref or not ref.archetype:
                continue
            if ref.archetype in seen:
                continue
            seen.add(ref.archetype)
            refs.append(ref)
        return refs

    def _select_meta_refs_for_profile(self, profile: Dict[str, Any], limit: int = 3) -> List[MetaReference]:
        scored: List[Tuple[float, MetaReference]] = []
        for ref in self._current_set_meta_refs():
            detail = _meta_match_details(profile, ref)
            if detail["score"] <= 0:
                continue
            scored.append((detail["score"], ref))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [ref for _, ref in scored[:limit]]

    def _prepare_agent_input(self, analysis: Dict[str, Any], label: str = "", rank: Optional[int] = None) -> Dict[str, Any]:
        payload = dict(analysis or {})
        profile = _build_profile(payload, label=label, rank=rank)
        refs = self._select_meta_refs_for_profile(profile, limit=3)
        payload["_profile"] = profile
        payload["_meta_refs"] = [asdict(ref) for ref in refs]
        return payload

    def _run_sub_agents_for_analysis(self, analysis: Dict[str, Any], label: str = "", rank: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        prepared = self._prepare_agent_input(analysis, label=label, rank=rank)
        self.orchestrator.run_parallel(prepared)
        return self.orchestrator.synthesize(), prepared

    @staticmethod
    def _format_meta_refs(refs: List[MetaReference]) -> str:
        if not refs:
            return "(无可用竞赛参考)"
        lines = []
        for idx, ref in enumerate(refs, 1):
            lines.append(
                f"[{idx}] {ref.archetype} | 样本 {ref.sample_size} | 平均名次 {ref.avg_placement:.2f} | Top4 {ref.top4_rate:.0%} | 吃鸡 {ref.win_rate:.0%} | 核心 {', '.join(ref.core_units[:6]) or '无'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_kb_hits(hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "(暂无外部参考)"
        lines = []
        for idx, hit in enumerate(hits, 1):
            lines.append(f"[参考 {idx} | {hit.get('source', '?')}] {hit.get('title', '')}")
            lines.append(hit.get("text", ""))
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _profile_tags(profile: Dict[str, Any]) -> List[str]:
        tags: List[str] = []
        tags.extend(profile.get("traits", [])[:4])
        tags.extend(profile.get("units", [])[:6])
        if profile.get("carry"):
            tags.append(profile["carry"])
        return [tag for tag in tags if tag]

    def _local_fact_context(self, names: List[str], traits: List[str], limit: int = 18) -> str:
        trait_lines = self.local.trait_fact_lines(traits, limit=max(6, min(12, limit)))
        champ_lines = self.local.champion_fact_lines(names, limit=max(6, limit))
        lines: List[str] = []
        if trait_lines:
            lines.append("[本地羁绊事实]")
            lines.extend(trait_lines)
        if champ_lines:
            if lines:
                lines.append("")
            lines.append("[本地英雄事实]")
            lines.extend(champ_lines)
        return "\n".join(lines) if lines else "(当前本地英雄库未命中费用/羁绊数据)"

    def _build_single_mode_context(self, mode: str) -> Tuple[str, str, List[str], str, str]:
        analysis = self.local.analysis or {}
        report, prepared = self._run_sub_agents_for_analysis(analysis, label="当前阵容")
        profile = prepared["_profile"]
        refs = [MetaReference(**entry) for entry in prepared.get("_meta_refs", [])]
        team_desc = self.local.team_summary()
        competitive_desc = self._format_meta_refs(refs)
        tags = self._profile_tags(profile)
        local_champion_desc = self._local_fact_context(profile.get("units", []) + ([profile.get("carry")] if profile.get("carry") else []), profile.get("traits", []), limit=12)
        if mode == "duel":
            default_question = "请结合当前识别到的双方信息，分析对位关系、阵容上限、装备完成度和优先补强点。"
        else:
            default_question = "请基于当前阵容、竞赛模板和子 Agent 报告，给出最可靠的阵容评价与后续建议。"
        sub_reports = report + "\n\n[竞赛参考]\n" + competitive_desc
        return team_desc, sub_reports, tags, "当前阵容", default_question, local_champion_desc

    def _build_global_mode_context(self) -> Tuple[str, str, List[str], str, str]:
        analysis = self.local.analysis or {}
        players = analysis.get("players") if isinstance(analysis.get("players"), list) else []
        if not players:
            return self._build_single_mode_context("global")

        lobby_lines = [
            "mode: global",
            "rank_order: 输入顺序即当前名次，不重新洗牌",
            f"players: {len(players)}",
        ]
        local_names: List[str] = []
        player_reports: List[str] = []
        tags: List[str] = []
        contested_traits: Counter[str] = Counter()
        contested_carries: Counter[str] = Counter()
        local_traits: List[str] = []

        for idx, player in enumerate(players, 1):
            label = player.get("label") or player.get("player_name") or player.get("name") or f"第{idx}名"
            report, prepared = self._run_sub_agents_for_analysis(player, label=label, rank=idx)
            profile = prepared["_profile"]
            refs = [MetaReference(**entry) for entry in prepared.get("_meta_refs", [])]
            tags.extend(self._profile_tags(profile))
            local_names.extend(profile.get("units", []))
            if profile.get("carry"):
                local_names.append(profile["carry"])
            for trait in profile.get("traits", [])[:3]:
                contested_traits[trait] += 1
            local_traits.extend(profile.get("traits", []))
            if profile.get("carry"):
                contested_carries[profile["carry"]] += 1

            lobby_lines.append(
                f"rank {idx}: {label} | size={profile['team_size']} | 2star={profile['two_star']} | 3star={profile['three_star']} | items={profile['item_count']} | carry={profile['carry'] or 'unknown'} {profile['carry_star']}* | traits={', '.join(profile['traits'][:4]) or 'none'} | units={', '.join(profile['units'][:7]) or 'none'}"
            )
            player_reports.append(f"[第{idx}名 {label}]\n{report}\n\n[竞赛参考]\n{self._format_meta_refs(refs)}")

        contested_trait_text = ", ".join(f"{name} x{count}" for name, count in contested_traits.most_common(6) if count >= 2)
        contested_carry_text = ", ".join(f"{name} x{count}" for name, count in contested_carries.most_common(6) if count >= 2)
        if contested_trait_text:
            lobby_lines.append("contested_traits: " + contested_trait_text)
        if contested_carry_text:
            lobby_lines.append("contested_carries: " + contested_carry_text)

        local_champion_desc = self._local_fact_context(local_names, local_traits, limit=24)
        default_question = (
            "请逐个比较所有玩家的阵容成型度、核心棋子质量、装备完成度、与竞赛模板的接近程度，"
            "再综合判断谁最强、谁更像前二、谁存在明显短板。"
        )
        return "\n".join(lobby_lines), "\n\n".join(player_reports), tags[:24], "全局棋盘", default_question, local_champion_desc

    def recommend(self, question: str = "", mode: str = "single") -> str:
        self.local.reload()
        self.power_agent.trait_dict = self.local.trait_dict
        for agent in self.orchestrator.agents:
            if isinstance(agent, PowerAgent):
                agent.trait_dict = self.local.trait_dict

        is_global = mode == "global" and isinstance(self.local.analysis.get("players"), list) and bool(self.local.analysis.get("players"))
        if is_global:
            team_desc, sub_reports, tags, board_label, default_question, local_champion_desc = self._build_global_mode_context()
        else:
            team_desc, sub_reports, tags, board_label, default_question, local_champion_desc = self._build_single_mode_context(mode)

        set_num = self._set_number()
        query_terms = [f"set{set_num}", mode] + tags[:12]
        if question:
            query_terms.append(question)
        hits = self.kb.search(" ".join(term for term in query_terms if term), top_k=CFG["top_k"])
        ctx = self._format_kb_hits(hits)

        if not self.llm.api_key:
            return (
                f"**{board_label}**\n{team_desc}\n\n"
                f"**子 Agent 报告**\n{sub_reports}\n\n"
                f"**知识库检索 ({len(hits)} 条)**\n{ctx}\n\n"
                "_配置 API Key 后可获得大模型综合分析。_"
            )

        user_prompt = (
            f"[模式]\n{mode}\n\n"
            f"[{board_label}]\n{team_desc}\n\n"
            f"[子 Agent 报告]\n{sub_reports}\n\n"
            f"[本地事实]\n{local_champion_desc}\n\n"
            f"[知识库检索参考]\n{ctx}\n\n"
            f"[用户问题]\n{question or default_question}\n\n"
            "请严格依据这些输入做分析。"
        )
        return self.llm.chat(SYSTEM_PROMPT.format(set_num=set_num), user_prompt)

    def run(self):
        set_num = self._set_number()
        provider = self.llm.provider.upper()
        model = self.llm.model
        print(f"\n{'=' * 52}")
        print(f"  TFT 阵容顾问 | Set{set_num}")
        print(f"  LLM: [{provider}] {model}")
        print(f"  知识库: {self.kb.stats()}")
        print(f"{'=' * 52}")
        print("  命令: refresh / status / mode <single|duel|global> / exit")
        print()

        mode = "single"
        while True:
            try:
                user_input = input(f"[{mode}] > ").strip()
                if not user_input:
                    continue
                cmd = user_input.lower()
                if cmd == "exit":
                    print("GL HF!")
                    break
                if cmd == "refresh":
                    self.build_kb(force=True)
                    continue
                if cmd == "status":
                    print(f"知识库: {self.kb.stats()}")
                    print(f"阵容摘要: {self.local.team_summary()[:160]}")
                    continue
                if cmd.startswith("mode "):
                    next_mode = cmd.split(maxsplit=1)[1]
                    if next_mode in {"single", "duel", "global"}:
                        mode = next_mode
                        print(f"已切换模式: {mode}")
                    else:
                        print("模式必须是 single / duel / global")
                    continue

                result = self.recommend(user_input, mode=mode)
                print(f"\n{'-' * 52}")
                print(result)
                print(f"{'-' * 52}\n")
            except KeyboardInterrupt:
                print("\nGL HF!")
                break
            except Exception as e:
                logger.error(f"处理出错: {e}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="TFT 阵容顾问 RAG Agent")
    ap.add_argument("--question", "-q", type=str, default="", help="直接提问，非交互模式")
    ap.add_argument("--mode", choices=["single", "duel", "global"], default="single", help="分析模式")
    ap.add_argument("--refresh", action="store_true", help="强制刷新知识库")
    args = ap.parse_args()

    agent = TFTRagAgent()
    agent.build_kb(force=args.refresh)

    if args.question:
        print(agent.recommend(args.question, mode=args.mode))
    else:
        agent.run()


if __name__ == "__main__":
    main()









