#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_rag_agent.py
TFT 阵容顾问 — RAG Agent 核心模块

架构:
  - LLMClient        : LLM 调用（Anthropic / OpenRouter）
  - TFTCrawler       : Riot API 高端局数据采集（仅使用 tft-* 系列 API）
  - JSONKnowledgeBase: BM25 知识库（无外部依赖）
  - LocalDataLoader  : 本地阵容数据加载
  - 三个子 Agent     : EconomyAgent / PowerAgent / PositionAgent
  - TFTRagAgent      : 主协调器，整合所有子 Agent 输出

用法:
  python tft_rag_agent.py                       # 交互模式
  python tft_rag_agent.py --question "如何过渡"  # 单次提问
"""

import json, os, re, time, math, hashlib, logging, sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter

import requests

# ──────────────────────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TFT-RAG")

# ──────────────────────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────────────────────
CFG: Dict[str, Any] = {
    # 赛季
    "current_set"           : 16,
    # 文件路径
    "data_dir"              : "./tft_rag_data",
    "analysis_file"         : "tft_team_analysis.json",
    "champion_db_file"      : "tft_champion_db.json",
    "trait_db_file"         : "tft_trait_db.json",
    "item_db_file"          : "tft_item_db.json",
    "trait_dict_file"       : "tft_trait_champion_dict.json",
    # LLM
    "llm_provider"          : os.getenv("LLM_PROVIDER", "openrouter"),
    "anthropic_api_key"     : os.getenv("ANTHROPIC_API_KEY", ""),
    "openrouter_api_key"    : os.getenv("OPENROUTER_API_KEY", "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"),
    "claude_model"          : "claude-opus-4-6",
    "openrouter_model"      : os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
    "max_tokens"            : 2048,
    # Riot API（仅 TFT 系列接口）
    "riot_api_key"          : os.getenv("RIOT_API_KEY", "RGAPI-a365951a-ce8f-4b19-836b-8af4b0e133b7"),
    "riot_region_platform"  : "kr",          # kr.api.riotgames.com
    "riot_region_regional"  : "asia",        # asia.api.riotgames.com
    "riot_tiers"            : ("challenger", "grandmaster"),
    # ── 性能关键参数 ──────────────────────────────────────────────
    # 每个榜单最多取多少名玩家（原30，缩减至10）
    # 10名 × 每人10场 × 8人/场 = 800条记录，足够生成20+个阵容文档
    "riot_max_players"      : 10,
    # 每名玩家最多抓多少场对局（原20，缩减至10）
    # Development Key 限速 ~85 req/min，10场≈10次API调用
    "riot_matches_per_player": 10,
    # 缓存有效期（小时）。爬虫耗时长，12h内复用缓存
    "cache_ttl_hours"       : 12,
    # ─────────────────────────────────────────────────────────────
    # RAG
    "top_k"                 : 6,
    "chunk_size"            : 600,
    "chunk_overlap"         : 80,
    # 后台爬虫：是否在独立线程异步爬取，不阻塞用户交互
    # True  = 启动时立即可用（用旧缓存），后台静默刷新
    # False = 旧行为：阻塞等待爬虫完成
    "background_crawl"      : True,
}

DATA_DIR = Path(CFG["data_dir"])
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TFT-Advisor/2.0",
}


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# BM25 知识库
# ══════════════════════════════════════════════════════════════
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
                logger.info(f"知识库加载：{len(self.chunks)} 块")
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
        logger.info(f"知识库新增 {len(new_chunks)} 块，共 {len(self.chunks)} 块")

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
        return f"{len(self.chunks)} 块 [{detail}]"


# ══════════════════════════════════════════════════════════════
# Riot API 爬虫（仅 TFT 接口）
# ══════════════════════════════════════════════════════════════
class TFTCrawler:
    """
    使用以下 TFT 专属 API（不涉及任何 LoL 接口）：
      tft-league-v1    → 高端局排行榜
      tft-summoner-v1  → summonerId → puuid
      tft-match-v1     → 对局 ID 列表 + 对局详情
    """
    CACHE_FILE = DATA_DIR / "riot_cache.json"
    # Development Key 限速：20 req/s (短期) / 100 req/2min (长期)
    # 0.5s 间隔 ≈ 2 req/s，远低于限制，同时比原来的 0.7s 快 30%
    RATE_DELAY = 0.5

    def __init__(self):
        self.api_key = CFG.get("riot_api_key", "")
        self._cache: Dict = self._load_cache()

    # ── 缓存 ─────────────────────────────────────────────────
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

    # ── HTTP 请求（含限速重试）────────────────────────────────
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
                    logger.warning(f"Riot API 限速，等待 {wait}s")
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
                logger.warning(f"请求失败: {e}")
                if attempt < 2:
                    time.sleep(2)
        return None

    # ── Step 1：排行榜 → puuid 列表（跳过 summonerId 中转）────
    def _get_top_players(self) -> List[str]:
        """
        从 Challenger / Grandmaster / Master 排行榜直接读取 puuid。

        2023 年 Riot ID 系统更新后，tft-league-v1 的 entries[] 每条记录
        已直接包含 puuid 字段，无需再经过 summonerId → puuid 的额外转换。

        字段优先级：puuid（优先）→ summonerId（兜底，用于后续 _get_puuid 查询）

        Development Key 权限说明：
          ✅ /tft/league/v1/challenger   — 允许，entries[] 含 puuid
          ✅ /tft/league/v1/grandmaster  — 允许
          ✅ /tft/league/v1/master       — 允许
          ❌ /tft/league/v1/entries/{tier}/{division} — Dev Key 封锁（403）
        """
        region = CFG["riot_region_platform"]
        base   = f"https://{region}.api.riotgames.com"
        puuids: List[str] = []
        sids:   List[str] = []      # 兜底：没有直接 puuid 时存 summonerId
        limit  = CFG["riot_max_players"]

        tiers = list(CFG.get("riot_tiers", ("challenger", "grandmaster")))
        if "master" not in tiers:
            tiers.append("master")

        for tier in tiers:
            url  = f"{base}/tft/league/v1/{tier}"
            data = self._get(url)
            if not data:
                logger.warning(f"  {tier} 榜单获取失败（HTTP 错误或网络超时）")
                continue

            entries = data.get("entries", [])
            if not entries:
                logger.warning(f"  {tier} 榜单 entries 为空")
                continue

            entries.sort(key=lambda x: x.get("leaguePoints", 0), reverse=True)

            taken = 0
            for e in entries[:limit]:
                # 优先直接取 puuid（2023+ API 响应已包含）
                puuid = e.get("puuid")
                if puuid and puuid not in puuids:
                    puuids.append(puuid)
                    taken += 1
                else:
                    # 兜底：记录 summonerId，稍后通过 _get_puuid 转换
                    sid = e.get("summonerId")
                    if sid and sid not in sids:
                        sids.append(sid)
                        taken += 1

            logger.info(f"  {tier}: {len(entries)} 名玩家，取 {taken} 名"
                        f"（直接puuid:{sum(1 for e in entries[:limit] if e.get('puuid'))}）")

        if not puuids and not sids:
            logger.warning(
                "未能从排行榜获取任何玩家 ID。\n"
                "  • Development Key 每 24h 过期，请到 https://developer.riotgames.com 重新生成\n"
                "  • 检查 riot_region_platform 配置（当前: "
                + str(CFG.get("riot_region_platform")) + "）\n"
                "  • 关闭代理后重试"
            )

        # 把 summonerId 兜底列表转换为 puuid，然后合并
        # 这里用特殊标记区分：以 "SID:" 前缀存储，crawl() 中会识别并调用 _get_puuid
        result = puuids + [f"SID:{sid}" for sid in sids]
        return result

    # ── Step 2：summonerId → puuid ─────────────────────────────
    def _get_puuid(self, summoner_id: str) -> Optional[str]:
        region = CFG["riot_region_platform"]
        url  = f"https://{region}.api.riotgames.com/tft/summoner/v1/summoners/{summoner_id}"
        data = self._get(url)
        return data.get("puuid") if data else None

    # ── Step 3：puuid → 对局 ID 列表 ──────────────────────────
    def _get_match_ids(self, puuid: str) -> List[str]:
        region = CFG["riot_region_regional"]
        url    = f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids"
        count  = CFG["riot_matches_per_player"]
        data   = self._get(url, params={"count": count, "type": "ranked"})
        return data if isinstance(data, list) else []

    # ── Step 4：match_id → 对局详情 ───────────────────────────
    def _get_match(self, match_id: str) -> Optional[Dict]:
        region = CFG["riot_region_regional"]
        url    = f"https://{region}.api.riotgames.com/tft/match/v1/matches/{match_id}"
        return self._get(url)

    # ── 解析单名参与者 ─────────────────────────────────────────
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
                traits.append({
                    "name" : strip(t.get("name", "")),
                    "count": t.get("num_units", 0),
                    "tier" : t.get("tier_current", 0),
                })
        traits.sort(key=lambda x: (-x["tier"], -x["count"]))

        units = [
            {
                "id"   : strip(u.get("character_id", "")),
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

    # ── 聚合统计 → Doc 列表 ────────────────────────────────────
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
            top4   = sum(1 for r in records if r["top4"]) / n
            wins   = sum(1 for r in records if r["win"])  / n

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
                f"阵容: {comp_key} | 样本: {n} | "
                f"均名: {avg_pl:.2f} | Top4: {top4:.0%} | 吃鸡: {wins:.0%}\n"
                f"核心英雄: {', '.join(core)}\n"
                f"常见奥金: {', '.join(top_augs) or '无'}"
            )
            doc_id = hashlib.md5(comp_key.encode()).hexdigest()[:10]
            docs.append(Doc(
                doc_id    =f"riot_{doc_id}",
                source    ="riot_api",
                title     =f"[KR高端局] {comp_key}",
                content   =content,
                url       =f"https://developer.riotgames.com/apis#tft-match-v1",
                fetched_at=datetime.now().isoformat(),
                tags      =comp_key.split(" + ") + core[:3],
            ))
        return docs

    # ── 主入口 ─────────────────────────────────────────────────
    def crawl(self) -> List[Doc]:
        if not self.api_key:
            logger.warning("RIOT_API_KEY 未设置，跳过 Riot API 采集")
            return []

        cache_key = f"riot_kb_{CFG['current_set']}"
        if self._cache_valid(cache_key):
            logger.info("Riot API 缓存有效，直接读取")
            cached_docs = self._cache[cache_key].get("docs", [])
            return [Doc(**d) for d in cached_docs]

        logger.info("=== Riot API 采集开始 ===")

        # Step 1: 排行榜 → puuid 或 summonerId 标记
        logger.info("Step 1: 获取高端局排行榜")
        player_tokens = self._get_top_players()
        if not player_tokens:
            logger.warning(
                "未获取到任何玩家 ID，Riot API 采集终止。\n"
                "  常见原因：\n"
                "  1. Development Key 每 24h 过期，请到 https://developer.riotgames.com 重新生成\n"
                "  2. 当前 riot_tiers 配置为: " + str(CFG.get("riot_tiers")) + "\n"
                "  3. 网络问题（VPN / 防火墙）"
            )
            return []

        # Step 2: 解析 puuid（兼容新旧两种格式）
        # _get_top_players 返回两类值：
        #   直接 puuid（78字符，2023+ API 新格式）→ 直接使用
        #   "SID:{summonerId}"（旧格式兜底）→ 调用 tft-summoner-v1 转换
        logger.info(f"Step 2: 解析 {len(player_tokens)} 个玩家标识符为 PUUID")
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
                logger.info(f"  PUUID 进度: {i+1}/{len(player_tokens)}")

        direct = len(player_tokens) - sid_count
        logger.info(f"  直接获取 puuid: {direct} 个，via summonerId 转换: {sid_count} 个，"
                    f"成功: {len(puuids)} 个")

        # Step 3 & 4: 对局 ID → 对局详情
        logger.info(f"Step 3: 采集 {len(puuids)} 名玩家的对局")
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
                logger.info(f"  玩家 {i+1}/{len(puuids)}, "
                            f"对局 {len(seen_matches)}, 记录 {len(all_participants)}")

        logger.info(f"采集完成: {len(seen_matches)} 场 / {len(all_participants)} 记录")

        # Step 5: 聚合
        docs = self._aggregate(all_participants)
        logger.info(f"生成 {len(docs)} 个阵容文档")

        # 写缓存
        self._cache[cache_key] = {
            "ts"  : datetime.now().isoformat(),
            "docs": [asdict(d) for d in docs],
        }
        self._save_cache()
        return docs


# ══════════════════════════════════════════════════════════════
# 本地数据加载器
# ══════════════════════════════════════════════════════════════
class LocalDataLoader:
    def __init__(self):
        self.analysis: Dict = {}
        self.champion_db: Dict = {}
        self.trait_db: Dict = {}
        self.item_db: Dict = {}
        self.trait_dict: Dict = {}
        self._load_all()

    def _load_all(self):
        # 阵容分析
        for path in [CFG["analysis_file"], "tft_team_analysis.json"]:
            if Path(path).exists():
                try:
                    self.analysis = json.loads(Path(path).read_text(encoding="utf-8"))
                    logger.info(f"阵容分析已加载: {path}")
                    break
                except Exception as e:
                    logger.warning(f"加载 {path} 失败: {e}")

        # 英雄/羁绊/装备 DB
        for attr, key in [("champion_db", "champion_db_file"),
                          ("trait_db", "trait_db_file"),
                          ("item_db", "item_db_file"),
                          ("trait_dict", "trait_dict_file")]:
            p = Path(CFG[key])
            if p.exists():
                try:
                    setattr(self, attr, json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    pass

    def reload(self):
        self._load_all()

    def to_docs(self) -> List[Doc]:
        docs = []

        # 当前阵容 Doc
        if self.analysis:
            docs.append(Doc(
                doc_id    ="local_analysis",
                source    ="local",
                title     ="当前阵容分析",
                content   =self._fmt_analysis(),
                url       ="local://tft_team_analysis.json",
                fetched_at=datetime.now().isoformat(),
                tags      =self.analysis_tags(),
            ))

        # 羁绊词典 Doc
        if self.trait_dict:
            lines = []
            for tname, tdata in self.trait_dict.items():
                if not isinstance(tdata, dict):
                    continue
                champs = ", ".join(tdata.get("champions", []))
                levels = tdata.get("activation", {}).get("levels", [])
                lines.append(f"[{tname}] 英雄: {champs} | 激活阈值: {levels}")
            docs.append(Doc(
                doc_id    ="local_traits",
                source    ="local",
                title     ="羁绊数据库",
                content   ="\n".join(lines[:80]),
                url       ="local://tft_trait_champion_dict.json",
                fetched_at=datetime.now().isoformat(),
                tags      =["羁绊", "trait", "激活"],
            ))

        return docs

    def _fmt_analysis(self) -> str:
        a = self.analysis
        parts = [f"阵容规模: {a.get('team_size', '?')} 人"]
        champs = a.get("champions", [])
        if champs:
            cstrs = []
            for c in champs:
                name = c.get("name_en") or c.get("name", "?")
                star = c.get("star", 1)
                items = ", ".join(c.get("items", [])) or "无装备"
                cstrs.append(f"{name} {star}★({items})")
            parts.append("英雄: " + " / ".join(cstrs))
        traits = a.get("traits", [])
        if traits:
            tstrs = []
            for t in traits:
                name  = t.get("name_en") or t.get("name", "?")
                count = t.get("count", 0)
                lvl   = t.get("level_name") or t.get("level", "")
                tstrs.append(f"{name}({count}人{'/'+str(lvl) if lvl else ''})")
            parts.append("激活羁绊: " + ", ".join(tstrs))
        s = a.get("summary", {})
        if s.get("front_row_ratio"):
            parts.append(f"前排比例: {s['front_row_ratio']}")
        if a.get("equipment_issues"):
            parts.append("装备问题: " + "; ".join(a["equipment_issues"]))
        return "\n".join(parts)

    def analysis_tags(self) -> List[str]:
        tags: List[str] = []
        for t in self.analysis.get("traits", []):
            tags.append(t.get("name_en") or t.get("name", ""))
        for c in self.analysis.get("champions", []):
            tags.append(c.get("name_en") or c.get("name", ""))
        return [t for t in tags if t]

    def team_summary(self) -> str:
        return self._fmt_analysis() if self.analysis else "（未检测到阵容数据）"


# ══════════════════════════════════════════════════════════════
# 多智能体：三个子 Agent
# ══════════════════════════════════════════════════════════════
class EconomyAgent:
    """经济 Agent：分析金币运营、升级节奏、连胜/连败策略"""

    def analyze(self, analysis: Dict) -> str:
        team_size = analysis.get("team_size", 0)
        parts: List[str] = []

        # 推断当前回合阶段
        if team_size <= 5:
            phase = "早期(1-3阶段)"
            advice = "保持连胜/连败；利息优先，不要轻易打破 50 金币的利息阈值"
        elif team_size <= 7:
            phase = "中期(3-4阶段)"
            advice = "决定是否d牌；若阵容成型可开始升 8 级"
        else:
            phase = "后期(5-6阶段)"
            advice = "全力保血量，适时d牌寻找主C主坦"

        parts.append(f"阶段判断: {phase}")
        parts.append(f"经济建议: {advice}")

        champs = analysis.get("champions", [])
        stars  = [c.get("star", 1) for c in champs]
        if stars:
            avg_star = sum(stars) / len(stars)
            if avg_star < 1.5:
                parts.append("英雄星级偏低，建议继续攒金币或d牌强化")
            elif avg_star >= 2.2:
                parts.append("英雄星级良好，可考虑升级扩大阵容")

        return "\n".join(parts)


class PowerAgent:
    """战力 Agent：分析羁绊激活、装备分配、英雄搭配"""

    def __init__(self, trait_dict: Dict):
        self.trait_dict = trait_dict

    def analyze(self, analysis: Dict) -> str:
        parts: List[str] = []
        traits = analysis.get("traits", [])
        champs = analysis.get("champions", [])

        # 激活羁绊评价
        if traits:
            active = [t for t in traits if t.get("count", 0) > 0]
            key_traits = [t.get("name_en") or t.get("name", "") for t in active[:3]]
            parts.append(f"核心羁绊: {', '.join(key_traits) or '未激活'}")

            # 检测是否接近下一级激活
            suggestions = []
            for t in active:
                tname  = t.get("name_en") or t.get("name", "")
                count  = t.get("count", 0)
                td     = self.trait_dict.get(tname, {})
                levels = td.get("activation", {}).get("levels", [])
                for lvl in levels:
                    if count < lvl <= count + 2:
                        suggestions.append(f"再添 {lvl - count} 个 [{tname}] 可升阶")
                        break
            if suggestions:
                parts.append("羁绊升阶提示: " + " | ".join(suggestions[:3]))
        else:
            parts.append("暂无激活羁绊，建议整理阵容方向")

        # 装备评价
        all_items: List[str] = []
        no_item_carries: List[str] = []
        for c in champs:
            items = c.get("items", [])
            all_items.extend(items)
            if not items and c.get("cost", 1) >= 4:
                no_item_carries.append(c.get("name_en") or c.get("name", ""))

        if no_item_carries:
            parts.append(f"⚠ 高费英雄缺装备: {', '.join(no_item_carries)}")
        if len(all_items) > 0:
            parts.append(f"当前已装备: {len(all_items)} 件 / {len(champs)} 名英雄")

        return "\n".join(parts)


class PositionAgent:
    """站位 Agent：分析棋盘布阵、前后排比例"""

    def analyze(self, analysis: Dict) -> str:
        parts: List[str] = []
        champs = analysis.get("champions", [])
        summary = analysis.get("summary", {})

        # 前后排比例
        front_ratio = summary.get("front_row_ratio", "")
        if front_ratio:
            parts.append(f"前后排比例: {front_ratio}")

        # 通过位置分析
        positions = [c.get("position", {}) for c in champs if c.get("position")]
        if positions:
            rows = [p.get("row", 0) for p in positions]
            if rows:
                front_row = sum(1 for r in rows if r >= 3)  # row 3+ 为前排
                back_row  = len(rows) - front_row
                parts.append(f"前排: {front_row} / 后排: {back_row}")
                if front_row < 2:
                    parts.append("⚠ 前排偏少，容易被穿透")
                elif back_row < 2:
                    parts.append("⚠ 后排输出位偏少")

        # 主C判断
        main_carry = summary.get("main_carry", "")
        if main_carry:
            parts.append(f"主C: {main_carry}")

        equipment_issues = analysis.get("equipment_issues", [])
        if equipment_issues:
            parts.append("站位/装备问题: " + "; ".join(equipment_issues[:3]))

        return "\n".join(parts) if parts else "站位数据不足"


# ══════════════════════════════════════════════════════════════
# LLM 客户端
# ══════════════════════════════════════════════════════════════
class LLMClient:
    ANTHROPIC_URL  = "https://api.anthropic.com/v1/messages"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.provider = CFG["llm_provider"]
        self.api_key  = self._resolve_key()

    def _resolve_key(self) -> str:
        if self.provider == "anthropic":
            key = CFG["anthropic_api_key"]
        else:
            key = CFG["openrouter_api_key"]
        if key:
            return key
        # Web 模式（非 TTY）静默返回空
        if not sys.stdin.isatty():
            logger.warning(f"LLM API Key 未设置（provider={self.provider}）")
            return ""
        # CLI 交互模式
        env_var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENROUTER_API_KEY"
        hint    = "sk-ant-..." if self.provider == "anthropic" else "sk-or-v1-..."
        url     = "https://console.anthropic.com/" if self.provider == "anthropic" else "https://openrouter.ai/keys"
        print(f"\n⚠  未设置 {env_var}")
        print(f"   获取: {url}")
        choice = input("输入 Key（1）或 跳过（2）: ").strip()
        if choice == "1":
            import getpass
            key = getpass.getpass(f"{env_var}: ").strip()
            if key:
                os.environ[env_var] = key
                if self.provider == "anthropic":
                    CFG["anthropic_api_key"] = key
                else:
                    CFG["openrouter_api_key"] = key
                return key
        return ""

    @property
    def model(self) -> str:
        return CFG["claude_model"] if self.provider == "anthropic" else CFG["openrouter_model"]

    def chat(self, system: str, user: str) -> str:
        if not self.api_key:
            return "ℹ  LLM 未配置，请在 Settings 中填写 API Key。"
        return (self._call_anthropic if self.provider == "anthropic" else self._call_openrouter)(system, user)

    def _call_anthropic(self, system: str, user: str) -> str:
        try:
            r = requests.post(
                self.ANTHROPIC_URL,
                headers={"x-api-key": self.api_key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": self.model, "max_tokens": CFG["max_tokens"],
                      "system": system, "messages": [{"role": "user", "content": user}]},
                timeout=90,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except requests.HTTPError as e:
            return self._http_err(e)
        except Exception as e:
            return f"❌ 请求失败: {e}"

    def _call_openrouter(self, system: str, user: str) -> str:
        try:
            r = requests.post(
                self.OPENROUTER_URL,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/tft-advisor",
                         "X-Title": "TFT Advisor"},
                json={"model": self.model, "max_tokens": CFG["max_tokens"],
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user",   "content": user}]},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                return f"❌ OpenRouter: {data['error'].get('message', data['error'])}"
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            return self._http_err(e)
        except Exception as e:
            return f"❌ 请求失败: {e}"

    @staticmethod
    def _http_err(e: requests.HTTPError) -> str:
        code = e.response.status_code
        return {
            401: "❌ API Key 无效，请检查",
            402: "❌ 余额不足，请充值或换用免费模型",
            429: "❌ 请求频率超限，请稍后再试",
        }.get(code, f"❌ HTTP {code}: {e.response.text[:200]}")


# ══════════════════════════════════════════════════════════════
# 主 RAG Agent（协调器）
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是专业的云顶之弈（TFT）战术顾问，精通 Set{set_num} 所有机制与当前 Meta。
根据玩家的阵容现状、子 Agent 分析报告和高端局参考数据，给出清晰、可执行的优化建议。

请按以下格式作答：
## ⚡ 阵容评价
2~3 句核心评价（优势/劣势）

## 🗺️ 发展路线（2~3 条）
每条包含：阵容名 (Tier) / 核心英雄 / 关键羁绊 / 装备优先级 / 与当前阵容的距离

## 💰 经济节奏
连胜/连败策略 / 升级时机 / 滚轮盘时机

## ⚔️ 装备与站位
主C装备优先级 / 站位注意事项

规则：用中文回答；简洁，每条建议可操作；引用参考数据时注明来源。
"""


class TFTRagAgent:
    def __init__(self):
        logger.info("🚀 初始化 TFT RAG Agent")
        self.local   = LocalDataLoader()
        self.kb      = JSONKnowledgeBase()
        self.crawler = TFTCrawler()
        self.llm     = LLMClient()

        # 子 Agent
        self.economy_agent  = EconomyAgent()
        self.power_agent    = PowerAgent(self.local.trait_dict)
        self.position_agent = PositionAgent()

        # 同步 riot key
        if not self.crawler.api_key:
            key = CFG.get("riot_api_key") or os.getenv("RIOT_API_KEY", "")
            if key:
                self.crawler.api_key = key

    # ── 知识库构建 ────────────────────────────────────────────
    def build_kb(self, force: bool = False, background: bool = None):
        """
        构建/刷新知识库。

        background=True（默认）：
          - 若已有缓存 → 立即加载缓存，后台线程静默刷新（不阻塞用户）
          - 若无缓存   → 同步等待首次爬取（仅第一次运行时阻塞）
        background=False：
          - 始终同步阻塞，等待爬虫完成后再返回
        force=True：
          - 清除所有缓存，强制重新爬取
        """
        if background is None:
            background = CFG.get("background_crawl", True)

        if force:
            self.crawler.clear_cache()
            self.kb.clear()

        # 检查是否已有可用知识库（缓存 chunks）
        has_kb = len(self.kb.chunks) > 0

        # 检查 Riot 缓存是否有效（即使 kb chunks 为空，Riot 缓存可能存在）
        cache_key = f"riot_kb_{CFG['current_set']}"
        riot_cache_valid = self.crawler._cache_valid(cache_key)

        if riot_cache_valid and not has_kb:
            # Riot 缓存有效但 KB chunks 未加载（首次启动），立即同步加载缓存
            logger.info("从 Riot 缓存快速加载知识库...")
            cached_docs = self.crawler._cache[cache_key].get("docs", [])
            riot_docs   = [Doc(**d) for d in cached_docs]
            local_docs  = self.local.to_docs()
            self.kb.add_docs(riot_docs + local_docs)
            self._print_kb_stats(riot_docs, local_docs)
            return  # 缓存有效，无需重新爬取

        if not has_kb and not riot_cache_valid:
            # 无任何缓存 → 必须首次同步爬取（仅此一次阻塞）
            print("\n" + "─" * 50)
            print("📚 首次构建知识库（仅需等待一次，之后12h内直接复用缓存）")
            print(f"   预计耗时：{CFG['riot_max_players']} 名玩家 × "
                  f"{CFG['riot_matches_per_player']} 场 ≈ 5~10 分钟")
            print("─" * 50)
            self._do_crawl_and_build()
            return

        # 已有 KB → 立即返回，后台刷新（若缓存即将过期）
        if background and riot_cache_valid:
            # 缓存仍有效，无需刷新
            logger.info(f"知识库就绪（{self.kb.stats()}），缓存有效，跳过刷新")
            local_docs = self.local.to_docs()
            self.kb.add_docs(local_docs)   # 仅刷新本地阵容数据（无延迟）
            return

        if background:
            # 缓存过期，后台线程刷新，不阻塞用户
            import threading
            logger.info(f"知识库就绪（{self.kb.stats()}），后台刷新中...")
            t = threading.Thread(target=self._do_crawl_and_build, daemon=True)
            t.start()
        else:
            self._do_crawl_and_build()

    def _do_crawl_and_build(self):
        """实际执行爬取 + 知识库构建（可在后台线程中调用）"""
        riot_docs  = self.crawler.crawl()
        local_docs = self.local.to_docs()
        if riot_docs or local_docs:
            self.kb.add_docs(riot_docs + local_docs)
        self._print_kb_stats(riot_docs, local_docs)

    def _print_kb_stats(self, riot_docs: list, local_docs: list):
        from collections import Counter as _Counter
        src_cnt = _Counter(d.source for d in riot_docs + local_docs)
        print(f"  Riot API  : {src_cnt.get('riot_api', 0)} 个阵容文档")
        print(f"  本地数据  : {src_cnt.get('local', 0)} 个文档")
        print(f"  知识库总计: {self.kb.stats()}")

    # ── 多 Agent 协同分析 ──────────────────────────────────────
    def _run_sub_agents(self) -> str:
        a = self.local.analysis
        if not a:
            return "（未检测到阵容数据）"

        economy  = self.economy_agent.analyze(a)
        power    = self.power_agent.analyze(a)
        position = self.position_agent.analyze(a)

        return (
            f"[经济Agent]\n{economy}\n\n"
            f"[战力Agent]\n{power}\n\n"
            f"[站位Agent]\n{position}"
        )

    # ── RAG 推荐 ──────────────────────────────────────────────
    def recommend(self, question: str = "", mode: str = "single") -> str:
        """
        mode: single（单人）/ duel（对局）/ global（全局）

        每次调用时会重新加载本地阵容数据（tft_team_analysis.json），
        确保截图识别后的最新阵容能立即反映在分析中，无需重启。
        """
        # 每次推荐前重新加载最新阵容（截图后文件可能已更新）
        self.local.reload()
        # 同步更新 PowerAgent 的 trait_dict（trait_dict 可能随 reload 变化）
        self.power_agent.trait_dict = self.local.trait_dict
        tags  = self.local.analysis_tags()[:6]
        if question:
            tags.append(question)
        query = " ".join(tags) or f"TFT Set{CFG['current_set']} meta"

        hits = self.kb.search(query, top_k=CFG["top_k"])
        ctx  = "\n\n".join(
            f"[参考{i} | {h.get('source','?')}] {h.get('title','')}\n{h.get('text','')}"
            for i, h in enumerate(hits, 1)
        ) or "（暂无外部参考）"

        sub_reports = self._run_sub_agents()
        team_desc   = self.local.team_summary()

        # 无 LLM Key → 返回原始分析
        if not self.llm.api_key:
            return (
                f"**阵容概况**\n{team_desc}\n\n"
                f"**子 Agent 分析**\n{sub_reports}\n\n"
                f"**知识库检索 ({len(hits)} 条)**\n{ctx}\n\n"
                "_配置 API Key 后可获得 AI 深度分析。_"
            )

        mode_hint = {
            "duel"  : "\n（当前为对局模式：请重点分析双方对位和克制关系）",
            "global": "\n（当前为全局模式：请结合场上多家阵容评估优先级）",
        }.get(mode, "")

        system = SYSTEM_PROMPT.format(set_num=CFG["current_set"]) + mode_hint
        user   = (
            f"【当前阵容】\n{team_desc}\n\n"
            f"【子 Agent 分析报告】\n{sub_reports}\n\n"
            f"【高端局参考（来源: {', '.join({h.get('source','?') for h in hits})}）】\n{ctx}\n\n"
            f"【问题】{question or '请给出完整阵容分析与最优发展路线'}\n\n"
            "请给出专业建议："
        )
        return self.llm.chat(system, user)

    # ── 交互 CLI ──────────────────────────────────────────────
    def run(self):
        SET   = CFG["current_set"]
        PROV  = self.llm.provider.upper()
        MODEL = self.llm.model
        print(f"\n{'='*52}")
        print(f"  🎮 TFT 阵容顾问  |  Set{SET}")
        print(f"  LLM: [{PROV}] {MODEL}")
        print(f"  知识库: {self.kb.stats()}")
        print(f"{'='*52}")
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
                    print("GL HF! 🎮"); break
                elif cmd == "refresh":
                    self.build_kb(force=True)
                elif cmd == "status":
                    print(f"知识库: {self.kb.stats()}")
                    print(f"阵容: {self.local.team_summary()[:120]}")
                elif cmd.startswith("mode "):
                    m = cmd.split(maxsplit=1)[1]
                    if m in ("single", "duel", "global"):
                        mode = m
                        print(f"切换模式: {mode}")
                    else:
                        print("模式: single / duel / global")
                else:
                    result = self.recommend(user_input, mode=mode)
                    print(f"\n{'─'*52}")
                    print(f"🤖 TFT顾问:\n{result}")
                    print(f"{'─'*52}\n")
            except KeyboardInterrupt:
                print("\nGL HF! 🎮"); break
            except Exception as e:
                logger.error(f"处理出错: {e}")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="TFT 阵容顾问 RAG Agent")
    ap.add_argument("--question", "-q", type=str, default="",
                    help="直接提问（非交互模式）")
    ap.add_argument("--mode", choices=["single","duel","global"],
                    default="single", help="分析模式")
    ap.add_argument("--refresh", action="store_true",
                    help="强制刷新知识库")
    args = ap.parse_args()

    agent = TFTRagAgent()
    agent.build_kb(force=args.refresh)

    if args.question:
        result = agent.recommend(args.question, mode=args.mode)
        print(result)
    else:
        agent.run()


if __name__ == "__main__":
    main()
