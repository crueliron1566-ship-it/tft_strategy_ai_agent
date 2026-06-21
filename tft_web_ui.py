#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_web_ui.py — TFT 阵容顾问 Web 界面 v3
运行: python tft_web_ui.py  |  访问: http://localhost:5000

新增:
  - 阵容生成标签页: 英雄下拉搜索、棋盘预览（含英雄图片）、装备弹窗选取
  - 简化聊天栏: 模式选择 Pills 内嵌在输入框上方
  - /api/data/champions  /api/data/items  数据接口
  - /api/input/builder   棋盘提交接口
"""

import json, os, re, sys, threading, traceback, mimetypes
from pathlib import Path
from difflib import SequenceMatcher

from flask import Flask, request, jsonify, send_file, abort
sys.path.insert(0, str(Path(__file__).parent))
import tft_rag_agent as rag

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ── Global state ───────────────────────────────────────────────
_agent: rag.TFTRagAgent = None
_agent_lock = threading.Lock()
_kb_status  = {"ready": False, "building": False, "stats": "not initialized"}

DDRAGON_VER  = "15.8.1"
# 本地资产目录（tft_assets/champions 和 tft_assets/items）
ASSETS_DIR   = Path("./tft_assets")
CHAMP_ASSETS = ASSETS_DIR / "champions"
ITEM_ASSETS  = ASSETS_DIR / "items"

def get_agent() -> rag.TFTRagAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = rag.TFTRagAgent()
    return _agent


def _build_kb_bg(force: bool = False):
    global _kb_status
    _kb_status["building"] = True
    try:
        agent = get_agent()
        agent.build_kb(force=force)
        _kb_status.update({"ready": True, "building": False, "stats": agent.kb.stats()})
    except Exception as e:
        _kb_status.update({"building": False, "error": str(e)})


threading.Thread(target=_build_kb_bg, daemon=True).start()


# ── DB helpers ─────────────────────────────────────────────────────────────
def _repair_json_text(text: str) -> str:
    fixed_lines = []
    for line in text.splitlines():
        if line.count('"') % 2 == 1:
            if re.match(r'^\s*"[^"]+:\s*\{\s*$', line):
                line = re.sub(r'^(\s*"[^"]+):', r'\1":', line)
            elif re.match(r'^\s*"[^"]+"\s*:\s*"[^"]*,\s*$', line):
                line = re.sub(r',\s*$', '",', line)
            elif re.match(r'^\s*"[^"]+"\s*:\s*"[^"]*\s*$', line):
                line = line + '"'
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _load_db(cfg_key: str, *fallbacks) -> dict:
    paths = [rag.CFG.get(cfg_key, ""), *fallbacks]
    for p in paths:
        if p and Path(p).exists():
            text = Path(p).read_text(encoding="utf-8", errors="replace")
            for candidate in (text, _repair_json_text(text)):
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
    return {}


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
    seen = []
    for item in candidates:
        if item and item not in seen:
            seen.append(item)
    return max(seen, key=_text_quality_score)


def _clean_match_text(text: str) -> str:
    text = _fix_mojibake_text(text)
    return re.sub(r'[?�\ufffd]+$', '', text).strip()


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _champion_names(info: dict, api_name: str, set_n: int) -> tuple[str, str, str]:
    short_id = (info.get("short_id") or api_name.replace(f"TFT{set_n}_", "").replace("TFT_", "") or api_name).strip()
    raw_cn = info.get("name_cn") or info.get("name_zh") or ""
    raw_name = info.get("name_en") or ""
    name_cn = _clean_match_text(raw_cn or raw_name)
    if not _has_cjk(name_cn):
        name_cn = ""
    name_en = short_id or _fix_mojibake_text(raw_name) or api_name
    return short_id, name_en.strip(), name_cn


def _item_names(info: dict, api_name: str) -> tuple[str, str]:
    raw_cn = info.get("name_cn") or info.get("name_zh") or ""
    raw_name = info.get("name_en") or api_name
    name_cn = _clean_match_text(raw_cn or raw_name)
    if not _has_cjk(name_cn):
        name_cn = ""
    name_en = (_fix_mojibake_text(raw_name) or api_name).strip()
    if name_cn and name_en == name_cn:
        name_en = api_name
    return name_en, name_cn

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

    db = _load_db("champion_db_file", "tft_champion_db.json", "./tft_rag_data/tft_champion_db.json")
    counts = {}
    for api_name in db.keys():
        if not api_name.startswith("TFT") or "_" not in api_name:
            continue
        prefix = api_name.split("_", 1)[0]
        try:
            set_number = int(prefix[3:])
        except ValueError:
            continue
        counts[set_number] = counts.get(set_number, 0) + 1
    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]

    return int(rag.CFG.get("current_set", default) or default)


def _champ_img(api_name: str) -> str:
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VER}/img/tft-champion/{api_name}.png"


def _item_img(api_name: str) -> str:
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VER}/img/tft-item/{api_name}.png"


def _normalize_lookup_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\s\-_'/·.]+", "", text)
    text = re.sub(r"[()（）\[\]{}]", "", text)
    return text


def _current_set_only(api_name: str, set_n: int) -> bool:
    if not api_name.startswith("TFT"):
        return False
    if '_' not in api_name:
        return True
    prefix = api_name.split('_', 1)[0]
    try:
        db_set = int(prefix[3:])
    except ValueError:
        return True
    return db_set == set_n or db_set == 16


def _champion_lookup_rows() -> list:
    db = _load_db("champion_db_file", "tft_champion_db.json", "./tft_rag_data/tft_champion_db.json")
    set_n = _effective_set_number()
    rows = []
    for api_name, info in db.items():
        if not _current_set_only(api_name, set_n):
            continue
        short_id, name_en, name_cn = _champion_names(info, api_name, set_n)
        aliases = {
            api_name,
            short_id,
            name_en,
            name_cn,
            info.get("name_en", ""),
            info.get("name_cn", ""),
            info.get("name_zh", ""),
        }
        aliases.update(info.get("traits", [])[:4])
        norm_aliases = sorted({_normalize_lookup_text(_fix_mojibake_text(alias)) for alias in aliases if alias})
        try:
            cost = int(info.get("cost", 1) or 1)
        except Exception:
            cost = 1
        rows.append({
            "api_name": api_name,
            "short_id": short_id,
            "name_en": name_en,
            "name_cn": name_cn,
            "cost": cost,
            "aliases": norm_aliases,
        })
    return rows

def _score_champion_alias(query: str, alias: str) -> float:
    if not query or not alias:
        return 0.0
    if query == alias:
        return 1.0
    if query in alias or alias in query:
        return 0.92 + min(len(query), len(alias)) / max(len(query), len(alias), 1) * 0.06
    return SequenceMatcher(None, query, alias).ratio()


def _find_champion_from_token(token: str, rows: list, used_ids: set) -> dict | None:
    q = _normalize_lookup_text(token)
    if not q:
        return None
    best = None
    best_score = 0.0
    for row in rows:
        if row["api_name"] in used_ids:
            continue
        row_score = max((_score_champion_alias(q, alias) for alias in row["aliases"]), default=0.0)
        if row_score > best_score:
            best = row
            best_score = row_score
    threshold = 0.74 if re.search(r"[\u4e00-\u9fff]", token) else 0.80
    if best is not None and best_score >= threshold:
        return best
    return None


def _convert_text_fuzzy(text: str) -> dict:
    from tft_converter import from_text

    result = from_text(text, _effective_set_number())
    if not result.get("error"):
        result["_source"] = "text_fuzzy"
    return result

# ── HTML ───────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TFT 阵容顾问</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#eef6ff;--sur:#ffffff;--brd:#c8daf2;--gold:#2f6fda;--gd2:#8eb7ee;
  --teal:#2563eb;--td2:#4f8ff7;--red:#dc4c64;--txt:#14324f;--dim:#6e88a6;--r:8px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Noto Sans SC',sans-serif;font-weight:300;
  min-height:100vh;display:flex;flex-direction:column}
/* Header */
header{border-bottom:1px solid var(--brd);padding:10px 18px;display:flex;align-items:center;gap:10px;
  background:linear-gradient(90deg,#f8fbff,#e8f2ff);flex-shrink:0}
.logo{font-family:Rajdhani,sans-serif;font-size:18px;font-weight:700;
  letter-spacing:3px;color:var(--gold);text-transform:uppercase;white-space:nowrap}
.logo span{color:var(--teal)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.kb-badge{font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid var(--brd);
  color:var(--dim);display:flex;align-items:center;gap:6px;white-space:nowrap}
.kb-badge .dot{width:7px;height:7px;border-radius:50%;background:var(--dim);transition:.4s;flex-shrink:0}
.kb-badge.ready .dot{background:var(--teal);box-shadow:0 0 6px var(--teal)}
.kb-badge.building .dot{background:var(--gold);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.hdr-btn{background:transparent;border:1px solid var(--brd);border-radius:4px;
  color:var(--dim);padding:4px 10px;cursor:pointer;font-size:12px;
  font-family:'Noto Sans SC',sans-serif;transition:.2s;white-space:nowrap}
.hdr-btn:hover{color:var(--gold);border-color:var(--gd2)}
/* Layout */
.layout{display:flex;flex:1;overflow:hidden;height:calc(100vh - 49px)}
/* Left panel */
.input-panel{width:330px;min-width:260px;border-right:1px solid var(--brd);
  display:flex;flex-direction:column;overflow:hidden}
.panel-title{font-family:Rajdhani,sans-serif;font-size:11px;font-weight:700;
  letter-spacing:2px;color:var(--dim);text-transform:uppercase;
  padding:9px 14px 7px;border-bottom:1px solid var(--brd);flex-shrink:0}
.inp-tabs{display:flex;border-bottom:1px solid var(--brd);flex-shrink:0}
.inp-tab{flex:1;padding:8px 4px;border:none;background:transparent;color:var(--dim);
  font-size:11px;cursor:pointer;font-family:'Noto Sans SC',sans-serif;
  transition:.15s;border-bottom:2px solid transparent}
.inp-tab.active{color:var(--teal);border-bottom-color:var(--teal)}
.inp-tab:hover:not(.active){color:var(--txt)}
.tab-pane{display:none;flex:1;overflow-y:auto;padding:12px;flex-direction:column;gap:9px}
.tab-pane.active{display:flex}
/* Common inputs */
.drop-zone{border:2px dashed var(--brd);border-radius:var(--r);padding:20px 10px;
  text-align:center;cursor:pointer;color:var(--dim);font-size:12px;transition:.2s;flex-shrink:0}
.drop-zone:hover,.drop-zone.over{border-color:var(--td2);color:var(--teal);background:#f2f8ff}
.drop-zone .dz-icon{font-size:22px;margin-bottom:5px}
.drop-zone small{display:block;margin-top:3px;font-size:10px;color:var(--dim)}
.preview-img{width:100%;border-radius:var(--r);display:none;margin-top:6px;flex-shrink:0}
.inp-textarea{width:100%;min-height:90px;background:var(--bg);border:1px solid var(--brd);
  border-radius:var(--r);color:var(--txt);font-family:'Noto Sans SC',sans-serif;font-size:12px;
  padding:8px 10px;outline:none;resize:vertical;transition:.2s;line-height:1.6}
.inp-textarea:focus{border-color:var(--td2)}
.inp-textarea::placeholder{color:var(--dim)}
.inp-submit{padding:8px;border:none;border-radius:var(--r);cursor:pointer;background:var(--td2);
  color:#fff;font-family:Rajdhani,sans-serif;font-size:13px;font-weight:600;
  letter-spacing:1px;transition:.2s;width:100%;flex-shrink:0}
.inp-submit:hover{background:var(--teal);color:var(--bg)}
.inp-submit:disabled{background:var(--brd);cursor:not-allowed;color:var(--dim)}
/* Screenshot mode selector */
.ss-mode-row{display:flex;gap:4px;flex-wrap:wrap;flex-shrink:0}
.ss-mode-btn{flex:1;min-width:50px;padding:5px 4px;border:1px solid var(--brd);border-radius:4px;
  background:transparent;color:var(--dim);font-size:10px;cursor:pointer;
  font-family:'Noto Sans SC',sans-serif;transition:.15s;white-space:nowrap}
.ss-mode-btn.active{border-color:var(--td2);color:#fff;background:var(--td2)}
.ss-mode-btn:hover:not(.active){color:var(--txt);border-color:var(--brd)}
.ss-mode-hint{font-size:10px;color:var(--dim);padding:2px 2px 0;flex-shrink:0;min-height:14px}
.result-box{background:#f6fbff;border:1px solid var(--td2);border-radius:var(--r);
  padding:8px 10px;font-size:11px;line-height:1.7;color:var(--txt);display:none;flex-shrink:0}
.result-box.show{display:block}
.result-tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;
  margin:2px;border:1px solid var(--td2);color:var(--teal);background:#eef6ff}
.result-tag.trait-active{border-color:var(--gold);color:var(--gold);background:#f8fbff}
.result-tag.trait-near{border-color:#7aa7ee;color:#2f6fda;background:#edf5ff}
.result-tag.trait-inactive{border-color:var(--brd);color:var(--dim);background:#f5f9ff}
.lbl{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}

/* ── Builder ── */
.builder-search{position:relative;flex-shrink:0;z-index:40}
.builder-search input{width:100%;background:var(--bg);border:1px solid var(--brd);
  border-radius:var(--r);color:var(--txt);padding:7px 10px;font-size:12px;
  font-family:'Noto Sans SC',sans-serif;outline:none;transition:.2s}
.builder-search input:focus{border-color:var(--td2)}
.builder-search input::placeholder{color:var(--dim)}
.champ-dropdown{background:var(--sur);border:1px solid var(--brd);border-radius:var(--r);
  max-height:320px;overflow-y:auto;display:none;position:absolute;z-index:120;
  width:100%;top:calc(100% + 2px);left:0;box-shadow:0 12px 28px rgba(37,99,235,.16)}
.champ-dropdown.show{display:block}
.champ-item{display:flex;align-items:center;gap:8px;padding:5px 8px;cursor:pointer;
  transition:.1s;border-bottom:1px solid #e3eefb}
.champ-item:hover{background:#eef6ff}
.champ-item:last-child{border-bottom:none}
.champ-item img{width:28px;height:28px;border-radius:3px;object-fit:cover;flex-shrink:0}
.champ-item .ci-name{font-size:12px;color:var(--txt)}
.champ-item .ci-cost{font-size:10px;margin-left:auto}
.c1{color:#aaa}.c2{color:#4caf50}.c3{color:#64b5f6}.c4{color:#ce93d8}.c5{color:#f1c40f}
.roster-list{display:flex;flex-wrap:wrap;gap:5px;min-height:32px;flex-shrink:0}
.roster-chip{display:flex;align-items:center;gap:4px;background:#eef6ff;border:1px solid var(--td2);
  border-radius:4px;padding:3px 6px;cursor:pointer;font-size:10px;color:var(--teal);position:relative}
.roster-chip img{width:22px;height:22px;border-radius:2px;object-fit:cover}
.roster-chip .rc-del{font-size:9px;color:var(--red);margin-left:2px}
.roster-chip:hover{border-color:var(--gold)}
.roster-chip .rc-name{max-width:52px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px}
.roster-chip .rc-items{display:flex;gap:1px;align-items:center}
.roster-chip .rc-star{font-size:9px;color:#f1c40f;margin-left:1px;cursor:pointer;padding:0 2px;user-select:none}
.roster-chip .rc-star:hover{color:#fff}
.roster-chip .rc-equip{font-size:9px;color:var(--teal);margin-left:2px;cursor:pointer;padding:0 2px;user-select:none}
.roster-chip .rc-equip:hover{color:var(--gold)}
/* builder control row */
.builder-ctrl-row{display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;margin-top:4px;gap:4px}

/* Hex board */
.hex-board{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;flex-shrink:0}
.hex-cell{aspect-ratio:1;border:1px dashed var(--brd);border-radius:5px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;transition:.15s;background:var(--sur);overflow:hidden;
  position:relative;min-height:0;font-size:8px;color:var(--dim)}
.hex-cell:hover{border-color:var(--td2);background:#eef6ff}
.hex-cell.occupied{border-style:solid}
.hex-cell.cost1{border-color:#666}.hex-cell.cost2{border-color:#4caf50}
.hex-cell.cost3{border-color:#64b5f6}.hex-cell.cost4{border-color:#ce93d8}
.hex-cell.cost5{border-color:#f1c40f}
.hex-cell .champ-portrait{width:100%;height:100%;object-fit:cover;border-radius:4px}
.hex-cell .champ-name{position:absolute;bottom:0;left:0;right:0;
  background:rgba(0,0,0,.78);font-size:7px;text-align:center;padding:1px 2px;
  color:#fff;line-height:1.2;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.hex-cell .star-row{position:absolute;top:1px;left:0;right:0;display:flex;justify-content:center;gap:1px}
.hex-cell .star{font-size:7px;line-height:1;color:#f1c40f}
.hex-cell .item-row{position:absolute;bottom:11px;left:0;right:0;
  display:flex;justify-content:center;gap:1px}
.hex-cell .item-icon{width:13px;height:13px;border-radius:2px;border:1px solid rgba(255,255,255,.25);object-fit:cover}
.hex-cell .remove-btn{position:absolute;top:1px;right:1px;background:rgba(200,0,0,.85);
  border:none;color:#fff;border-radius:2px;font-size:7px;cursor:pointer;
  padding:0 2px;line-height:13px;display:none;z-index:10}
.hex-cell .equip-btn{position:absolute;top:1px;left:1px;background:rgba(79,143,247,.92);
  border:none;color:#fff;border-radius:2px;font-size:7px;cursor:pointer;
  padding:0 3px;line-height:13px;display:none;z-index:10}
.hex-cell:hover .remove-btn{display:block}
.hex-cell:hover .equip-btn{display:block}
.hex-cell.drag-over{border-color:var(--teal);background:#dbeafe;box-shadow:0 0 0 2px rgba(37,99,235,.18) inset}
.hex-cell.drag-src{opacity:.68}

/* Item modal */
.item-modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:300;
  display:none;align-items:center;justify-content:center}
.item-modal-bg.show{display:flex}
.item-modal{background:var(--sur);border:1px solid var(--brd);border-radius:10px;
  padding:18px;width:380px;max-height:78vh;display:flex;flex-direction:column;gap:10px}
.item-modal h4{font-family:Rajdhani,sans-serif;font-size:14px;letter-spacing:1px;
  color:var(--gold);flex-shrink:0}
.item-modal-search{background:var(--bg);border:1px solid var(--brd);border-radius:var(--r);
  color:var(--txt);padding:6px 10px;font-size:12px;outline:none;width:100%;
  font-family:'Noto Sans SC',sans-serif;flex-shrink:0}
.item-modal-search:focus{border-color:var(--td2)}
.sel-items-preview{display:flex;gap:4px;align-items:center;flex-wrap:wrap;
  min-height:22px;font-size:10px;color:var(--dim);flex-shrink:0}
.sel-items-preview img{width:22px;height:22px;border-radius:3px;object-fit:cover}
.item-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;overflow-y:auto;flex:1}
.item-thumb{display:flex;flex-direction:column;align-items:center;gap:2px;cursor:pointer;
  padding:4px;border-radius:4px;border:1px solid transparent;transition:.15s}
.item-thumb:hover{background:#eef6ff;border-color:var(--td2)}
.item-thumb.selected{background:#dbeafe;border-color:var(--teal)}
.item-thumb img{width:40px;height:40px;border-radius:4px;object-fit:cover}
.item-thumb span{font-size:9px;color:var(--dim);text-align:center;
  word-break:break-all;line-height:1.2;max-width:56px}
.item-modal-footer{display:flex;gap:8px;flex-shrink:0}
.item-modal-footer button{flex:1;padding:7px;border:none;border-radius:var(--r);
  cursor:pointer;font-family:Rajdhani,sans-serif;font-size:13px;font-weight:600;letter-spacing:1px}
.btn-ok{background:var(--td2);color:#fff}.btn-ok:hover{background:var(--teal);color:var(--bg)}
.btn-c2{background:transparent;border:1px solid var(--brd);color:var(--dim)}.btn-c2:hover{color:var(--txt)}

/* Right: chat */
.chat-panel{flex:1;display:flex;flex-direction:column;overflow:hidden;background:linear-gradient(180deg,#fbfdff 0%,#eef6ff 100%)}
.messages{flex:1;overflow-y:auto;padding:14px 18px;display:flex;
  flex-direction:column;gap:12px;scroll-behavior:smooth;background:linear-gradient(180deg,#f8fbff 0%,#eef6ff 100%)}
.messages::-webkit-scrollbar{width:4px}
.messages::-webkit-scrollbar-thumb{background:var(--brd);border-radius:2px}
.msg{display:flex;gap:9px;max-width:820px;animation:fadeUp .2s ease}
.msg.user{flex-direction:row-reverse;align-self:flex-end}
.msg.ai{align-self:flex-start}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.avatar{width:26px;height:26px;border-radius:4px;display:flex;align-items:center;
  justify-content:center;font-size:12px;flex-shrink:0}
.msg.user .avatar{background:#dbeafe}.msg.ai .avatar{background:#eef6ff;border:1px solid var(--td2)}
.bubble{padding:9px 13px;border-radius:var(--r);font-size:13px;line-height:1.75;max-width:680px}
.msg.user .bubble{background:#edf4ff;border:1px solid #c9dbf4;color:var(--txt)}
.msg.ai   .bubble{background:#ffffff;border:1px solid #d6e6fb;color:var(--txt)}
.bubble strong,.bubble b{color:var(--gold);font-weight:500}
.bubble em,.bubble i{color:var(--teal);font-style:normal}
.bubble h2,.bubble h3{color:var(--gold);font-family:Rajdhani,sans-serif;font-size:14px;letter-spacing:1px;margin:8px 0 3px}
.bubble ul,.bubble ol{padding-left:16px;margin:4px 0}
.bubble li{margin:2px 0}
.bubble code{background:#edf4ff;padding:1px 4px;border-radius:3px;font-size:11px;color:var(--teal);font-family:monospace}
.bubble hr{border:none;border-top:1px solid var(--brd);margin:7px 0}
.bubble p{margin:3px 0}
.thinking{display:flex;gap:5px;align-items:center;padding:11px 13px}
.thinking span{width:5px;height:5px;background:var(--td2);border-radius:50%;animation:bounce 1.2s infinite}
.thinking span:nth-child(2){animation-delay:.2s}.thinking span:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:scale(1);opacity:.5}40%{transform:scale(1.4);opacity:1}}
.quick-bar{padding:6px 16px 8px;display:flex;flex-wrap:wrap;gap:4px;flex-shrink:0;background:#f3f8ff}
.qp{padding:3px 10px;border:1px solid var(--brd);border-radius:20px;background:transparent;
  color:var(--dim);font-size:11px;cursor:pointer;font-family:'Noto Sans SC',sans-serif;
  transition:.15s;white-space:nowrap}
.qp:hover{border-color:var(--td2);color:var(--teal);background:#eef6ff}
/* Simplified chat bar */
.chat-bar{padding:8px 16px 12px;border-top:1px solid var(--brd);
  display:flex;flex-direction:column;gap:6px;background:#f7fbff;flex-shrink:0}
.chat-bar-top{display:flex;align-items:center;gap:8px}
.mode-pills{display:flex;gap:4px}
.mode-pill{padding:4px 12px;border:1px solid var(--brd);border-radius:20px;background:transparent;
  color:var(--dim);font-size:11px;cursor:pointer;font-family:'Noto Sans SC',sans-serif;
  transition:.15s;white-space:nowrap}
.mode-pill.active{border-color:var(--td2);color:#fff;background:var(--td2)}
.mode-pill:hover:not(.active){color:var(--txt)}
.chat-bar-row{display:flex;gap:7px;align-items:flex-end}
.chat-bar-row textarea{flex:1;background:#ffffff;border:1px solid var(--brd);border-radius:var(--r);
  color:var(--txt);font-family:'Noto Sans SC',sans-serif;font-size:13px;font-weight:300;
  padding:8px 12px;resize:none;min-height:40px;max-height:120px;outline:none;line-height:1.6;transition:.2s}
.chat-bar-row textarea:focus{border-color:var(--td2)}
.chat-bar-row textarea::placeholder{color:var(--dim)}
.send-btn{width:40px;height:40px;background:var(--td2);border:none;border-radius:var(--r);
  cursor:pointer;display:flex;align-items:center;justify-content:center;transition:.2s;flex-shrink:0}
.send-btn:hover{background:var(--teal)}.send-btn:disabled{background:var(--brd);cursor:not-allowed}
.send-btn svg{width:16px;height:16px;fill:var(--bg)}
/* Welcome */
.welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;color:var(--dim);padding:40px;text-align:center}
.welcome h2{font-family:Rajdhani,sans-serif;font-size:22px;letter-spacing:2px;color:var(--teal);font-weight:600}
.welcome p{font-size:12px;max-width:360px;line-height:1.7}
/* Settings modal */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:100;
  display:none;align-items:center;justify-content:center}
.modal-bg.show{display:flex}
.modal{background:var(--sur);border:1px solid var(--brd);border-radius:10px;
  padding:22px;width:400px;max-height:80vh;overflow-y:auto}
.modal h3{font-family:Rajdhani,sans-serif;font-size:16px;letter-spacing:2px;color:var(--gold);margin-bottom:14px}
.form-row{margin-bottom:11px}
.form-row label{display:block;font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.form-row input,.form-row select{width:100%;background:var(--bg);border:1px solid var(--brd);
  border-radius:var(--r);color:var(--txt);padding:7px 10px;font-size:12px;
  font-family:'Noto Sans SC',sans-serif;outline:none}
.form-row input:focus,.form-row select:focus{border-color:var(--td2)}
.modal-btns{display:flex;gap:8px;margin-top:14px}
.modal-btn{flex:1;padding:8px;border:none;border-radius:var(--r);cursor:pointer;
  font-family:Rajdhani,sans-serif;font-size:13px;font-weight:600;letter-spacing:1px}
.modal-btn.save{background:var(--td2);color:#fff}.modal-btn.save:hover{background:var(--teal);color:var(--bg)}
.modal-btn.cancel{background:transparent;border:1px solid var(--brd);color:var(--dim)}.modal-btn.cancel:hover{color:var(--txt)}
/* Toast */
 .toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#edf5ff;
  border:1px solid var(--td2);border-radius:var(--r);padding:8px 18px;font-size:12px;
  color:var(--teal);z-index:400;opacity:0;transition:.3s;pointer-events:none;white-space:nowrap}
.toast.show{opacity:1}
</style>
</head>
<body>
<!-- Header -->
<header>
  <div class="logo">TFT <span>Advisor</span></div>
  <div class="hdr-right">
    <div class="kb-badge" id="kbBadge"><span class="dot"></span><span id="kbText">initializing</span></div>
    <button class="hdr-btn" onclick="refreshKB()">⟳ KB</button>
    <button class="hdr-btn" onclick="openSettings()">⚙ Settings</button>
  </div>
</header>

<div class="layout">
  <!-- Left panel -->
  <div class="input-panel">
    <div class="panel-title">阵容输入</div>
    <div class="inp-tabs">
      <button class="inp-tab active" onclick="switchTab('screenshot',this)">📷 截图</button>
      <button class="inp-tab" onclick="switchTab('text',this)">💬 文字</button>
      <button class="inp-tab" onclick="switchTab('builder',this)">🎮 生成</button>
    </div>

    <!-- Screenshot -->
    <div class="tab-pane active" id="tab-screenshot">
      <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()"
           ondragover="event.preventDefault();this.classList.add('over')"
           ondragleave="this.classList.remove('over')" ondrop="handleDrop(event)">
        <div class="dz-icon">🖼</div><div>点击或拖拽截图到此处</div>
        <small>支持 JPG / PNG / WebP</small>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
      <img class="preview-img" id="previewImg">
      <!-- 识别模式选择 -->
      <div class="lbl" style="margin-top:6px">识别模式</div>
      <div class="ss-mode-row" id="ssModeRow">
        <button type="button" class="ss-mode-btn active" onclick="setSsMode('auto',this)">🤖 自动</button>
        <button type="button" class="ss-mode-btn" onclick="setSsMode('board',this)">♟ 棋盘</button>
        <button type="button" class="ss-mode-btn" onclick="setSsMode('lineup',this)">➖ 横排</button>
        <button type="button" class="ss-mode-btn" onclick="setSsMode('global',this)">🌐 全局</button>
        <button type="button" class="ss-mode-btn" onclick="setSsMode('duel',this)">🆚 对战</button>
      </div>
      <div class="ss-mode-hint" id="ssModeHint">自动检测截图类型</div>
      <button class="inp-submit" id="btnSS" onclick="submitScreenshot()" disabled>🔍 识别阵容</button>
      <div class="result-box" id="resSS"></div>
    </div>

    <!-- Text -->
    <div class="tab-pane" id="tab-text">
      <div class="lbl">英雄ID / JSON</div>
      <textarea class="inp-textarea" id="textInput"
        placeholder="输入英雄中文名、英文名、简称，或直接粘贴 JSON&#10;示例: 贝蕾亚 阿卡丽 茂凯 / Briar Akali Maokai"></textarea>
      <button class="inp-submit" onclick="submitText()">📥 导入阵容</button>
      <div class="result-box" id="resTxt"></div>
    </div>

    <!-- Builder -->
    <div class="tab-pane" id="tab-builder">
      <div class="lbl">搜索英雄</div>
      <div class="builder-search">
        <input type="text" id="champSearch" placeholder="输入英雄中文名、英文名或羁绊..." autocomplete="off"
               oninput="filterChamps(this.value)" onfocus="showDD()">
        <div class="champ-dropdown" id="champDD"></div>
      </div>
      <div class="builder-ctrl-row">
        <span class="lbl" style="margin:0">
          已选 <span id="rosterCnt" style="color:var(--teal)">0</span>/9
        </span>
        <span style="font-size:9px;color:var(--dim)">点英雄↓选星级/装备 · 拖格子换位置</span>
      </div>
      <div class="roster-list" id="rosterList"></div>
      <div class="lbl" style="margin-top:4px">棋盘预览</div>
      <div class="hex-board" id="hexBoard"></div>
      <button class="inp-submit" style="margin-top:6px" onclick="submitBuilder()">📋 提交阵容分析</button>
      <div class="result-box" id="resBuilder"></div>
    </div>
  </div>

  <!-- Right: chat -->
  <div class="chat-panel">
    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <h2>🎮 TFT 阵容顾问</h2>
        <p>在左侧导入阵容，选择分析模式后提问。</p>
        <p style="margin-top:8px;font-size:11px;color:var(--dim)">
          ⚔ Single — 单人分析 &nbsp;|&nbsp; 🆚 Duel — 对局克制 &nbsp;|&nbsp; 🌐 Global — 全局研判
        </p>
      </div>
    </div>
    <div class="quick-bar">
      <button class="qp" onclick="sendQ('阵容整体评价和优劣势')">📊 评价</button>
      <button class="qp" onclick="sendQ('最优发展路线和过渡思路')">🗺 路线</button>
      <button class="qp" onclick="sendQ('主C装备优先级建议')">⚔ 装备</button>
      <button class="qp" onclick="sendQ('经济节奏和升级时机')">💰 经济</button>
      <button class="qp" onclick="sendQ('站位和阵型建议')">🛡 站位</button>
      <button class="qp" onclick="sendQ('当前赛季 S 级阵容推荐')">🏆 Meta</button>
    </div>
    <div class="chat-bar">
      <div class="chat-bar-top">
        <span style="font-size:11px;color:var(--dim);white-space:nowrap">模式：</span>
        <div class="mode-pills">
          <button type="button" class="mode-pill active" onclick="setMode('single',this)">⚔ Single</button>
          <button type="button" class="mode-pill" onclick="setMode('duel',this)">🆚 Duel</button>
          <button type="button" class="mode-pill" onclick="setMode('global',this)">🌐 Global</button>
        </div>
      </div>
      <div class="chat-bar-row">
        <textarea id="chatInput" rows="1" placeholder="输入问题… Enter 发送，Shift+Enter 换行"
          onkeydown="handleChatKey(event)" oninput="autoResize(this)"></textarea>
        <button class="send-btn" id="sendBtn" onclick="sendMsg()">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
    </div>
  </div>
</div>

<!-- Item picker modal -->
<div class="item-modal-bg" id="itemModalBg">
  <div class="item-modal">
    <h4 id="itemModalTitle">选择装备（最多3件）</h4>
    <input class="item-modal-search" id="itemSearch" type="text"
           placeholder="搜索装备..." oninput="filterItems(this.value)">
    <div class="sel-items-preview" id="selItemsPrev"></div>
    <div class="item-grid" id="itemGrid"></div>
    <div class="item-modal-footer">
      <button class="btn-ok" onclick="confirmItems()">✓ 确认</button>
      <button class="btn-c2" onclick="closeItemModal()">取消</button>
    </div>
  </div>
</div>

<!-- Settings modal -->
<div class="modal-bg" id="settingsModal">
  <div class="modal">
    <h3>⚙ SETTINGS</h3>
    <div class="form-row">
      <label>LLM Provider</label>
      <select id="cfgProvider" onchange="toggleModelRow()">
        <option value="openrouter">OpenRouter（免费模型）</option>
        <option value="anthropic">Anthropic Claude</option>
      </select>
    </div>
    <div class="form-row" id="rowApiKey">
      <label id="apiKeyLabel">API Key</label>
      <input type="password" id="cfgApiKey" placeholder="sk-or-v1-...">
    </div>
    <div class="form-row" id="rowModel">
      <label>OpenRouter Model</label>
      <input type="text" id="cfgModel" placeholder="deepseek/deepseek-chat-v3-0324:free">
    </div>
    <div class="form-row">
      <label>Riot API Key</label>
      <input type="password" id="cfgRiotKey" placeholder="RGAPI-...">
    </div>
    <div style="font-size:10px;color:var(--dim);margin-top:4px;line-height:1.6">
      <a href="https://openrouter.ai/keys" target="_blank" style="color:var(--teal)">openrouter.ai/keys</a>
      &nbsp;·&nbsp;
      <a href="https://developer.riotgames.com/" target="_blank" style="color:var(--teal)">developer.riotgames.com</a>
    </div>
    <div class="modal-btns">
      <button class="modal-btn save" onclick="saveSettings()">保存</button>
      <button class="modal-btn cancel" onclick="closeSettings()">取消</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
'use strict';
// ── State ──────────────────────────────────────────────────────────────────
let currentMode='single', selectedFile=null, isThinking=false;
let champDB=[], itemDB=[];
let roster=[];  // [{api_name,name_en,cost,img_url,star,items,row,col}]
let itemModalIdx=-1, itemModalSel=[];
let dragBoardApiName='';
let ssRecognizeMode='auto';   // 截图识别模式: auto|board|lineup|global|duel
const DDRAGON_VER = '15.8.1';
const SS_MODE_HINTS={
  auto:'自动检测截图类型',board:'棋盘视角（4行×7列）',
  lineup:'结算横排（水平一行）',global:'全局视角（8玩家总览）',duel:'对战双方对比'
};
function getChampName(c){ return (c.name_cn||c.name_en||c.short_id||c.api_name||'').toString(); }
function getItemName(i){ return (i.name_cn||i.name_en||i.api_name||'').toString(); }
function getChampImgUrl(c){ return c.img_url || `https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VER}/img/tft-champion/${c.api_name}.png`; }
function getItemImgUrl(i){ return i.img_url || `https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VER}/img/tft-item/${i.api_name}.png`; }

// ── Init ───────────────────────────────────────────────────────────────────
(async()=>{
  pollKB(); setInterval(pollKB,3000);
  initHexBoard();
  try{
    // 优先请求带本地图标的接口，失败时自动回退到 data 接口。
    let cr = await fetch('/api/assets/champions').then(r=>r.json()).catch(()=>({ok:false}));
    if(!cr.ok){ cr = await fetch('/api/data/champions').then(r=>r.json()).catch(()=>({ok:false})); }
    let ir = await fetch('/api/assets/items').then(r=>r.json()).catch(()=>({ok:false}));
    if(!ir.ok){ ir = await fetch('/api/data/items').then(r=>r.json()).catch(()=>({ok:false})); }
    if(cr.ok && Array.isArray(cr.champions) && cr.champions.length>0){
      champDB=cr.champions;
      console.log('[TFT] champion DB loaded:',champDB.length,'entries');
      toast('✓ 英雄库已加载 '+champDB.length+' 位英雄');
    } else {
      console.warn('[TFT] champion DB empty or failed',cr);
    }
    if(ir.ok && Array.isArray(ir.items) && ir.items.length>0){
      itemDB=ir.items;
      console.log('[TFT] item DB loaded:',itemDB.length,'entries');
    } else {
      console.warn('[TFT] item DB empty or failed',ir);
    }
  }catch(e){ console.warn('init load failed',e); }
})();

// ── KB ─────────────────────────────────────────────────────────────────────
function pollKB(){
  fetch('/api/kb/status').then(r=>r.json()).then(d=>{
    const b=document.getElementById('kbBadge'),t=document.getElementById('kbText');
    b.className='kb-badge '+(d.building?'building':(d.ready?'ready':''));
    t.textContent=d.building?'Building...':(d.stats||'ready');
  }).catch(()=>{});
}
function refreshKB(){ fetch('/api/kb/refresh',{method:'POST'}).then(()=>toast('知识库刷新中...')); }

// ── Mode ───────────────────────────────────────────────────────────────────
function setMode(m,el){
  currentMode=m;
  document.querySelectorAll('.mode-pill').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  toast('模式: '+m);
}
// 截图识别模式
function setSsMode(m,el){
  ssRecognizeMode=m;
  document.querySelectorAll('.ss-mode-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  const hint=document.getElementById('ssModeHint');
  if(hint) hint.textContent=SS_MODE_HINTS[m]||'';
}

// ── Tabs ───────────────────────────────────────────────────────────────────
function switchTab(name,el){
  document.querySelectorAll('.inp-tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

// ── Screenshot ─────────────────────────────────────────────────────────────
function handleFile(f){
  if(!f)return; selectedFile=f;
  const r=new FileReader();
  r.onload=e=>{const i=document.getElementById('previewImg');i.src=e.target.result;i.style.display='block';};
  r.readAsDataURL(f);
  document.getElementById('btnSS').disabled=false;
}
function handleDrop(e){
  e.preventDefault();document.getElementById('dropZone').classList.remove('over');
  const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('image/'))handleFile(f);
}
async function submitScreenshot(){
  if(!selectedFile)return;
  const btn=document.getElementById('btnSS');btn.disabled=true;btn.textContent='识别中...';
  const fd=new FormData();
  fd.append('image',selectedFile);
  fd.append('mode',ssRecognizeMode);   // 传入用户选择的识别模式
  try{
    const d=await fetch('/api/input/screenshot',{method:'POST',body:fd}).then(r=>r.json());
    showResult('resSS',d);
    if(d.ok){
      const used=d.used_mode||ssRecognizeMode;
      const chatMode=(used==='global')?'global':((used==='duel')?'duel':'single');
      const modeBtn=document.querySelector(`.mode-pill[onclick*="${chatMode}"]`);
      if(modeBtn) setMode(chatMode,modeBtn);
      toast(`✓ 识别完成 · 模式: ${used}`);
    }
  }catch(e){toast('识别失败');}
  btn.disabled=false;btn.textContent='🔍 识别阵容';
}
// ── Text ───────────────────────────────────────────────────────────────────
async function submitText(){
  const text=document.getElementById('textInput').value.trim();if(!text)return;
  const box=document.getElementById('resTxt');box.innerHTML='处理中...';box.classList.add('show');
  try{
    const d=await fetch('/api/input/text',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}).then(r=>r.json());
    showResult('resTxt',d);
  }catch(e){box.innerHTML='请求失败';}
}
// ── Champion dropdown ──────────────────────────────────────────────────────
function normLookupText(v){
  return (v||'').toString().toLowerCase().replace(/[\s\-_'/·.()（）\[\]{}]/g,'');
}
function champSearchScore(champ,q){
  const qn=normLookupText(q);
  if(!qn)return 0;
  const fields=[getChampName(champ),champ.name_cn||'',champ.api_name,...(champ.traits||[])];
  let best=0;
  for(const field of fields){
    const fn=normLookupText(field);
    if(!fn) continue;
    if(fn===qn) best=Math.max(best,1);
    else if(fn.includes(qn) || qn.includes(fn)) best=Math.max(best,0.9 + Math.min(fn.length,qn.length)/Math.max(fn.length,qn.length,1)*0.06);
    else {
      let same=0;
      for(const ch of qn){ if(fn.includes(ch)) same++; }
      const ratio=same/Math.max(qn.length,fn.length,1);
      if(ratio>best) best=ratio;
    }
  }
  return best;
}
function renderDD(list){
  const dd=document.getElementById('champDD');
  if(!list||!list.length){dd.innerHTML='<div style="padding:8px;color:var(--dim);font-size:11px">无匹配英雄</div>';return;}
  dd.innerHTML=list.map(c=>{
    const nameEn=getChampName(c);
    const nameCn=c.name_cn||'';
    const showDual=!!(nameCn && nameCn!==nameEn);
    const display=showDual?`${nameEn} <span style="color:var(--dim)">${nameCn}</span>`:nameEn;
    const traits=(c.traits||[]).slice(0,2).map(t=>`<span style="font-size:8px;color:var(--dim);background:#edf4ff;padding:0 3px;border-radius:2px">${t}</span>`).join('');
    const localBadge=c.local?'<span style="font-size:7px;color:#2563eb;margin-left:2px">●</span>' : '';
    return [
      `<div class="champ-item" onmousedown="addChamp('${c.api_name}')">`,
      `  <img src="${getChampImgUrl(c)}" alt="${nameEn}" width="28" height="28"`,
      `       onerror="this.onerror=null;this.src='https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VER}/img/tft-champion/${c.api_name}.png'">`,
      `  <span class="ci-name">${display}${traits?'&nbsp;'+traits:''}${localBadge}</span>`,
      `  <span class="ci-cost c${c.cost}">${c.cost}费</span>`,
      `</div>`
    ].join('');
  }).join('');
}
function filterChamps(q){
  const dd=document.getElementById('champDD');
  if(!q){ renderDD(champDB); dd.classList.toggle('show',champDB.length>0); return; }
  const scored=champDB.map(c=>({champ:c,score:champSearchScore(c,q)}))
    .filter(x=>x.score>=0.34)
    .sort((a,b)=>b.score-a.score || (a.champ.cost||1)-(b.champ.cost||1) || getChampName(a.champ).localeCompare(getChampName(b.champ),'zh-CN'))
    .map(x=>x.champ);
  renderDD(scored);
  dd.classList.toggle('show',scored.length>0);
}
function showDD(){
  const dd=document.getElementById('champDD');
  renderDD(champDB);
  if(champDB.length>0) dd.classList.add('show');
}
function hideDD(){ document.getElementById('champDD').classList.remove('show'); }
document.addEventListener('click',e=>{
  const wrap=document.querySelector('.builder-search');
  if(wrap && !wrap.contains(e.target)) hideDD();
});

function addChamp(apiName){
  if(roster.length>=9){toast('最多9名英雄');return;}
  if(roster.find(c=>c.api_name===apiName)){toast('已添加');return;}
  const c=champDB.find(x=>x.api_name===apiName);if(!c)return;
  const pos=nextPos();
  roster.push({...c,star:2,items:[],row:pos.row,col:pos.col});
  document.getElementById('champSearch').value='';
  document.getElementById('champDD').classList.remove('show');
  renderRoster();renderBoard();
}
function nextPos(){
  const occ=new Set(roster.map(c=>`${c.row}_${c.col}`));
  for(let row=0;row<4;row++) for(let col=0;col<7;col++)
    if(!occ.has(`${row}_${col}`))return{row,col};
  return{row:0,col:0};
}
function removeChamp(apiName){
  roster=roster.filter(c=>c.api_name!==apiName);renderRoster();renderBoard();
}
function renderRoster(){
  document.getElementById('rosterCnt').textContent=roster.length;
  document.getElementById('rosterList').innerHTML=roster.map(c=>{
    const stars='★'.repeat(c.star||1)+'☆'.repeat(3-(c.star||1));
    const name=getChampName(c);
    const itemIcons=(c.items||[]).slice(0,3).map(n=>{
      const it=itemDB.find(i=>i.api_name===n);
      if(!it) return '';
      const itemName=getItemName(it);
      return `<img src="${getItemImgUrl(it)}" alt="${itemName}" title="${itemName}" width="12" height="12" style="border-radius:2px;object-fit:cover" onerror="this.style.display='none'">`;
    }).join('');
    return `<div class="roster-chip" title="点击设置星级、装备或删除">
      <img src="${getChampImgUrl(c)}" alt="${name}" title="${name}" onerror="this.onerror=null;this.style.opacity='0.35'">
      <span class="rc-name">${name.length>7?name.slice(0,6)+'…':name}</span>
      <span class="rc-star" title="点击切换星级" onclick="event.stopPropagation();cycleChampStar('${c.api_name}')">${stars}</span>
      <span class="rc-equip" title="选择装备" onclick="event.stopPropagation();openItemModal('${c.api_name}')">装</span>
      ${itemIcons?`<span class="rc-items">${itemIcons}</span>`:''}
      <span class="rc-del" title="删除英雄" onclick="event.stopPropagation();removeChamp('${c.api_name}')">✕</span>
    </div>`;
  }).join('');
}

function cycleChampStar(apiName){
  const c=roster.find(x=>x.api_name===apiName);if(!c)return;
  c.star=c.star>=3?1:c.star+1;
  renderRoster();renderBoard();
}

// ── Hex board ──────────────────────────────────────────────────────────────
function initHexBoard(){
  const board=document.getElementById('hexBoard');board.innerHTML='';
  for(let row=0;row<4;row++) for(let col=0;col<7;col++){
    const cell=document.createElement('div');
    cell.className='hex-cell';cell.dataset.row=row;cell.dataset.col=col;
    cell.textContent=`${row},${col}`;
    cell.onclick=()=>boardClick(row,col);
    cell.ondragover=e=>{e.preventDefault();cell.classList.add('drag-over');};
    cell.ondragleave=()=>cell.classList.remove('drag-over');
    cell.ondrop=e=>{e.preventDefault();cell.classList.remove('drag-over');boardDrop(row,col);};
    board.appendChild(cell);
  }
}
function renderBoard(){
  const cells=document.querySelectorAll('.hex-cell');
  cells.forEach(c=>{
    c.className='hex-cell';
    c.innerHTML=`<span>${c.dataset.row},${c.dataset.col}</span>`;
    c.draggable=false;
    c.ondragstart=null;
    c.ondragend=null;
  });
  roster.forEach(champ=>{
    const idx=champ.row*7+champ.col;if(idx<0||idx>=cells.length)return;
    const cell=cells[idx];
    cell.className=`hex-cell occupied cost${champ.cost||1}`;
    cell.draggable=true;
    cell.ondragstart=e=>boardDragStart(e,champ.api_name);
    cell.ondragend=()=>boardDragEnd();
    const stars=champ.star>1?`<div class="star-row">${'<span class="star">★</span>'.repeat(champ.star)}</div>`:'';
    const items=(champ.items||[]).map(n=>{
      const it=itemDB.find(i=>i.api_name===n);
      return it?`<img class="item-icon" src="${getItemImgUrl(it)}" alt="${getItemName(it)}" draggable="false" onerror="this.style.display='none'">`:'' ;
    }).join('');
    const name=getChampName(champ);
    cell.innerHTML=`
      ${stars}
      <img class="champ-portrait" src="${getChampImgUrl(champ)}" alt="${name}" draggable="false"
           onerror="this.onerror=null;this.src='https://ddragon.leagueoflegends.com/cdn/${DDRAGON_VER}/img/tft-champion/${champ.api_name}.png'"
           onclick="event.stopPropagation();openItemModal('${champ.api_name}')">
      ${items?`<div class="item-row">${items}</div>`:''}
      <div class="champ-name">${name}</div>
      <button class="equip-btn" onclick="event.stopPropagation();openItemModal('${champ.api_name}')">装</button>
      <button class="remove-btn" onclick="event.stopPropagation();removeChamp('${champ.api_name}')">✕</button>`;
  });
}
function boardDragStart(e,apiName){
  dragBoardApiName=apiName;
  const cell=e.currentTarget;
  if(cell) cell.classList.add('drag-src');
  if(e.dataTransfer){
    e.dataTransfer.effectAllowed='move';
    e.dataTransfer.setData('text/plain',apiName);
  }
}
function boardDragEnd(){
  dragBoardApiName='';
  document.querySelectorAll('.hex-cell').forEach(c=>c.classList.remove('drag-over','drag-src'));
}
function boardDrop(row,col){
  const apiName=dragBoardApiName;
  if(!apiName)return;
  const src=roster.find(x=>x.api_name===apiName);
  if(!src) return;
  if(src.row===row && src.col===col) return boardDragEnd();
  const dst=roster.find(x=>x.row===row&&x.col===col);
  const prev={row:src.row,col:src.col};
  src.row=row;src.col=col;
  if(dst){ dst.row=prev.row; dst.col=prev.col; }
  renderBoard();
  boardDragEnd();
}
function boardClick(row,col){
  const c=roster.find(x=>x.row===row&&x.col===col);
  if(c) openItemModal(c.api_name);
}

// ── Item modal ─────────────────────────────────────────────────────────────
function openItemModal(apiName){
  itemModalIdx=roster.findIndex(c=>c.api_name===apiName);if(itemModalIdx<0)return;
  const c=roster[itemModalIdx];itemModalSel=[...(c.items||[])];
  document.getElementById('itemModalTitle').textContent=`${getChampName(c)} — 装备（最多3件）`;
  document.getElementById('itemSearch').value='';
  renderItemGrid(itemDB);updateItemPrev();
  document.getElementById('itemModalBg').classList.add('show');
}
function closeItemModal(){
  document.getElementById('itemModalBg').classList.remove('show');
  itemModalIdx=-1;itemModalSel=[];
}
function filterItems(q){
  const ql=q.toLowerCase();
  renderItemGrid(ql?itemDB.filter(i=>{
    const en=(i.name_en||'').toLowerCase();
    const cn=(i.name_cn||'').toLowerCase();
    return en.includes(ql)||cn.includes(ql)||i.api_name.toLowerCase().includes(ql);
  }):itemDB);
}
function _currentItemList(){
  const q=(document.getElementById('itemSearch').value||'').toLowerCase();
  return q?itemDB.filter(i=>{
    const en=(i.name_en||'').toLowerCase();
    const cn=(i.name_cn||'').toLowerCase();
    return en.includes(q)||cn.includes(q)||i.api_name.toLowerCase().includes(q);
  }):itemDB;
}
function renderItemGrid(list){
  const grid=document.getElementById('itemGrid');if(!grid)return;
  grid.innerHTML=(list||[]).map(it=>{
    const itemName=getItemName(it);
    const subName=(it.name_en&&it.name_en!==itemName)?it.name_en:'';
    const short=itemName.length>10?itemName.slice(0,9)+'…':itemName;
    const localDot=it.local?'<i style="font-size:6px;color:#2ecc71;position:absolute;top:1px;right:1px;font-style:normal">●</i>':'';
    return `<div class="item-thumb ${itemModalSel.includes(it.api_name)?'selected':''}"
         style="position:relative" title="${subName?itemName+' / '+subName:itemName}"
         onclick="event.stopPropagation();toggleItem('${it.api_name}')">
      ${localDot}
      <img src="${getItemImgUrl(it)}" alt="${itemName}" onerror="this.onerror=null;this.style.opacity='.3'">
      <span>${short}</span>
    </div>`;
  }).join('');
}

function toggleItem(apiName){
  if(itemModalSel.includes(apiName)){
    itemModalSel=itemModalSel.filter(i=>i!==apiName);
  }else{
    if(itemModalSel.length>=3){toast('最多3件装备');return;}
    itemModalSel.push(apiName);
  }
  renderItemGrid(_currentItemList());
  updateItemPrev();
}
function updateItemPrev(){
  const p=document.getElementById('selItemsPrev');
  p.innerHTML=itemModalSel.length?itemModalSel.map(n=>{
    const it=itemDB.find(i=>i.api_name===n);
    return it?`<img src="${getItemImgUrl(it)}" title="${getItemName(it)}" onerror="this.style.display='none'" width="22" height="22" style="border-radius:3px;object-fit:cover">`:'';
  }).join(''):'<span>未选择装备</span>';
}
function confirmItems(){
  if(itemModalIdx<0)return;
  roster[itemModalIdx].items=[...itemModalSel];
  renderBoard();closeItemModal();toast('装备已更新');
}

// ── Builder submit ─────────────────────────────────────────────────────────
async function submitBuilder(){
  if(!roster.length){toast('请先添加英雄');return;}
  const box=document.getElementById('resBuilder');box.innerHTML='处理中...';box.classList.add('show');
  const units=roster.map(c=>({champion_id:c.api_name,star:c.star,items:c.items,position:{row:c.row,col:c.col}}));
  try{
    const d=await fetch('/api/input/builder',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({units})}).then(r=>r.json());
    showResult('resBuilder',d);
  }catch(e){box.innerHTML='请求失败';}
}

// ── Shared result renderer ──────────────────────────────────────────────────
function showResult(boxId,d){
  const box=document.getElementById(boxId);box.classList.add('show');
  if(d.ok){
    const a=d.analysis||{};
    const renderChampTags=list=>(list||[]).map(c=>`<span class="result-tag">${c.name_en||c.short_id||c.id||'?'} ${c.star||1}★</span>`).join('');
    const renderTraitTags=list=>(list||[]).filter(t=>t.count>0).map(t=>{
      const active=!!t.active;
      const need=Math.max(0,(t.next_threshold||0)-(t.count||0));
      const near=!active && need===1;
      const cls=active?'trait-active':(near?'trait-near':'trait-inactive');
      const badge=active
        ? `${t.name_en||t.id||'?'} ${t.count||0}${t.level_name?` · ${t.level_name}`:''}`
        : `${t.name_en||t.id||'?'} ${t.count||0}/${t.next_threshold||'?'}`;
      const hint=active
        ? `已激活${t.level_name?` ${t.level_name}`:''}${(t.thresholds||[]).length?` · 阈值 ${t.thresholds.join('/')}`:''}`
        : `未激活 · 还差 ${need} 个 · 阈值 ${(t.thresholds||[]).join('/')||'未知'}`;
      return `<span class="result-tag ${cls}" title="${hint}">${badge}</span>`;
    }).join('');

    if(a.layout==='global' && (a.players||[]).length){
      const rows=(a.players||[]).map(p=>{
        const champs=renderChampTags(p.champions||[]);
        const traits=renderTraitTags(p.traits||[]);
        return `<div style="margin-top:6px"><strong>第${p.rank}名</strong> · ${p.champion_count||0} 名英雄${champs?`<br>${champs}`:''}${traits?`<br>${traits}`:''}</div>`;
      }).join('');
      box.innerHTML=`<strong>✓ 全局识别 ${a.player_count||0} 名玩家 / ${a.champion_count||0} 名英雄</strong>${a.contested&&a.contested.length?`<br><span style="color:var(--dim)">争抢: ${a.contested.join(' / ')}</span>`:''}${rows}`;
      hideWelcome();toast(`✓ 已导入全局视角 · ${a.player_count||0} 名玩家`);
      return;
    }

    if(a.layout==='duel' && (a.boards||[]).length){
      const boards=(a.boards||[]).map(b=>{
        const champs=renderChampTags(b.champions||[]);
        const traits=renderTraitTags(b.traits||[]);
        return `<div style="margin-top:6px"><strong>${b.label||('棋盘'+(b.board_idx||0))}</strong> · ${b.champion_count||0} 名英雄${champs?`<br>${champs}`:''}${traits?`<br>${traits}`:''}</div>`;
      }).join('');
      box.innerHTML=`<strong>✓ 对战识别 ${a.board_count||0} 个棋盘 / ${a.champion_count||0} 名英雄</strong>${boards}`;
      hideWelcome();toast(`✓ 已导入对战视角 · ${a.board_count||0} 个棋盘`);
      return;
    }

    const ctags=renderChampTags(a.champions||[]);
    const ttags=renderTraitTags(a.traits||[]);
    box.innerHTML=`<strong>✓ ${a.champion_count||0} 名英雄</strong><br>${ctags}${ttags?'<br>'+ttags:''}`;
    hideWelcome();toast(`✓ 已导入 ${a.champion_count||0} 名英雄`);
  }else{
    box.innerHTML=`<span style="color:var(--red)">✗ ${d.error||'失败'}</span>`;
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────
function handleChatKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}}
function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';}
function sendQ(q){document.getElementById('chatInput').value=q;sendMsg();}
async function sendMsg(){
  const inp=document.getElementById('chatInput'),q=inp.value.trim();
  if(!q||isThinking)return;
  isThinking=true;inp.value='';inp.style.height='auto';
  document.getElementById('sendBtn').disabled=true;
  hideWelcome();appendMsg('user',q);const tid=appendThinking();
  try{
    const d=await fetch('/api/recommend',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,mode:currentMode})}).then(r=>r.json());
    removeThinking(tid);appendMsg('ai',d.answer||'❌ '+(d.error||'未知错误'));
  }catch(e){removeThinking(tid);appendMsg('ai','❌ 网络错误: '+e.message);}
  isThinking=false;document.getElementById('sendBtn').disabled=false;
}
function appendMsg(role,text){
  const msgs=document.getElementById('messages'),div=document.createElement('div');
  div.className='msg '+role;
  div.innerHTML=`<div class="avatar">${role==='user'?'👤':'🤖'}</div>
    <div class="bubble">${md2html(text)}</div>`;
  msgs.appendChild(div);msgs.scrollTop=msgs.scrollHeight;
}
function appendThinking(){
  const msgs=document.getElementById('messages'),div=document.createElement('div');
  const id='tk'+Date.now();div.id=id;div.className='msg ai';
  div.innerHTML=`<div class="avatar">🤖</div><div class="bubble">
    <div class="thinking"><span></span><span></span><span></span></div></div>`;
  msgs.appendChild(div);msgs.scrollTop=msgs.scrollHeight;return id;
}
function removeThinking(id){const e=document.getElementById(id);if(e)e.remove();}
function hideWelcome(){const w=document.getElementById('welcome');if(w)w.style.display='none';}
function md2html(text){
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/^## (.+)$/gm,'<h2>$1</h2>').replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/`(.+?)`/g,'<code>$1</code>').replace(/^---$/gm,'<hr>')
    .replace(/^[-*] (.+)$/gm,'<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g,'<ul>$1</ul>')
    .replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
}

// ── Settings ───────────────────────────────────────────────────────────────
function openSettings(){document.getElementById('settingsModal').classList.add('show');}
function closeSettings(){document.getElementById('settingsModal').classList.remove('show');}
function toggleModelRow(){
  const p=document.getElementById('cfgProvider').value;
  document.getElementById('apiKeyLabel').textContent=p==='anthropic'?'Anthropic API Key':'OpenRouter API Key';
  document.getElementById('rowModel').style.display=p==='anthropic'?'none':'';
}
async function saveSettings(){
  const prov=document.getElementById('cfgProvider').value;
  const key=document.getElementById('cfgApiKey').value.trim();
  const model=document.getElementById('cfgModel').value.trim();
  const riot=document.getElementById('cfgRiotKey').value.trim();
  try{
    const d=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({provider:prov,api_key:key,model,riot_key:riot})}).then(r=>r.json());
    if(d.ok){toast('设置已保存');closeSettings();}else toast('保存失败: '+(d.error||''));
  }catch(e){toast('保存失败');}
}
document.getElementById('settingsModal').addEventListener('click',e=>{if(e.target===e.currentTarget)closeSettings();});
document.getElementById('itemModalBg').addEventListener('click',e=>{if(e.target===e.currentTarget)closeItemModal();});

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg){
  const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2200);
}
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML


@app.route("/api/kb/status")
def api_kb_status():
    return jsonify(_kb_status)


@app.route("/api/kb/refresh", methods=["POST"])
def api_kb_refresh():
    global _kb_status
    if _kb_status.get("building"):
        return jsonify({"ok": False, "message": "Already building"})
    _kb_status["building"] = True
    threading.Thread(target=_build_kb_bg, kwargs={"force": True}, daemon=True).start()
    return jsonify({"ok": True})


# ── Data endpoints ──────────────────────────────────────────────────────────
@app.route("/api/data/champions")
def api_data_champions():
    db = _load_db("champion_db_file", "tft_champion_db.json", "./tft_rag_data/tft_champion_db.json")
    set_n = _effective_set_number()
    champs = []
    for api_name, info in db.items():
        if not _current_set_only(api_name, set_n):
            continue
        short_id, name_en, name_cn = _champion_names(info, api_name, set_n)
        try:
            cost = int(info.get("cost", 1) or 1)
        except Exception:
            cost = 1
        local_url = _local_img_url("champions", api_name, name_cn, name_en, short_id)
        champs.append({
            "api_name": api_name,
            "short_id": short_id,
            "name_en": name_en,
            "name_cn": name_cn,
            "cost": cost,
            "traits": [_fix_mojibake_text(t) for t in info.get("traits", [])[:4]],
            "img_url": local_url or _champ_img(api_name),
            "local": local_url is not None,
        })
    champs.sort(key=lambda c: (c["cost"], c["name_cn"] or c["name_en"]))
    return jsonify({"ok": True, "champions": champs})

@app.route("/api/data/items")
def api_data_items():
    db = _load_db("item_db_file", "tft_item_db.json", "./tft_rag_data/tft_item_db.json")
    skip_words = {"tutorial","consumable","debug","grant","random","explorer",
                  "placeholder","elusive","support","hextech","dummy","training"}
    items = []
    for api_name, info in db.items():
        name_en, name_cn = _item_names(info, api_name)
        key = api_name.lower()
        probe = (name_cn or name_en).lower()
        if any(w in key for w in skip_words) or any(w in probe for w in skip_words):
            continue
        if not (name_cn or name_en) or (name_en == api_name and not name_cn):
            continue
        local_url = _local_img_url("items", api_name, name_cn, name_en)
        items.append({
            "api_name": api_name,
            "name_en": name_en,
            "name_cn": name_cn,
            "img_url": local_url or _item_img(api_name),
            "local": local_url is not None,
        })
    items.sort(key=lambda i: (i["name_cn"] or i["name_en"]))
    return jsonify({"ok": True, "items": items})

# ── Input endpoints ─────────────────────────────────────────────────────────
@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    provider = data.get("provider", "openrouter")
    api_key  = data.get("api_key", "").strip()
    model    = data.get("model", "").strip()
    riot_key = data.get("riot_key", "").strip()
    try:
        agent = get_agent()
        rag.CFG["llm_provider"] = provider
        if api_key:
            if provider == "anthropic":
                rag.CFG["anthropic_api_key"] = api_key
                os.environ["ANTHROPIC_API_KEY"] = api_key
            else:
                rag.CFG["openrouter_api_key"] = api_key
                os.environ["OPENROUTER_API_KEY"] = api_key
            if model:
                rag.CFG["openrouter_model"] = model
            agent.llm          = rag.LLMClient.__new__(rag.LLMClient)
            agent.llm.provider = provider
            agent.llm.api_key  = api_key
        if riot_key:
            rag.CFG["riot_api_key"]    = riot_key
            os.environ["RIOT_API_KEY"] = riot_key
            agent.crawler.api_key      = riot_key
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/input/screenshot", methods=["POST"])
def api_input_screenshot():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image file"}), 400
    img_bytes  = request.files["image"].read()
    # 前端传入的识别模式（auto/board/lineup/global/duel）
    user_mode  = request.form.get("mode", "auto").strip().lower()
    recognize_mode = user_mode or "auto"
    try:
        from tft_screen_capture_yolo_clip import recognize
        result = recognize(img_bytes, mode=recognize_mode)
    except ImportError:
        try:
            from tft_screen_capture import recognize
            result = recognize(img_bytes, mode=recognize_mode)
        except ImportError:
            return jsonify({"ok": False, "error": "tft_screen_capture 模块未找到"})
        except TypeError:
            # 旧版 recognize 不支持 mode 参数
            from tft_screen_capture import recognize as _rec
            result = _rec(img_bytes)
    except TypeError:
        from tft_screen_capture_yolo_clip import recognize as _rec
        result = _rec(img_bytes)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    if result.get("error"):
        return jsonify({"ok": False, "error": result["error"], "hint": result.get("hint","")})
    used_mode = result.get("_layout") or user_mode
    _save_analysis(result)
    return jsonify({"ok": True, "analysis": _summarize(result),
                    "raw": result, "used_mode": used_mode})
@app.route("/api/input/text", methods=["POST"])
def api_input_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Empty input"}), 400
    try:
        from tft_converter import convert
        if text.startswith(("[", "{")):
            result = convert(text, set_num=_effective_set_number())
        else:
            result = _convert_text_fuzzy(text)
            if result.get("error"):
                fallback = convert(text, set_num=_effective_set_number())
                if not fallback.get("error"):
                    result = fallback
    except ImportError:
        return jsonify({"ok": False, "error": "tft_converter.py 未找到"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    if result.get("error"):
        return jsonify({"ok": False, "error": result["error"]})
    _save_analysis(result)
    return jsonify({"ok": True, "analysis": _summarize(result), "raw": result})


@app.route("/api/input/builder", methods=["POST"])
def api_input_builder():
    """Receive roster from builder tab → convert → save → summarize."""
    data  = request.get_json(force=True)
    units = data.get("units", [])
    if not units:
        return jsonify({"ok": False, "error": "Empty roster"}), 400

    champ_db = _load_db("champion_db_file", "tft_champion_db.json",
                        "./tft_rag_data/tft_champion_db.json")
    set_n    = _effective_set_number()

    champions = []
    for u in units:
        api_name = u.get("champion_id", "")
        info     = champ_db.get(api_name, {})
        short_id = (info.get("short_id")
                    or api_name.replace(f"TFT{set_n}_","").replace("TFT_",""))
        name_en  = info.get("name_en") or short_id
        champions.append({
            "id"      : api_name,
            "short_id": short_id,
            "name_en" : name_en,
            "star"    : u.get("star", 1),
            "cost"    : info.get("cost", 0),
            "items"   : u.get("items", []),
            "position": u.get("position", {"row": 0, "col": 0}),
        })

    result = {"champions": champions, "traits": [], "summary": {}, "equipment_issues": []}
    try:
        from tft_converter import calc_traits, build_summary
        result["traits"] = calc_traits(champions)
        result["summary"], result["equipment_issues"] = build_summary(champions, [])
    except Exception:
        pass

    _save_analysis(result)
    return jsonify({"ok": True, "analysis": _summarize(result), "raw": result})


def _save_analysis(result: dict):
    try:
        path = rag.CFG.get("analysis_file", "tft_team_analysis.json")
        Path(path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        get_agent().local.reload()
    except Exception:
        pass


def _summarize(result: dict) -> dict:
    layout = result.get("_layout") or "single"

    def champ_summary(champs: list) -> list:
        return [
            {"id": c.get("id", ""), "short_id": c.get("short_id", ""),
             "name_en": c.get("name_en", ""), "star": c.get("star", 1), "cost": c.get("cost", 0)}
            for c in champs if c.get("id")
        ]

    def trait_summary(traits: list) -> list:
        rows = []
        for t in traits:
            count = int(t.get("count", 0) or 0)
            thresholds = [int(x) for x in (t.get("thresholds") or []) if str(x).isdigit() or isinstance(x, int)]
            thresholds = sorted(dict.fromkeys(thresholds))
            active = int(t.get("level", 0) or 0) > 0 or int(t.get("tier", 0) or 0) > 0
            next_threshold = next((lvl for lvl in thresholds if lvl > count), 0)
            rows.append({
                "id": t.get("id", ""),
                "name_en": t.get("name_en") or t.get("name") or t.get("short_id", ""),
                "count": count,
                "active": active,
                "level": int(t.get("level", 0) or 0),
                "level_name": t.get("level_name", "") or "",
                "thresholds": thresholds,
                "next_threshold": next_threshold,
            })
        return rows

    if layout == "global":
        players = result.get("players", []) or []
        player_rows = []
        all_names = []
        from collections import Counter
        for idx, player in enumerate(players, 1):
            champs = champ_summary(player.get("champions", []))
            traits = trait_summary(player.get("traits", []))
            player_rows.append({
                "rank": player.get("rank", idx),
                "label": player.get("label") or player.get("player_name") or player.get("name") or f"第{idx}名",
                "champion_count": len(champs),
                "champions": champs,
                "traits": traits,
            })
            all_names.extend([c.get("name_en") or c.get("short_id") for c in champs if c.get("id")])
        contested = [name for name, cnt in Counter([n for n in all_names if n]).most_common(6) if cnt >= 2]
        return {
            "layout": "global",
            "player_count": len(player_rows),
            "champion_count": sum(p["champion_count"] for p in player_rows),
            "players": player_rows,
            "contested": contested,
        }

    if layout == "duel":
        boards = result.get("boards", []) or []
        board_rows = []
        for idx, board in enumerate(boards):
            champs = champ_summary(board.get("champions", []))
            traits = trait_summary(board.get("traits", []))
            board_rows.append({
                "board_idx": board.get("board_idx", idx),
                "label": board.get("label") or ("self" if idx else "opponent"),
                "champion_count": len(champs),
                "champions": champs,
                "traits": traits,
            })
        return {
            "layout": "duel",
            "board_count": len(board_rows),
            "champion_count": sum(b["champion_count"] for b in board_rows),
            "boards": board_rows,
        }

    champs = result.get("champions", [])
    traits = result.get("traits", [])
    return {
        "layout": layout,
        "champion_count": len([c for c in champs if c.get("id")]),
        "champions": champ_summary(champs),
        "traits": trait_summary(traits),
    }


# -- Local asset routes --------------------------------------------------------

def _local_img_url(subfolder, stem, *names):
    folder = ASSETS_DIR / subfolder
    if not folder.exists():
        return None

    suffixes = (".png", ".jpg", ".jpeg", ".webp")
    candidates = []

    def add_candidate(value: str):
        value = _clean_match_text(value)
        if value and value not in candidates:
            candidates.append(value)

    add_candidate(stem)
    for name in names:
        add_candidate(name)

    for value in candidates:
        for ext in suffixes:
            exact = folder / f"{value}{ext}"
            if exact.exists():
                return f"/api/assets/img/{subfolder}/{exact.name}"

    stem_prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    for name in candidates[1:]:
        for pattern in (f"{stem}_{name}*", f"{stem_prefix}_{name}*", f"{name}*"):
            for candidate in sorted(folder.glob(pattern)):
                if candidate.suffix.lower() in suffixes:
                    return f"/api/assets/img/{subfolder}/{candidate.name}"
    return None

@app.route('/api/assets/img/<path:rel_path>')
def api_assets_img(rel_path):
    safe = Path(rel_path)
    if '..' in safe.parts:
        abort(403)
    full = ASSETS_DIR / safe
    if not full.exists():
        abort(404)
    mime, _ = mimetypes.guess_type(str(full))
    return send_file(str(full), mimetype=mime or 'image/png')


@app.route('/api/assets/champions')
def api_assets_champions():
    db    = _load_db('champion_db_file', 'tft_champion_db.json', './tft_rag_data/tft_champion_db.json')
    set_n = _effective_set_number()
    champs = []
    for api_name, info in db.items():
        if not api_name.startswith('TFT'):
            continue
        if not _current_set_only(api_name, set_n):
            continue
        short_id, name_en, name_cn = _champion_names(info, api_name, set_n)
        try:
            cost = int(info.get('cost', 1) or 1)
        except Exception:
            cost = 1
        local_url = _local_img_url('champions', api_name, name_cn, name_en, short_id)
        champs.append({
            'api_name': api_name,
            'short_id': short_id,
            'name_en' : name_en,
            'name_cn' : name_cn,
            'cost'    : cost,
            'traits'  : [_fix_mojibake_text(t) for t in info.get('traits', [])[:4]],
            'img_url' : local_url or _champ_img(api_name),
            'local'   : local_url is not None,
        })
    champs.sort(key=lambda c: (c['cost'], c['name_cn'] or c['name_en']))
    return jsonify({'ok': True, 'champions': champs, 'count': len(champs)})

@app.route('/api/assets/items')
def api_assets_items():
    db = _load_db('item_db_file', 'tft_item_db.json', './tft_rag_data/tft_item_db.json')
    skip = {'tutorial','consumable','debug','grant','random','explorer',
            'placeholder','elusive','support','dummy','training'}
    items = []
    for api_name, info in db.items():
        name_en, name_cn = _item_names(info, api_name)
        probe = (name_cn or name_en).lower()
        key = api_name.lower()
        if any(w in key for w in skip) or any(w in probe for w in skip):
            continue
        if not (name_cn or name_en) or (name_en == api_name and not name_cn):
            continue
        local_url = _local_img_url('items', api_name, name_cn, name_en)
        items.append({
            'api_name': api_name,
            'name_en' : name_en,
            'name_cn' : name_cn,
            'img_url' : local_url or _item_img(api_name),
            'local'   : local_url is not None,
        })
    items.sort(key=lambda i: (i['name_cn'] or i['name_en']))
    return jsonify({'ok': True, 'items': items, 'count': len(items)})

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json(force=True)
    q    = data.get("question", "").strip()
    mode = data.get("mode", "single")
    if not q:
        return jsonify({"error": "question required"}), 400
    try:
        answer = get_agent().recommend(q, mode=mode)
        return jsonify({"answer": answer})
    except Exception:
        return jsonify({"error": traceback.format_exc()[-400:]}), 500


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"""
╔══════════════════════════════════════╗
║   TFT 阵容顾问  ·  Web UI  v3       ║
╠══════════════════════════════════════╣
║  http://localhost:{port:<5}               ║
║  停止: Ctrl+C                        ║
╚══════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)









