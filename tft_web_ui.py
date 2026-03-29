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

import json, os, sys, threading, traceback
from pathlib import Path

from flask import Flask, request, jsonify
sys.path.insert(0, str(Path(__file__).parent))
import tft_rag_agent as rag

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ── Global state ───────────────────────────────────────────────────────────
_agent: rag.TFTRagAgent = None
_agent_lock = threading.Lock()
_kb_status  = {"ready": False, "building": False, "stats": "not initialized"}

DDRAGON_VER = "15.8.1"


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
def _load_db(cfg_key: str, *fallbacks) -> dict:
    paths = [rag.CFG.get(cfg_key, ""), *fallbacks]
    for p in paths:
        if p and Path(p).exists():
            try:
                return json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _champ_img(api_name: str) -> str:
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VER}/img/tft-champion/{api_name}.png"


def _item_img(api_name: str) -> str:
    return f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VER}/img/tft-item/{api_name}.png"


# ── HTML ───────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TFT 阵容顾问</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#09090f;--sur:#111118;--brd:#1e1e2e;--gold:#c8a84b;--gd2:#7a6230;
  --teal:#3de8c8;--td2:#1a6b5e;--red:#e84040;--txt:#d4d4e8;--dim:#5a5a7a;--r:6px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Noto Sans SC',sans-serif;font-weight:300;
  min-height:100vh;display:flex;flex-direction:column}
/* Header */
header{border-bottom:1px solid var(--brd);padding:10px 18px;display:flex;align-items:center;gap:10px;
  background:linear-gradient(90deg,#0d0d18,#111120);flex-shrink:0}
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
.drop-zone:hover,.drop-zone.over{border-color:var(--td2);color:var(--teal);background:#0b1a16}
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
.result-box{background:#0b1a16;border:1px solid var(--td2);border-radius:var(--r);
  padding:8px 10px;font-size:11px;line-height:1.7;color:var(--txt);display:none;flex-shrink:0}
.result-box.show{display:block}
.result-tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;
  margin:2px;border:1px solid var(--td2);color:var(--teal);background:#061410}
.lbl{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}

/* ── Builder ── */
.builder-search{position:relative;flex-shrink:0}
.builder-search input{width:100%;background:var(--bg);border:1px solid var(--brd);
  border-radius:var(--r);color:var(--txt);padding:7px 10px;font-size:12px;
  font-family:'Noto Sans SC',sans-serif;outline:none;transition:.2s}
.builder-search input:focus{border-color:var(--td2)}
.builder-search input::placeholder{color:var(--dim)}
.champ-dropdown{background:var(--sur);border:1px solid var(--brd);border-radius:var(--r);
  max-height:170px;overflow-y:auto;display:none;position:absolute;z-index:50;
  width:100%;top:calc(100% + 2px);left:0;box-shadow:0 4px 16px rgba(0,0,0,.5)}
.champ-dropdown.show{display:block}
.champ-item{display:flex;align-items:center;gap:8px;padding:5px 8px;cursor:pointer;
  transition:.1s;border-bottom:1px solid #1a1a28}
.champ-item:hover{background:#0b1a16}
.champ-item:last-child{border-bottom:none}
.champ-item img{width:28px;height:28px;border-radius:3px;object-fit:cover;flex-shrink:0}
.champ-item .ci-name{font-size:12px;color:var(--txt)}
.champ-item .ci-cost{font-size:10px;margin-left:auto}
.c1{color:#aaa}.c2{color:#4caf50}.c3{color:#64b5f6}.c4{color:#ce93d8}.c5{color:#f1c40f}
.roster-list{display:flex;flex-wrap:wrap;gap:5px;min-height:32px;flex-shrink:0}
.roster-chip{display:flex;align-items:center;gap:4px;background:#0b1a16;border:1px solid var(--td2);
  border-radius:4px;padding:3px 6px;cursor:pointer;font-size:10px;color:var(--teal)}
.roster-chip img{width:20px;height:20px;border-radius:2px;object-fit:cover}
.roster-chip .rc-del{font-size:9px;color:var(--red);margin-left:2px}
.roster-chip:hover{border-color:var(--red)}

/* Hex board */
.hex-board{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;flex-shrink:0}
.hex-cell{aspect-ratio:1;border:1px dashed var(--brd);border-radius:5px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;transition:.15s;background:var(--sur);overflow:hidden;
  position:relative;min-height:0;font-size:8px;color:var(--dim)}
.hex-cell:hover{border-color:var(--td2);background:#0b1a16}
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
.hex-cell:hover .remove-btn{display:block}

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
.item-thumb:hover{background:#0b1a16;border-color:var(--td2)}
.item-thumb.selected{background:#0f2c22;border-color:var(--teal)}
.item-thumb img{width:40px;height:40px;border-radius:4px;object-fit:cover}
.item-thumb span{font-size:9px;color:var(--dim);text-align:center;
  word-break:break-all;line-height:1.2;max-width:56px}
.item-modal-footer{display:flex;gap:8px;flex-shrink:0}
.item-modal-footer button{flex:1;padding:7px;border:none;border-radius:var(--r);
  cursor:pointer;font-family:Rajdhani,sans-serif;font-size:13px;font-weight:600;letter-spacing:1px}
.btn-ok{background:var(--td2);color:#fff}.btn-ok:hover{background:var(--teal);color:var(--bg)}
.btn-c2{background:transparent;border:1px solid var(--brd);color:var(--dim)}.btn-c2:hover{color:var(--txt)}

/* Right: chat */
.chat-panel{flex:1;display:flex;flex-direction:column;overflow:hidden}
.messages{flex:1;overflow-y:auto;padding:14px 18px;display:flex;
  flex-direction:column;gap:12px;scroll-behavior:smooth}
.messages::-webkit-scrollbar{width:4px}
.messages::-webkit-scrollbar-thumb{background:var(--brd);border-radius:2px}
.msg{display:flex;gap:9px;max-width:820px;animation:fadeUp .2s ease}
.msg.user{flex-direction:row-reverse;align-self:flex-end}
.msg.ai{align-self:flex-start}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.avatar{width:26px;height:26px;border-radius:4px;display:flex;align-items:center;
  justify-content:center;font-size:12px;flex-shrink:0}
.msg.user .avatar{background:#1e1e38}.msg.ai .avatar{background:#0f1f1a;border:1px solid var(--td2)}
.bubble{padding:9px 13px;border-radius:var(--r);font-size:13px;line-height:1.75;max-width:680px}
.msg.user .bubble{background:#14142a;border:1px solid #252545;color:var(--txt)}
.msg.ai   .bubble{background:#0b1a16;border:1px solid #1a3028;color:var(--txt)}
.bubble strong,.bubble b{color:var(--gold);font-weight:500}
.bubble em,.bubble i{color:var(--teal);font-style:normal}
.bubble h2,.bubble h3{color:var(--gold);font-family:Rajdhani,sans-serif;font-size:14px;letter-spacing:1px;margin:8px 0 3px}
.bubble ul,.bubble ol{padding-left:16px;margin:4px 0}
.bubble li{margin:2px 0}
.bubble code{background:#1a1a2e;padding:1px 4px;border-radius:3px;font-size:11px;color:var(--teal);font-family:monospace}
.bubble hr{border:none;border-top:1px solid var(--brd);margin:7px 0}
.bubble p{margin:3px 0}
.thinking{display:flex;gap:5px;align-items:center;padding:11px 13px}
.thinking span{width:5px;height:5px;background:var(--td2);border-radius:50%;animation:bounce 1.2s infinite}
.thinking span:nth-child(2){animation-delay:.2s}.thinking span:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:scale(1);opacity:.5}40%{transform:scale(1.4);opacity:1}}
.quick-bar{padding:0 16px 6px;display:flex;flex-wrap:wrap;gap:4px;flex-shrink:0}
.qp{padding:3px 10px;border:1px solid var(--brd);border-radius:20px;background:transparent;
  color:var(--dim);font-size:11px;cursor:pointer;font-family:'Noto Sans SC',sans-serif;
  transition:.15s;white-space:nowrap}
.qp:hover{border-color:var(--td2);color:var(--teal);background:#0b1a16}
/* Simplified chat bar */
.chat-bar{padding:8px 16px 12px;border-top:1px solid var(--brd);
  display:flex;flex-direction:column;gap:6px;background:#0b0b14;flex-shrink:0}
.chat-bar-top{display:flex;align-items:center;gap:8px}
.mode-pills{display:flex;gap:4px}
.mode-pill{padding:4px 12px;border:1px solid var(--brd);border-radius:20px;background:transparent;
  color:var(--dim);font-size:11px;cursor:pointer;font-family:'Noto Sans SC',sans-serif;
  transition:.15s;white-space:nowrap}
.mode-pill.active{border-color:var(--td2);color:var(--teal);background:#0b1a16}
.mode-pill:hover:not(.active){color:var(--txt)}
.chat-bar-row{display:flex;gap:7px;align-items:flex-end}
.chat-bar-row textarea{flex:1;background:#111118;border:1px solid var(--brd);border-radius:var(--r);
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
.welcome h2{font-family:Rajdhani,sans-serif;font-size:22px;letter-spacing:2px;color:var(--gold);font-weight:600}
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
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#0f2c22;
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
        <small>棋盘 · 结算横排 · 对局回顾</small>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
      <img class="preview-img" id="previewImg">
      <button class="inp-submit" id="btnSS" onclick="submitScreenshot()" disabled>🔍 识别阵容</button>
      <div class="result-box" id="resSS"></div>
    </div>

    <!-- Text -->
    <div class="tab-pane" id="tab-text">
      <div class="lbl">英雄ID / JSON</div>
      <textarea class="inp-textarea" id="textInput"
        placeholder="输入英雄英文 ID（逗号/空格分隔）或粘贴 JSON&#10;示例: Draven Kindred Sett Leona"></textarea>
      <button class="inp-submit" onclick="submitText()">📥 导入阵容</button>
      <div class="result-box" id="resTxt"></div>
    </div>

    <!-- Builder -->
    <div class="tab-pane" id="tab-builder">
      <div class="lbl">搜索英雄</div>
      <div class="builder-search">
        <input type="text" id="champSearch" placeholder="输入英雄名..." autocomplete="off"
               oninput="filterChamps(this.value)" onfocus="showDD()" onblur="hideDD()">
        <div class="champ-dropdown" id="champDD"></div>
      </div>
      <div class="lbl" style="margin-top:4px">
        已选 <span id="rosterCnt" style="color:var(--teal)">0</span>/9
        <span style="color:var(--dim);font-size:9px;margin-left:6px">点击英雄图标选装备</span>
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
          <button class="mode-pill active" onclick="setMode('single',this)">⚔ Single</button>
          <button class="mode-pill" onclick="setMode('duel',this)">🆚 Duel</button>
          <button class="mode-pill" onclick="setMode('global',this)">🌐 Global</button>
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

// ── Init ───────────────────────────────────────────────────────────────────
(async()=>{
  pollKB(); setInterval(pollKB,3000);
  initHexBoard();
  try{
    const [cr,ir]=await Promise.all([
      fetch('/api/data/champions').then(r=>r.json()),
      fetch('/api/data/items').then(r=>r.json()),
    ]);
    if(cr.ok){ champDB=cr.champions; renderDD(champDB); }
    if(ir.ok) itemDB=ir.items;
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
  const fd=new FormData();fd.append('image',selectedFile);
  try{
    const d=await fetch('/api/input/screenshot',{method:'POST',body:fd}).then(r=>r.json());
    showResult('resSS',d);
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
function renderDD(list){
  const dd=document.getElementById('champDD');
  dd.innerHTML=list.slice(0,60).map(c=>`
    <div class="champ-item" onmousedown="addChamp('${c.api_name}')">
      <img src="${c.img_url}" alt="" width="28" height="28"
           onerror="this.style.visibility='hidden'">
      <span class="ci-name">${c.name_en}</span>
      <span class="ci-cost c${c.cost}">${c.cost}费</span>
    </div>`).join('');
}
function filterChamps(q){
  const f=champDB.filter(c=>c.name_en.toLowerCase().includes(q.toLowerCase())||
                            c.api_name.toLowerCase().includes(q.toLowerCase()));
  renderDD(f);
  document.getElementById('champDD').classList.toggle('show',f.length>0&&q.length>0);
}
function showDD(){
  const q=document.getElementById('champSearch').value;
  if(q.length>0) document.getElementById('champDD').classList.add('show');
}
function hideDD(){ setTimeout(()=>document.getElementById('champDD').classList.remove('show'),150); }

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
  document.getElementById('rosterList').innerHTML=roster.map(c=>`
    <div class="roster-chip">
      <img src="${c.img_url}" alt="" onerror="this.style.display='none'">
      ${c.name_en.length>8?c.name_en.slice(0,7)+'…':c.name_en}
      <span class="rc-del" onclick="removeChamp('${c.api_name}')">✕</span>
    </div>`).join('');
}

// ── Hex board ──────────────────────────────────────────────────────────────
function initHexBoard(){
  const board=document.getElementById('hexBoard');board.innerHTML='';
  for(let row=0;row<4;row++) for(let col=0;col<7;col++){
    const cell=document.createElement('div');
    cell.className='hex-cell';cell.dataset.row=row;cell.dataset.col=col;
    cell.textContent=`${row},${col}`;
    cell.onclick=()=>boardClick(row,col);
    board.appendChild(cell);
  }
}
function renderBoard(){
  const cells=document.querySelectorAll('.hex-cell');
  cells.forEach(c=>{c.className='hex-cell';c.innerHTML=`<span>${c.dataset.row},${c.dataset.col}</span>`;});
  roster.forEach(champ=>{
    const idx=champ.row*7+champ.col;if(idx<0||idx>=cells.length)return;
    const cell=cells[idx];
    cell.className=`hex-cell occupied cost${champ.cost||1}`;
    const stars=champ.star>1?`<div class="star-row">${'<span class="star">★</span>'.repeat(champ.star)}</div>`:'';
    const items=(champ.items||[]).map(n=>{
      const it=itemDB.find(i=>i.api_name===n);
      return it?`<img class="item-icon" src="${it.img_url}" alt="" onerror="this.style.display='none'">`:'' ;
    }).join('');
    cell.innerHTML=`
      ${stars}
      <img class="champ-portrait" src="${champ.img_url}" alt="${champ.name_en}"
           onerror="this.style.opacity='.3'"
           onclick="event.stopPropagation();openItemModal('${champ.api_name}')">
      ${items?`<div class="item-row">${items}</div>`:''}
      <div class="champ-name">${champ.name_en}</div>
      <button class="remove-btn" onclick="event.stopPropagation();removeChamp('${champ.api_name}')">✕</button>`;
  });
}
function boardClick(row,col){
  const c=roster.find(x=>x.row===row&&x.col===col);
  if(c) openItemModal(c.api_name);
}

// ── Item modal ─────────────────────────────────────────────────────────────
function openItemModal(apiName){
  itemModalIdx=roster.findIndex(c=>c.api_name===apiName);if(itemModalIdx<0)return;
  const c=roster[itemModalIdx];itemModalSel=[...(c.items||[])];
  document.getElementById('itemModalTitle').textContent=`${c.name_en} — 装备（最多3件）`;
  document.getElementById('itemSearch').value='';
  renderItemGrid(itemDB);updateItemPrev();
  document.getElementById('itemModalBg').classList.add('show');
}
function closeItemModal(){
  document.getElementById('itemModalBg').classList.remove('show');
  itemModalIdx=-1;itemModalSel=[];
}
function filterItems(q){
  renderItemGrid(itemDB.filter(i=>i.name_en.toLowerCase().includes(q.toLowerCase())||
                                   i.api_name.toLowerCase().includes(q.toLowerCase())));
}
function renderItemGrid(list){
  document.getElementById('itemGrid').innerHTML=list.map(it=>`
    <div class="item-thumb ${itemModalSel.includes(it.api_name)?'selected':''}"
         onclick="toggleItem('${it.api_name}')">
      <img src="${it.img_url}" alt="${it.name_en}" onerror="this.style.opacity='.3'">
      <span>${it.name_en.length>10?it.name_en.slice(0,9)+'…':it.name_en}</span>
    </div>`).join('');
}
function toggleItem(apiName){
  if(itemModalSel.includes(apiName)){
    itemModalSel=itemModalSel.filter(i=>i!==apiName);
  }else{
    if(itemModalSel.length>=3){toast('最多3件装备');return;}
    itemModalSel.push(apiName);
  }
  renderItemGrid(document.getElementById('itemSearch').value
    ?itemDB.filter(i=>i.name_en.toLowerCase().includes(document.getElementById('itemSearch').value.toLowerCase()))
    :itemDB);
  updateItemPrev();
}
function updateItemPrev(){
  const p=document.getElementById('selItemsPrev');
  p.innerHTML=itemModalSel.length?itemModalSel.map(n=>{
    const it=itemDB.find(i=>i.api_name===n);
    return it?`<img src="${it.img_url}" title="${it.name_en}" onerror="this.style.display='none'" width="22" height="22" style="border-radius:3px;object-fit:cover">`:'';
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
    const a=d.analysis;
    const ctags=(a.champions||[]).map(c=>`<span class="result-tag">${c.name_en||c.short_id} ${c.star}★</span>`).join('');
    const ttags=(a.traits||[]).filter(t=>t.active||t.count>0).map(t=>
      `<span class="result-tag" style="border-color:var(--gold);color:var(--gold)">${t.name_en}(${t.count})</span>`).join('');
    box.innerHTML=`<strong>✓ ${a.champion_count} 名英雄</strong><br>${ctags}${ttags?'<br>'+ttags:''}`;
    hideWelcome();toast(`✓ 已导入 ${a.champion_count} 名英雄`);
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
    set_n = rag.CFG.get("current_set", 16)
    champs = []
    for api_name, info in db.items():
        if not (api_name.startswith(f"TFT{set_n}_") or api_name.startswith("TFT16_")):
            continue
        name = info.get("name_en") or info.get("short_id") or api_name
        cost = info.get("cost", 1)
        champs.append({
            "api_name": api_name,
            "name_en" : name,
            "cost"    : cost,
            "traits"  : info.get("traits", [])[:4],
            "img_url" : _champ_img(api_name),
        })
    champs.sort(key=lambda c: (c["cost"], c["name_en"]))
    return jsonify({"ok": True, "champions": champs})


@app.route("/api/data/items")
def api_data_items():
    db = _load_db("item_db_file", "tft_item_db.json", "./tft_rag_data/tft_item_db.json")
    SKIP_WORDS = {"tutorial","consumable","debug","grant","random","explorer",
                  "placeholder","elusive","support","hextech","dummy","training"}
    items = []
    for api_name, info in db.items():
        name = info.get("name_en", api_name)
        key  = api_name.lower()
        if any(w in key for w in SKIP_WORDS):
            continue
        if any(w in name.lower() for w in SKIP_WORDS):
            continue
        if name in ("", "???", api_name):
            continue
        items.append({
            "api_name": api_name,
            "name_en" : name,
            "img_url" : _item_img(api_name),
        })
    items.sort(key=lambda i: i["name_en"])
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
    img_bytes = request.files["image"].read()
    try:
        from tft_screen_capture import recognize
        result = recognize(img_bytes)
    except ImportError:
        return jsonify({"ok": False, "error": "tft_screen_capture.py 未找到"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    if result.get("error"):
        return jsonify({"ok": False, "error": result["error"], "hint": result.get("hint","")})
    _save_analysis(result)
    return jsonify({"ok": True, "analysis": _summarize(result), "raw": result})


@app.route("/api/input/text", methods=["POST"])
def api_input_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Empty input"}), 400
    try:
        from tft_converter import convert
        result = convert(text)
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
    set_n    = rag.CFG.get("current_set", 16)

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
    champs = result.get("champions", [])
    traits = result.get("traits", [])
    return {
        "champion_count": len([c for c in champs if c.get("id")]),
        "champions": [
            {"id": c.get("id",""), "short_id": c.get("short_id",""),
             "name_en": c.get("name_en",""), "star": c.get("star",1), "cost": c.get("cost",0)}
            for c in champs if c.get("id")
        ],
        "traits": [
            {"id": t.get("id",""), "name_en": t.get("name_en",""),
             "count": t.get("count",0), "active": t.get("level",0) > 0}
            for t in traits
        ],
    }


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
