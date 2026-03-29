# TFT 阵容顾问（AI Agent）

基于 Python + 大语言模型的云顶之弈智能分析工具。支持截图识别、文字/JSON 导入、可视化棋盘三种阵容输入方式，通过 RAG 检索增强生成，结合 KR 服高端局数据给出阵容评价与发展建议。

---

## 项目架构

```
tft_advisor/
├── tft_web_ui.py              # Web 界面主程序（Flask 单文件前端）
├── tft_rag_agent.py           # RAG Agent + 多智能体协同分析
├── tft_screen_capture.py      # 截图识别引擎（OpenCV 模板匹配）
├── tft_converter.py           # 阵容格式转换 + 羁绊计算
├── tft_data_manager.py        # 赛季数据自动拉取（CommunityDragon/DDragon）
├── tft_fetch_assets.py        # 模板图片下载（英雄头像 + 装备图标）
│
├── tft_assets/                # 模板图片目录（fetch_assets 后自动生成）
│   ├── champions/             #   英雄头像 (TFT16_Draven.png ...)
│   └── items/                 #   装备图标 (TFT_Item_Deathblade.png ...)
│
├── tft_champion_db.json       # 英雄完整数据（data_manager 生成）
├── tft_trait_db.json          # 羁绊激活阈值
├── tft_item_db.json           # 装备数据
├── tft_champion_trait_map.json # apiName → [trait_short_id]
├── tft_trait_champion_dict.json # short_trait_id → {champions, activation}
├── tft_meta.json              # 版本元信息
├── tft_team_analysis.json     # 当前阵容（导入后生成）
│
└── tft_rag_data/              # RAG 知识库（运行时自动生成）
    ├── kb_chunks.json
    ├── kb_idf.json
    └── riot_cache.json        # Riot API 缓存（12 小时有效）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install flask requests opencv-python pillow
```

### 2. 获取赛季数据

```bash
# 从 CommunityDragon 拉取 Set 16 英雄/羁绊/装备数据
python tft_data_manager.py --set 16

# 验证文件完整性
python tft_data_manager.py --verify
```

### 3. 下载模板图片（截图识别必须）

```bash
# 下载全部英雄头像 + 装备图标
python tft_fetch_assets.py

# 检查缺失文件
python tft_fetch_assets.py --list-missing
```

### 4. 配置 API Key

在 `tft_rag_agent.py` 顶部的 `CFG` 字典中填写，或使用环境变量：

```bash
export OPENROUTER_API_KEY="sk-or-v1-你的Key"  # LLM（必填）
export RIOT_API_KEY="RGAPI-你的Key"            # 高端局数据（可选）
```

也可以在启动 Web 界面后，点击右上角 **⚙ Settings** 在线填写，无需重启。

> **获取 Key：**
> - OpenRouter（免费）：https://openrouter.ai/keys
> - Riot API：https://developer.riotgames.com/

### 5. 启动 Web 界面

```bash
python tft_web_ui.py
```

浏览器打开 **http://localhost:5000**

---

## 功能说明

### 阵容输入（左侧面板）

| 标签 | 说明 |
|------|------|
| 📷 截图 | 上传游戏截图，OpenCV 模板匹配自动识别英雄和装备 |
| 💬 文字 | 粘贴英雄英文 ID（逗号/空格分隔）或 Riot JSON |
| ⊞ 棋盘 | 4×7 棋盘手动点选英雄位置 |

### 三种分析模式

| 模式 | 说明 |
|------|------|
| ⚔ Single | 单人模式：分析自身阵容构成与优化路径 |
| 🆚 Duel | 对局模式：分析双方阵容对抗关系与站位策略 |
| 🌐 Global | 全局模式：基于八人环境评估整体局势 |

### 多智能体协同分析

系统将分析任务拆分为三个子 Agent，各自独立建模后由 LLM 融合：

- **EconomyAgent**：金币运营、升级节奏、连胜/连败策略
- **PowerAgent**  ：羁绊激活、装备分配、升阶提示
- **PositionAgent**：前后排比例、站位建议、主C判断

### 知识库（RAG）

知识库基于 **KR 服 Challenger + Grandmaster** 近期对局数据，通过 Riot TFT API 实时采集：

- `tft-league-v1` → 高端局排行榜
- `tft-summoner-v1` → PUUID 解析
- `tft-match-v1` → 对局详情（英雄/羁绊/奥金/名次）

数据缓存 12 小时自动更新，支持 **⟳ Refresh KB** 手动刷新。

---

## 截图识别说明

### 支持的截图类型

| 类型 | 检测方法 |
|------|---------|
| 对局中棋盘 | 检测彩色六边形边框（青/紫/金/蓝） |
| 结算/回顾横排 | Canny 边缘 + 水平对齐检测 |

### 识别流程

```
截图 → 检测六边形边框 → 裁剪英雄区域
     → 颜色直方图粗筛（前 10 候选）
     → 灰度模板 NCC 精匹配 → 确定英雄 ID
     → 检测星点数量 → 装备区域匹配 → 推断棋盘坐标
```

### 标定模式（识别不准时）

```bash
# 找最佳阈值（已知截图中有哪些英雄时使用）
python tft_screen_capture.py screenshot.png --calibrate --known Draven Kindred Leona

# 输出标注图查看匹配框
python tft_screen_capture.py screenshot.png --debug --threshold 0.50
```

---

## 赛季更新

新赛季上线时：

```bash
# 1. 在 tft_fetch_assets.py 的 CHAMPIONS 列表末尾添加新英雄英文 ID
# 2. 重新拉取赛季数据
python tft_data_manager.py --set 17

# 3. 重新下载模板图片
python tft_fetch_assets.py
```

---

## 配置参考

主要配置在 `tft_rag_agent.py` 的 `CFG` 字典：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `current_set` | `16` | 当前赛季编号 |
| `llm_provider` | `"openrouter"` | `openrouter` 或 `anthropic` |
| `openrouter_model` | `deepseek/deepseek-chat-v3-0324:free` | LLM 模型 |
| `riot_tiers` | `("challenger","grandmaster")` | 数据来源段位 |
| `riot_max_players` | `30` | 每段位取前 N 名玩家 |
| `riot_matches_per_player` | `20` | 每玩家拉取对局数 |
| `cache_ttl_hours` | `12` | 缓存有效时间（小时） |
| `top_k` | `6` | RAG 检索返回文档数 |

---

## CLI 使用

不启动 Web 界面，也可以直接用命令行：

```bash
# 交互模式
python tft_rag_agent.py

# 单次提问
python tft_rag_agent.py --question "如何从现在的阵容过渡到德莱厄斯流"

# 对局模式
python tft_rag_agent.py --question "对面有瑟提前排，我怎么站位" --mode duel

# 强制刷新知识库
python tft_rag_agent.py --refresh

# 截图识别
python tft_screen_capture.py screenshot.png
python tft_screen_capture.py screenshot.png --debug
python tft_screen_capture.py screenshot.png --save result.json
```

---

## 依赖

```
flask
requests
opencv-python
pillow
```

可选（ORB 特征点匹配，识别更精准）：
```
opencv-contrib-python
```

---

## 数据来源

| 来源 | 用途 |
|------|------|
| [Riot TFT API](https://developer.riotgames.com/apis) | KR 服高端局对局数据 |
| [CommunityDragon](https://raw.communitydragon.org) | 赛季英雄/羁绊/装备数据 |
| [Data Dragon](https://ddragon.leagueoflegends.com) | 模板图片 |
| [OpenRouter](https://openrouter.ai) | LLM 推理（支持 DeepSeek/Gemini/Llama 免费模型） |

---

## 常见问题

**Q: 截图识别结果全是 unknown**

A: 模板图片未下载。运行 `python tft_fetch_assets.py`，然后用 `--calibrate` 找适合该截图的阈值。

**Q: 知识库一直显示 Building KB...**

A: Riot API Key 未设置或无效。在 ⚙ Settings 填入 `RGAPI-...` 格式 Key，再点 ⟳ Refresh KB。

**Q: 没有 Riot Key 还能用吗？**

A: 可以。没有 Key 时跳过高端局数据采集，仅使用本地羁绊词典作为知识库，LLM 分析仍正常工作。

**Q: 如何切换到 Anthropic Claude？**

A: 在 Settings 面板选 Anthropic，填入 `sk-ant-...` Key；或在 `tft_rag_agent.py` 中设置 `"llm_provider": "anthropic"`。
