# TFT Advisor AI Agent

这是一个本地化的《云顶之弈》辅助分析项目。它把几件事串成了一条完整链路：

1. 从 Riot / CommunityDragon 下载当前赛季数据
2. 下载英雄与装备素材图
3. 用 YOLO 检测截图中的英雄、装备、星级位置
4. 用 CLIP 对检测出的局部图片做分类
5. 把结果整理成统一 JSON
6. 结合本地知识库、竞赛数据和大模型，输出阵容分析与建议
7. 提供 Web 前端，支持截图识别、文本输入、棋盘手工搭建

如果你是第一次接触这个项目，可以把它理解成：

- `tft_data_manager.py` 负责“准备赛季数据”
- `tft_fetch_assets.py` 负责“准备识别素材”
- `tft_screen_capture_yolo_clip.py` 负责“识别截图”
- `tft_converter.py` 负责“把各种输入统一成标准格式”
- `tft_rag_agent.py` 负责“结合知识库和大模型做分析”
- `tft_web_ui.py` 负责“把这些能力放到网页里”

这份 README 已经合并了原来的 `README.md` 和 `TFT_YOLO.md`。如果你只看一个文档，就看这一份。

---

## 1. 项目能做什么

这个项目目前主要支持 3 类使用方式：

### 1. 截图识别

你给它一张 TFT 截图，它会尝试识别：

- 英雄是谁
- 英雄几星
- 英雄带了哪些装备
- 是单棋盘、双人模式还是全局排行视图

### 2. 手工输入阵容

你可以直接输入文本，例如：

```text
阿卡丽 2星 蓝盾 法爆
慎 3星 狂徒 反甲 龙牙
```

或者在前端里手工摆棋盘、选英雄、选装备。

### 3. AI 分析

项目会把识别/输入结果转成标准 JSON，再结合：

- 当前赛季英雄/羁绊/装备本地库
- 本地构建的 RAG 知识库
- Riot 抓取的竞赛/高分数据
- 大模型

最后输出阵容分析、发展方向、强弱判断和建议。

---

## 2. 适合谁看

这份文档按两种读法写：

- 你只想跑起来：重点看“快速开始”和“常用命令”
- 你想彻底看懂：继续看“技术架构”“识别原理”“YOLO 训练”“数据结构”“排错指南”

---

## 3. 项目目录说明

下面这些文件和目录最重要：

### 核心脚本

- `tft_data_manager.py`
  - 下载当前赛季 TFT 数据
  - 生成本地 JSON 数据库
- `tft_fetch_assets.py`
  - 下载英雄头像和装备图标素材
  - 为 CLIP 分类提供素材库
- `tft_screen_capture.py`
  - 旧版 OpenCV 模板识别脚本
- `tft_screen_capture_yolo_clip.py`
  - 当前主识别脚本
  - YOLO 检测位置，CLIP 负责分类
- `tft_converter.py`
  - 统一处理截图结果、文本输入、Riot JSON
  - 输出标准阵容 JSON
- `tft_rag_agent.py`
  - RAG、竞赛数据、知识库和大模型分析主入口
- `tft_web_ui.py`
  - 本地 Web 前端

### 数据与素材目录

- `tft_assets/`
  - 识别素材库
  - 包含 `champions/` 和 `items/`
- `tft_dataset/`
  - YOLO 训练数据集
- `tft_rag_data/`
  - RAG 缓存、索引、知识块、Riot 缓存等

### 关键 JSON 文件

- `tft_champion_db.json`
  - 英雄数据库
- `tft_trait_db.json`
  - 羁绊数据库
- `tft_item_db.json`
  - 装备数据库
- `tft_champion_trait_map.json`
  - 英雄到羁绊映射
- `tft_trait_champion_dict.json`
  - 羁绊到英雄映射，包含羁绊激活阈值等信息
- `tft_meta.json`
  - 当前赛季、来源、语言等元信息
- `tft_team_analysis.json`
  - 当前一次识别/输入后的标准分析数据
- `tft_assets/asset_index.json`
  - 本地素材文件名与内部 ID 的映射

### 其他常见文件

- `tft_detector.pt`
  - YOLO 检测模型权重
- `t1.jpg` / `t2.jpg`
  - 测试图片
- `t1_duel_debug.json` / `t2_global_debug.json`
  - 调试输出示例

---

## 4. 整体技术架构

项目的完整链路可以概括成下面这张逻辑图：

```text
赛季数据下载
  -> 本地英雄/羁绊/装备数据库
  -> 素材下载
  -> YOLO + CLIP 截图识别
  -> 标准化 JSON
  -> RAG + 竞赛数据 + 大模型分析
  -> Web 前端展示 / 终端输出
```

更细一点：

### A. 数据层

来源主要有两个：

- CommunityDragon
- DDragon

其中：

- CommunityDragon 更适合拿 TFT 的结构化赛季数据
- DDragon 更像兜底或部分资源补充

项目当前优先使用 CommunityDragon，并且会尝试限制只保留当前赛季数据，避免混入老赛季英雄、羁绊或装备。

### B. 识别层

截图识别不是“一个模型包打天下”，而是两段式：

- YOLO 负责定位：这个东西在哪里
- CLIP 负责分类：这个东西是谁

这样做的原因很简单：

- YOLO 擅长框位置
- CLIP 擅长在已有裁剪图上做相似度分类

### C. 结构化层

无论输入来自：

- Riot JSON
- 截图识别
- 前端文本输入
- 前端棋盘搭建

最后都会被整理成同一种 JSON 结构，方便后续分析模块统一消费。

### D. 分析层

分析并不只是把识别结果直接丢给大模型，而是尽量补充：

- 英雄费用
- 羁绊信息
- 装备信息
- 本地知识库检索结果
- Riot 竞赛/高分缓存数据

再交给大模型生成建议。

---

## 5. 快速开始

如果你完全不熟悉这个项目，建议按下面顺序来。

### 第一步：准备 Python 环境

建议 Python 3.10 到 3.13。

如果你想使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 第二步：安装基础依赖

至少需要这些：

```bash
pip install requests flask pillow numpy opencv-python ultralytics openai
```

### 第三步：安装 CLIP

这个项目的截图识别依赖 CLIP。常见安装方式：

```bash
pip install git+https://github.com/openai/CLIP.git
```

如果你的环境里 `torch` 没有自动装好，还需要先装 PyTorch。

### 第四步：下载赛季数据

```bash
python tft_data_manager.py --set 17
```

建议再做一次校验：

```bash
python tft_data_manager.py --verify
```

### 第五步：下载识别素材

```bash
python tft_fetch_assets.py
```

如果你想看哪些素材还缺失：

```bash
python tft_fetch_assets.py --list-missing
```

### 第六步：测试截图识别

```bash
python tft_screen_capture_yolo_clip.py t1.jpg --mode auto
```

### 第七步：启动前端

```bash
python tft_web_ui.py
```

浏览器打开：

```text
http://localhost:5000
```

### 第八步：测试 RAG 分析

```bash
python tft_rag_agent.py
```

如果你只想问一个问题：

```bash
python tft_rag_agent.py --question "现在这套阵容还差什么？"
```

---

## 6. 常用命令速查

### 下载当前赛季数据

```bash
python tft_data_manager.py --set 17
python tft_data_manager.py --verify
```

### 强制指定下载语言

项目当前默认倾向中文数据流。如果你想显式指定：

```bash
python tft_data_manager.py --set 17 --cdragon-locale zh_cn --ddragon-locale zh_CN
```

### 下载素材

```bash
python tft_fetch_assets.py
python tft_fetch_assets.py --list-missing
python tft_fetch_assets.py --verify
```

### 只下载英雄或装备

```bash
python tft_fetch_assets.py --champs
python tft_fetch_assets.py --items
```

### 允许退回原皮素材

默认不建议，因为会降低识别准确率：

```bash
python tft_fetch_assets.py --allow-base-skin
```

### 截图识别

```bash
python tft_screen_capture_yolo_clip.py t1.jpg --mode auto
python tft_screen_capture_yolo_clip.py t1.jpg --mode duel --debug --save t1_duel_debug.json
python tft_screen_capture_yolo_clip.py t2.jpg --mode global --debug --save t2_global_debug.json
```

### Web 前端

```bash
python tft_web_ui.py
```

### RAG 分析

```bash
python tft_rag_agent.py
python tft_rag_agent.py --refresh
python tft_rag_agent.py --mode single
python tft_rag_agent.py --mode duel
python tft_rag_agent.py --mode global
python tft_rag_agent.py --question "这局我该不该上人口？"
```

---

## 7. 标准使用流程

如果你想“正确地”用这个项目，一般流程是：

### 路线 A：识别截图再分析

1. 更新当前赛季数据
2. 更新素材库
3. 用 `tft_screen_capture_yolo_clip.py` 识别截图
4. 得到标准 JSON
5. 用 `tft_rag_agent.py` 分析

### 路线 B：前端手工输入阵容再分析

1. 启动 `tft_web_ui.py`
2. 在网页里输入文本、上传截图或手动摆盘
3. 前端把结果保存为标准 JSON
4. 后端 RAG 代理读取 JSON 分析

### 路线 C：重新训练识别模型

1. 准备 `tft_dataset/`
2. 训练 YOLO
3. 生成 `tft_detector.pt`
4. 更新本地素材
5. 重新测试识别精度

---

## 8. 数据管理模块详解

### 入口脚本

- `tft_data_manager.py`

### 它做了什么

- 从 CommunityDragon 拉取当前赛季 TFT 数据
- 如果 CommunityDragon 失败，尝试切换到 DDragon
- 只保留目标赛季的数据
- 过滤掉非可玩单位、错误条目和跨赛季内容
- 构建出本地 JSON 数据库

### 输出文件

运行后通常会生成：

- `tft_champion_db.json`
- `tft_trait_db.json`
- `tft_item_db.json`
- `tft_champion_trait_map.json`
- `tft_trait_champion_dict.json`
- `tft_meta.json`

### 你需要知道的设计点

#### 1. 为什么赛季过滤很重要

如果混入老赛季数据，会导致：

- 英雄库里出现历史单位
- 羁绊映射错乱
- 大模型引用过期阵容词汇
- 识别和分析严重偏离当前版本

#### 2. 为什么要保留中文命名

这个项目已经明显偏向中文数据流，原因是：

- 大模型会自行翻译英文羁绊名
- 自行翻译时容易“张冠李戴”
- 中文本地库更方便和前端、文本输入保持一致

#### 3. 验证命令的意义

`--verify` 不是可有可无。它可以帮助你检查：

- 文件是否缺失
- 赛季是否混入错误数据
- 映射关系是否完整

---

## 9. 素材下载模块详解

### 入口脚本

- `tft_fetch_assets.py`

### 它做了什么

- 根据本地英雄/装备数据库下载素材
- 优先下载赛季专属、适合识别的素材
- 生成 `asset_index.json` 映射文件
- 为 CLIP 建立可用的图像素材库

### 默认目录结构

```text
tft_assets/
├─ champions/
├─ items/
└─ asset_index.json
```

### 为什么默认不允许“原皮退回”

你之前遇到过一个典型问题：素材下载成了原皮，而不是赛季内的 TFT 专属形象。这样会让识别几乎失效。

所以现在的策略是：

- 优先找赛季专属素材
- 默认禁用普通 `_square.png` 的原皮退回
- 宁可缺图，也尽量不要错素材

只有在你明确接受“覆盖率大于准确率”的情况下，才建议使用：

```bash
python tft_fetch_assets.py --allow-base-skin
```

### `asset_index.json` 的作用

它不是给大模型看的，而是给程序自己做映射用的：

- 本地文件名可能是中文
- 内部识别 ID 仍然需要稳定的英文 API 名
- `asset_index.json` 负责把两者对上

---

## 10. 截图识别模块详解

### 主脚本

- `tft_screen_capture_yolo_clip.py`

### 支持模式

- `auto`
- `board`
- `lineup`
- `global`
- `duel`

### 命令行参数

```bash
python tft_screen_capture_yolo_clip.py <image> --mode auto --debug --save out.json
```

主要参数：

- `image`：截图路径
- `--mode`：识别模式
- `--debug`：输出调试图 `tft_debug.png`
- `--save`：保存结果 JSON
- `--assets-dir`：指定素材目录
- `--device`：指定 CLIP 运行设备，如 `cpu` 或 `cuda`

### 识别链路

#### 第 1 步：YOLO 检测

YOLO 负责找出位置，不负责最终类别判断。

当前推荐保留 4 类检测目标：

- `unit_box`
  - 棋盘或双人模式中的完整英雄框
- `unit_icon`
  - 全局视图或结算页中的小头像
- `item_slot`
  - 装备图标位置
- `star_pip`
  - 星级点位

#### 第 2 步：局部裁剪

脚本会根据检测框再做二次裁剪：

- 英雄框会裁掉边框、背景干扰区、星级区、装备区的一部分
- 装备框会适当扩边并增强对比度
- 小图标模式与大框模式会使用不同的裁剪策略

#### 第 3 步：CLIP 分类

CLIP 不直接看整张图，而是看每个裁好的小块图，再与本地素材库做相似度比较。

它会分别对：

- 英雄素材库
- 装备素材库

做 embedding 匹配。

#### 第 4 步：星级统计

星级不是让 CLIP 猜，而是通过 `star_pip` 检测点数量来估计，更稳定。

#### 第 5 步：统一输出 JSON

最后输出标准结构，供前端、RAG 和调试模块复用。

### 为什么双人模式容易识别差

双人模式往往比普通棋盘更难，因为：

- 两个棋盘同时出现
- 英雄尺寸更小
- 装备更密集
- 上下区域容易互相干扰

所以脚本里会先尝试把截图分成上下两个棋盘，再分别识别。

### 为什么全局模式也容易错

全局模式的英雄通常不是完整棋子，而是小头像。这个时候：

- YOLO 需要识别 `unit_icon`
- CLIP 看到的是更小、更模糊的图
- 英雄相似头像更容易混淆

这也是为什么全局模式和棋盘模式本质上是两类不同任务。

---

## 11. YOLO 训练说明

这一节是把原来 `TFT_YOLO.md` 的内容完整并入 README 后的版本。

### 训练目标是什么

YOLO 在这个项目里只负责一件事：

- 回答“这个东西在哪里”

它不负责回答：

- 这个英雄是谁
- 这个装备叫什么

后者交给 CLIP。

### 推荐检测类别

当前推荐 4 类：

- `unit_box`
- `unit_icon`
- `item_slot`
- `star_pip`

### 标注原则

#### `unit_box`

- 框完整英雄区域
- 尽量包含主要边框
- 不要带太多背景
- 主要用于 `board` / `duel`

#### `unit_icon`

- 只框头像本体
- 尽量不要把装备和星级一起框进去
- 主要用于 `global` / `lineup`

#### `item_slot`

- 每件装备单独一个框
- 空槽不要标
- 一名英雄有 3 件装备，就标 3 个框

#### `star_pip`

- 每个星点单独一个框
- 1 星标 1 个，2 星标 2 个，3 星标 3 个
- 目标很小，标注要尽量准

### 数据集目录结构

```text
tft_dataset/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

YOLO 标签格式：

```text
<class_id> <x_center> <y_center> <width> <height>
```

### `dataset.yaml` 示例

```yaml
path: ./tft_dataset
train: images/train
val: images/val

nc: 4
names:
  - unit_box
  - unit_icon
  - item_slot
  - star_pip
```

### 训练命令示例

```bash
yolo detect train \
  data=tft_dataset/dataset.yaml \
  model=yolov8s.pt \
  epochs=100 \
  imgsz=1280 \
  batch=8 \
  name=tft_detector \
  project=runs/detect
```

训练完成后，把最佳权重复制为：

```text
tft_detector.pt
```

放到项目根目录。

### 样本来源建议

建议混合这些截图：

- 普通棋盘 `board`
- 双人模式 `duel`
- 全局排行 `global`
- 结算阵容页 `lineup`

同时尽量覆盖：

- 不同分辨率
- 不同亮度
- 不同压缩质量
- 不同星级数量
- 不同装备数量

### 为什么一定要标装备和星级

因为这个项目已经决定走：

- YOLO 定位
- CLIP 分类

所以你标出的装备框和星级点，不是浪费，反而正是提高识别精度的关键。

当前正确思路就是：

1. YOLO 先给出英雄、装备、星级的位置
2. 英雄框单独裁剪再给 CLIP
3. 装备框单独裁剪再给 CLIP
4. 星级点单独统计

### 提高精度的建议

- 提高训练分辨率，尤其是 `item_slot` 和 `star_pip`
- 多补双人模式和全局模式样本
- 保证素材库是当前赛季专属皮肤，而不是原皮
- 调试时一定保留 `--debug` 输出
- 对经常混淆的英雄重点补样本
- 英雄框要减少边框和背景占比
- 装备框要尽量紧，不要框太大

---

## 12. CLIP 在这个项目里的作用

CLIP 在这里不是拿来“自由生成理解”的，而是做图像相似度分类。

### 工作方式

1. 把本地素材库图片编码成 embedding
2. 把截图裁剪块也编码成 embedding
3. 做相似度匹配
4. 选出最像的英雄或装备

### 素材库质量为什么决定上限

如果素材库本身错了，比如：

- 下载的是原皮
- 中文命名和 ID 映射错了
- 赛季图不全

那么 CLIP 再强也会分错，因为它根本没有看到正确的参考图。

所以你要把精度问题拆成两半看：

- YOLO 框得准不准
- CLIP 的参考素材对不对

---

## 13. 文本输入与标准化 JSON

### 文本输入现在支持什么

前端和 `tft_converter.py` 现在会尝试识别：

- 英雄名
- 星级
- 装备

例如：

```text
阿卡丽 2星 蓝盾 法爆
慎 3星 狂徒 反甲 龙牙
Akali 2* BlueBuff JeweledGauntlet
```

### 标准 JSON 的作用

项目里很多功能之所以能串起来，是因为最后都要落到同一种结构上。大致会包含：

- `champions`
- `traits`
- `summary`
- `equipment_issues`
- `_source`
- 某些模式下还会包含 `players` 或 `boards`

这样做的好处是：

- 前端、截图识别、文本输入、RAG 共用同一套数据模型
- 排错时可以直接看 JSON
- 后续接别的模型或接口也容易

---

## 14. RAG 与大模型分析模块

### 主脚本

- `tft_rag_agent.py`

### 它负责什么

它不是单纯把一句提示词扔给大模型，而是会尽量利用：

- 当前分析文件 `tft_team_analysis.json`
- 本地英雄/羁绊/装备数据库
- RAG 知识块
- Riot 缓存数据
- 多个分析代理的结果

### 你需要知道的关键点

#### 1. 它依赖本地数据，不应该让大模型自己胡猜

大模型对 TFT 的先验经常是过期的，所以项目必须尽量把：

- 当前赛季英雄池
- 羁绊名称
- 装备名称
- 费用
- 激活阈值
- 竞赛数据

都作为上下文明确提供给它。

#### 2. 它有多模式分析

命令行里支持：

- `single`
- `duel`
- `global`

这三种模式本质上输入信息量不同：

- `single`：只看一个阵容
- `duel`：看两个棋盘对位
- `global`：看整局多个玩家

#### 3. 全局模式不应该靠手写分数公式瞎估

理想状态应该是：

- 基于当前玩家识别结果
- 结合竞赛数据和知识库
- 让多个分析代理分别看不同维度
- 再综合成最终结论

也就是说，“结构化数据 + 检索上下文 + 多代理推理”才是这个模块的正确方向。

### 命令行参数

```bash
python tft_rag_agent.py --question "这套阵容缺什么"
python tft_rag_agent.py --mode global
python tft_rag_agent.py --refresh
```

---

## 15. Web 前端模块

### 主脚本

- `tft_web_ui.py`

### 前端支持的输入方式

- 上传截图
- 文本输入
- 手动棋盘搭建

### Web 端背后做了什么

- 读取本地英雄/装备数据库
- 提供 `/api/data/champions` 和 `/api/data/items`
- 接收截图、文本或棋盘输入
- 调用识别与转换逻辑
- 保存标准分析 JSON
- 调用 RAG 代理做进一步分析

### 为什么有时“终端能识别，前端却失败”

常见原因有：

- 前端传的模式和终端命令不一致
- 前端走了另一条旧逻辑
- 前端没有把同样的参数传给识别脚本
- 浏览器拿到的是错误 JSON 或接口异常

所以排查时一定要同时看：

- 终端输出
- 前端网络请求返回
- `tft_team_analysis.json`
- 调试图片和调试 JSON

---

## 16. 常见输出文件说明

### `tft_team_analysis.json`

这是整个项目里最重要的中间文件之一。很多后续分析都依赖它。

你可以把它看成：

- 这一次识别/输入最终被系统理解成什么

### `t1_duel_debug.json` / `t2_global_debug.json`

这些调试文件适合用来回答：

- 识别到了哪些英雄
- 哪些是重复识别
- 哪些英雄完全漏识别
- 双人/全局模式切分是否正确

### `tft_debug.png`

这个图很重要。它通常能直接告诉你：

- YOLO 框画歪了没有
- 装备框有没有落到正确位置
- 星级点有没有框到
- 双人模式是不是切错棋盘了

---

## 17. 推荐的调试顺序

如果项目结果不对，不要一上来就怪大模型。建议按这个顺序排查：

### 第 1 层：赛季数据对不对

检查：

- `tft_champion_db.json`
- `tft_trait_db.json`
- `tft_item_db.json`
- `tft_meta.json`

先确认没有混入老赛季数据。

### 第 2 层：素材对不对

检查：

- `tft_assets/champions/`
- `tft_assets/items/`
- `tft_assets/asset_index.json`

先确认素材不是原皮、不是旧赛季、不是错名。

### 第 3 层：YOLO 框对不对

运行：

```bash
python tft_screen_capture_yolo_clip.py t1.jpg --mode duel --debug --save t1_duel_debug.json
```

看：

- `tft_debug.png`
- `t1_duel_debug.json`

### 第 4 层：CLIP 分类对不对

如果框是对的但英雄名还是乱，说明更可能是：

- 素材错了
- 裁剪区域不对
- 英雄之间太像，缺训练样本或素材区分度不足

### 第 5 层：标准 JSON 对不对

看：

- `tft_team_analysis.json`

确认：

- 英雄名
- 星级
- 装备
- 羁绊
- 模式

都已经结构化正确。

### 第 6 层：RAG / 大模型对不对

如果前面都对，但最终建议还是胡说，那么再看：

- 当前传给模型的上下文够不够
- 本地知识库是否刷新
- 竞赛数据是否过期
- 模型是否还在引用旧赛季先验

---

## 18. 常见问题

### Q1：`tft_data_manager.py` 报代理或 SSL 错误怎么办？

你之前碰到过类似：

- `ProxyError`
- `SSLEOFError`
- 连接 `raw.communitydragon.org` 或 `ddragon.leagueoflegends.com` 失败

优先排查：

- 你的系统代理是否异常
- 终端环境变量里是否配置了错误代理
- 网络是否能直连这些站点

### Q2：为什么下载到的素材是原皮，不是赛季专属皮肤？

如果素材源没给到合适路径，或者你启用了基底退回，就可能出现这个问题。现在默认已经尽量避免原皮退回，但如果素材源本身缺失，仍然可能有少量不完整情况。

### Q3：为什么英雄识别大概只有 50% 准确率？

常见原因：

- YOLO 框不准
- 素材不是赛季专属皮肤
- 双人模式或全局模式样本不足
- CLIP 看到的裁剪图过小、过糊或背景太多
- 很多英雄头像本来就相似

### Q4：为什么装备识别几乎全错？

装备比英雄更小，难度更高。通常要重点检查：

- `item_slot` 标注质量
- 训练分辨率
- 裁剪扩边策略
- 本地装备素材是否齐全

### Q5：羁绊会自动计算吗？

会。它依赖：

- 英雄到羁绊映射
- 羁绊数据库
- 激活阈值信息

前提是本地数据文件本身正确。

### Q6：为什么大模型还会说老赛季词汇？

这通常不是“模型学不会”，而是：

- 上下文没把当前赛季数据喂够
- 识别结果本身就错了
- 本地库混入了旧赛季数据
- Riot/RAG 缓存没刷新

---

## 19. 建议的项目维护习惯

每次大版本或新赛季切换时，建议你这样做：

1. 重新运行 `tft_data_manager.py --set <赛季号>`
2. 运行 `--verify`
3. 重新运行 `tft_fetch_assets.py`
4. 检查 `asset_index.json`
5. 用几张典型截图做识别测试
6. 刷新 RAG：`python tft_rag_agent.py --refresh`
7. 再去看前端和大模型输出

这样能最大程度避免“底层数据都变了，上层还在沿用旧缓存”的问题。

---

## 20. 给新人的最短上手路线

如果你只想最少步骤跑起来，看这里：

### 路线 1：最短可运行

```bash
pip install requests flask pillow numpy opencv-python ultralytics openai
pip install git+https://github.com/openai/CLIP.git
python tft_data_manager.py --set 17
python tft_fetch_assets.py
python tft_screen_capture_yolo_clip.py t1.jpg --mode auto
python tft_web_ui.py
```

### 路线 2：最短可分析

```bash
python tft_data_manager.py --set 17
python tft_fetch_assets.py
python tft_rag_agent.py --refresh
python tft_rag_agent.py
```

### 路线 3：最短可训练

```bash
yolo detect train data=tft_dataset/dataset.yaml model=yolov8s.pt epochs=100 imgsz=1280 batch=8 name=tft_detector project=runs/detect
```

训练完把权重放成：

```text
tft_detector.pt
```

---

## 21. 最后一句话：怎么理解这个项目

这个项目不是“一个脚本识别图片”那么简单，它实际上是一个多层系统：

- 底层是赛季数据和素材数据
- 中层是 YOLO + CLIP 识别和标准化 JSON
- 上层是 RAG、竞赛数据和大模型分析
- 外层是前端交互与调试工具

所以当结果不对时，最重要的不是盯着最后一句 AI 回复，而是顺着链路往前找：

- 数据对不对
- 素材对不对
- 检测框对不对
- 分类对不对
- JSON 对不对
- 最后才是提示词和大模型

只要你按这个顺序排查，这个项目其实是可维护、可迭代、可持续提升精度的。
