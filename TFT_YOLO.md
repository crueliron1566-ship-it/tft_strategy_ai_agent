# TFT YOLO 检测器训练指南

本文件说明如何为 `tft_screen_capture_yolo_clip.py` 准备训练数据、标注标签，
并训练 `tft_detector.pt` 检测权重。

---

## 一、类别设计（4 个检测目标）

| 类别 ID | 名称        | 用途                         | 适用模式       |
| ------- | ----------- | ---------------------------- | -------------- |
| 0       | `unit_box`  | 带六边形彩色边框的英雄头像框 | board、duel    |
| 1       | `unit_icon` | 无边框的小型英雄图标         | lineup、global |
| 2       | `item_slot` | 装备图标（英雄框下方或上方） | 所有模式       |
| 3       | `star_pip`  | 单颗星星点（1~3 颗/英雄）    | 所有模式       |

> **关键原则**：YOLO 只负责"在哪里"，不区分是哪个英雄/装备。
> 英雄/装备的具体名称由 CLIP 在后续步骤中识别。

---

## 二、数据采集

### 2.1 截图来源

收集以下四种截图，每种各 150~300 张（共约 600~1200 张）：

| 截图类型 | 来源                     | 特点                            |
| -------- | ------------------------ | ------------------------------- |
| board    | DataTFT / 阵容模拟器截图 | 4×7 六边形网格，有彩色边框      |
| lineup   | 游戏内结算页横排英雄图   | 英雄水平一字排列，无边框        |
| global   | 游戏内阵容羁绊页全局视图 | 8行，每行若干小图标，含星级装备 |
| duel     | 游戏内战绩回顾双棋盘     | 上下两个棋盘叠放                |

**采集建议**：

- 分辨率覆盖常见手机（2400×1080、2778×1284、1920×1080）
- 包含不同赛季版本（英雄外观略有不同）
- 包含 1星/2星/3星 英雄（星级分布均匀）
- 包含装备数量 0/1/2/3 件的英雄

### 2.2 数据目录结构

```
tft_dataset/
├── images/
│   ├── train/      # 80% 数据
│   └── val/        # 20% 数据
└── labels/
    ├── train/      # 与 images/train 对应的 .txt 标注
    └── val/
```

---

## 三、标注规范（YOLO 格式）

每张图对应一个 `.txt` 文件，每行代表一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

所有值均为相对图片宽高的归一化值（0~1）。

### 3.1 unit_box（class 0）— 六边形英雄框

标注英雄头像的**外接矩形**，包住整个六边形框。

```
示意：
┌──────────────┐
│  ╱──────╲   │  ← 六边形边框（青/紫/金/蓝色）
│ ╱ 英雄头像 ╲  │
│ ╲          ╱  │
│  ╲────────╱   │
└──────────────┘
   ↑ 标注这整个矩形（含六边形边框外的空白）
```

**注意**：

- 框要略大于六边形本身（包含 5~8px 边距）
- 不要框进装备区域（装备在框下方，单独标注）
- 星星点在框上方，也单独标注

### 3.2 unit_icon（class 1）— 小图标（lineup/global）

标注英雄头像的实际矩形区域。

```
示意（global 模式一行）：
┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
│英雄│ │英雄│ │英雄│ │英雄│ │英雄│
└───┘ └───┘ └───┘ └───┘ └───┘
  ★★   ★     ★★★   ★     ★★
 [装][装]  [装][装][装]  ...
```

- 只框头像本身，不含上方星级和下方装备

### 3.3 item_slot（class 2）— 装备图标

每件装备单独标注为一个矩形。

**board/duel 模式**：装备在英雄框正下方

```
┌────────────┐  ← unit_box
│  英雄头像  │
└────────────┘
┌──┐┌──┐┌──┐   ← 每件装备单独标注
│装│ │装│ │装│
└──┘└──┘└──┘
```

**global 模式**：装备在小图标下方（有时上方），需根据实际截图标注。

**注意**：

- 装备图标通常正方形，边长约为英雄框高度的 20~30%
- 空装备槽（灰色空白方块）不需要标注
- 只标注有内容的装备

### 3.4 star_pip（class 3）— 单颗星星

每颗星星单独标注，每个英雄最多 3 颗。

```
  ★  ★  ★      ← 三颗星各标注为一个小矩形
┌────────────┐
│  英雄头像  │
└────────────┘
```

- 星星较小（约 10~20px），标注矩形要紧贴星星本体
- 1星英雄只有 1 颗，2星 2 颗，3星 3 颗

---

## 四、标注工具推荐

### 方案 A：Label Studio（推荐，支持团队协作）

```bash
pip install label-studio
label-studio start
```

1. 新建项目，选择 Object Detection with Bounding Boxes
2. 设置标签：`unit_box`, `unit_icon`, `item_slot`, `star_pip`
3. 导入图片后标注，导出为 YOLO 格式

### 方案 B：Roboflow（在线，操作简单）

1. 访问 https://roboflow.com，创建项目
2. 上传截图，在线框选标注
3. 导出时选择 "YOLOv8" 格式，会自动生成 `data.yaml`

### 方案 C：labelImg（本地离线）

```bash
pip install labelImg
labelImg
```

- 打开图片目录，切换到 YOLO 格式
- 手动框选，快捷键 W = 新建框

---

## 五、标注数量建议

| 类别      | 建议标注数量 | 说明                          |
| --------- | ------------ | ----------------------------- |
| unit_box  | 2000+ 个     | 每张 board 图约 7~10 个框     |
| unit_icon | 1500+ 个     | lineup 每图 8 个，global 更多 |
| item_slot | 1500+ 个     | 空装备槽不标，有效装备即可    |
| star_pip  | 2000+ 个     | 每英雄 1~3 颗，数量较多       |

**最低可训练数量**（效果一般）：每类 500 个实例。
**推荐数量**：每类 1500+ 实例，模型才能稳定泛化到不同分辨率。

---

## 六、dataset.yaml 配置文件

保存为 `tft_dataset/dataset.yaml`：

```yaml
# TFT 检测数据集配置
path: ./tft_dataset # 数据集根目录
train: images/train
val: images/val

nc: 4 # 类别数量
names:
  - unit_box # 0
  - unit_icon # 1
  - item_slot # 2
  - star_pip # 3
```

---

## 七、模型训练

### 7.1 安装依赖

```bash
pip install ultralytics torch torchvision
```

### 7.2 训练命令

```bash
# 标准训练（CPU 或单 GPU）
yolo detect train \
    data=tft_dataset/dataset.yaml \
    model=yolov8s.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=tft_detector \
    project=runs/detect

# 训练完成后最佳权重路径：
# runs/detect/tft_detector/weights/best.pt
# → 复制为 tft_detector.pt 放到脚本同目录
cp runs/detect/tft_detector/weights/best.pt ./tft_detector.pt
```

### 7.3 推荐模型选择

| 模型     | 速度 | 精度 | 适用场景                  |
| -------- | ---- | ---- | ------------------------- |
| yolov8n  | 最快 | 一般 | 手机/低算力，图标较大时   |
| yolov8s  | 快   | 较好 | **推荐**，平衡速度与精度  |
| yolov8m  | 中等 | 好   | GPU 充足时，global 小图标 |
| yolov11s | 快   | 更好 | 最新架构，优先选择        |

### 7.4 训练技巧

**针对小目标（star_pip、item_slot）**：

```bash
# 提高输入分辨率，star_pip 约 15px 在 640 下太小
yolo detect train ... imgsz=1280 batch=8

# 或使用 SAHI（切片辅助超分推理）
pip install sahi
```

**数据增强建议**（在 `dataset.yaml` 中或 ultralytics 配置）：

```yaml
# 针对 TFT 截图的增强策略
hsv_h: 0.015 # 色调微扰（模拟不同设备色彩）
hsv_s: 0.3 # 饱和度
hsv_v: 0.3 # 明度（模拟不同亮度设置）
degrees: 0 # 不旋转（TFT 截图不会旋转）
translate: 0.05
scale: 0.3 # 缩放（模拟不同分辨率）
fliplr: 0.0 # 不水平翻转（会破坏棋盘坐标语义）
mosaic: 0.5
```

---

## 八、CLIP 候选列表准备

YOLO 训练完后，CLIP 分类只需维护英雄/装备的 PNG 资产（用于构建文本 prompt）：

### 8.1 英雄头像下载（来自 CommunityDragon）

```python
# tft_fetch_assets.py（原版脚本已实现，继续复用即可）
# 英雄头像下载后放入：tft_assets/champions/TFT16_Draven.png
# 装备图标放入：tft_assets/items/BFSword.png
```

### 8.2 更新新赛季英雄

新赛季只需：

1. 下载新英雄的 PNG 头像到 `tft_assets/champions/`
2. 重启脚本，CLIP 会自动重建文本嵌入
3. **不需要重新训练 YOLO**（YOLO 只检测"英雄框"这一类别）

---

## 九、常见问题

**Q: YOLO 总是漏检 star_pip（小星星）？**

- 提高输入分辨率到 1280（`imgsz=1280`）
- 增加 star_pip 的标注数量（建议 3000+）
- 降低推理置信度阈值（`YOLO_CONF_STAR = 0.25`）

**Q: global 模式小图标检测混乱？**

- global 截图图标更密集，建议单独训练一个小模型专门处理 global 模式
- 或者用 SAHI（Slicing Aided Hyper Inference）将图片切片后推理

**Q: CLIP 识别错误率高？**

- 检查 CLIP 候选 prompt 是否与图片内容匹配（`--debug` 查看裁剪区域）
- 对 CLIP 做 fine-tune：使用少量标注过的英雄截图做 linear probe 微调分类头
- 降低 `CLIP_CONF_MIN` 阈值，观察 top-3 候选

**Q: 装备识别偏移（装备配给了错误英雄）？**

- 检查 `_items_for_unit` 中的 X 轴重叠范围（`± 0.5 * uw`），可根据实际截图调整
- global 模式的 `search_above=True` 逻辑需要根据实际装备在图标上方还是下方来调整

---

## 十、快速验证

训练完成后，运行以下命令快速验证：

```bash
# 单张截图验证（带 debug 图）
python tft_screen_capture_yolo_clip.py your_screenshot.png --debug

# 验证 global 模式
python tft_screen_capture_yolo_clip.py global_screenshot.png --mode global

# 保存 JSON 结果
python tft_screen_capture_yolo_clip.py board.png --save result.json
```

查看 `tft_debug.png`，应看到：

- 绿色框 = 成功识别的英雄
- 红色框 = 检测到但 CLIP 置信度不足的英雄
- 蓝色小框 = 装备图标
- 黄色小点 = star_pip
