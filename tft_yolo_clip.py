#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tft_yolo_clip.py
TFT 截图识别引擎（YOLO + CLIP 双阶段识别）

识别流程：
  1. YOLOv8 检测英雄位置（bounding box）
  2. CLIP 零样本分类识别英雄身份
  3. 装备检测（可选 YOLO 或模板匹配）

优势：
  - YOLO：精准定位英雄位置，不受六边形边框颜色限制
  - CLIP：语义理解能力强，对新英雄/皮肤泛化性好
  - 无需大量标注数据，CLIP 支持零样本识别

用法:
  # 训练 YOLO 模型（需准备标注数据）
  python tft_yolo_clip.py --train --data ./yolo_data.yaml --epochs 100
  
  # 识别截图
  python tft_yolo_clip.py screenshot.png
  python tft_yolo_clip.py screenshot.png --mode yolo_clip
  python tft_yolo_clip.py screenshot.png --detect-only  # 仅 YOLO 检测
  python tft_yolo_clip.py screenshot.png --debug        # 显示检测结果
"""

import cv2
import numpy as np
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import time

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
ASSETS_DIR       = Path("./tft_assets")
CHAMP_DIR        = ASSETS_DIR / "champions"
ITEM_DIR         = ASSETS_DIR / "items"
MODEL_DIR        = Path("./tft_yolo_models")
YOLO_MODEL_PATH  = MODEL_DIR / "tft_champion_det.pt"  # YOLO 检测模型
CLIP_MODEL_NAME  = "ViT-B/32"  # CLIP 模型版本：ViT-B/32, ViT-L/14, RN50

# 识别阈值
DETECT_CONF      = 0.25  # YOLO 检测置信度阈值
IOU_THRESHOLD    = 0.45  # NMS IoU 阈值
CLIP_THRESHOLD   = 0.50  # CLIP 相似度阈值

# TFT16 英雄列表（用于 CLIP 零样本分类）
# 从 tft_champion_db.json 动态加载，此处为备用
DEFAULT_CHAMPIONS = [
    "Aatrox", "Ahri", "Akali", "Ashe", "AurelionSol", "Bard",
    "Brand", "Braum", "Cassiopeia", "ChoGath", "Darius", "Diana",
    "Draven", "Ekko", "Elise", "Evelynn", "Ezreal", "Fiora",
    "Galio", "Garen", "Gnar", "Gragas", "Graves", "Gwen",
    "Hecarim", "Heimerdinger", "Illaoi", "Irelia", "Janna", "JarvanIV",
    "Jax", "Jayce", "Jinx", "Kaisa", "Kalista", "Karma",
    "Katarina", "Kayle", "Kennen", "KogMaw", "LeBlanc", "LeeSin",
    "Leona", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite",
    "Maokai", "MasterYi", "MissFortune", "Mordekaiser", "Morgana", "Nami",
    "Nautilus", "Neeko", "Nocturne", "Nunu", "Olaf", "Orianna",
    "Poppy", "Pyke", "Qiyana", "Rakan", "RekSai", "Renekton",
    "Riven", "Rumble", "Sejuani", "Sett", "Shen", "Shyvana",
    "Singed", "Sion", "Sivir", "Soraka", "Swain", "Syndra",
    "TahmKench", "Talon", "Taric", "Teemo", "Thresh", "Tristana",
    "Trundle", "Tryndamere", "TwistedFate", "Twitch", "Varus", "Vayne",
    "Veigar", "VelKoz", "Vi", "Viktor", "Vladimir", "Volibear",
    "Warwick", "Wukong", "Xayah", "XinZhao", "Yasuo", "Yone",
    "Yorick", "Yuumi", "Zac", "Zed", "Ziggs", "Zilean", "Zoe", "Zyra"
]

# ──────────────────────────────────────────────────────────────
# 全局缓存
# ──────────────────────────────────────────────────────────────
_yolo_model = None
_clip_model = None
_clip_preprocess = None
_champion_names = []
_templates_loaded = False


def _load_champion_names() -> List[str]:
    """从 JSON 数据库加载英雄名称列表"""
    global _champion_names
    if _champion_names:
        return _champion_names
    
    db_path = Path("./tft_champion_db.json")
    if db_path.exists():
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                champ_db = json.load(f)
                # 提取 apiName，去除 TFT16_ 前缀
                names = []
                for key, value in champ_db.items():
                    api_name = value.get("apiName", key)
                    # 移除前缀和数字后缀
                    clean_name = api_name.replace("TFT16_", "").replace("TFT_", "")
                    if clean_name and clean_name not in names:
                        names.append(clean_name)
                if names:
                    _champion_names = sorted(names)
                    return _champion_names
        except Exception as e:
            print(f"[WARN] 加载英雄数据库失败：{e}")
    
    _champion_names = DEFAULT_CHAMPIONS.copy()
    return _champion_names


def _load_yolo_model(model_path: Optional[Path] = None):
    """加载 YOLO 检测模型"""
    global _yolo_model
    
    if _yolo_model is not None:
        return _yolo_model
    
    try:
        from ultralytics import YOLO
        
        # 优先使用自定义训练模型
        if model_path and model_path.exists():
            print(f"[YOLO] 加载自定义模型：{model_path}")
            _yolo_model = YOLO(str(model_path))
        else:
            # 使用预训练模型（需后续 fine-tune）
            print(f"[YOLO] 使用预训练 YOLOv8n 模型（建议训练专用模型）")
            _yolo_model = YOLO("yolov8n.pt")
        
        return _yolo_model
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请运行：pip install ultralytics")
        return None
    except Exception as e:
        print(f"[ERROR] 加载 YOLO 模型失败：{e}")
        return None


def _load_clip_model():
    """加载 CLIP 模型"""
    global _clip_model, _clip_preprocess
    
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    
    try:
        import clip
        import torch
        
        print(f"[CLIP] 加载模型：{CLIP_MODEL_NAME}")
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
        _clip_model.eval()
        
        return _clip_model, _clip_preprocess
    except ImportError:
        print("[ERROR] 未安装 clip，请运行：pip install git+https://github.com/openai/CLIP.git")
        return None, None
    except Exception as e:
        print(f"[ERROR] 加载 CLIP 模型失败：{e}")
        return None, None


def _prepare_clips_texts(champion_names: List[str]) -> List[str]:
    """准备 CLIP 文本提示"""
    # 使用多种提示模板增强鲁棒性
    templates = [
        "a photo of {} champion from Teamfight Tactics",
        "the League of Legends character {}",
        "{} from TFT game",
        "hero portrait of {}",
    ]
    
    texts = []
    for name in champion_names:
        for tmpl in templates:
            texts.append(tmpl.format(name))
    
    return texts


# ──────────────────────────────────────────────────────────────
# YOLO 检测
# ──────────────────────────────────────────────────────────────
def detect_heroes_yolo(image: np.ndarray, conf_threshold: float = DETECT_CONF) -> List[Tuple[int, int, int, int, float]]:
    """
    使用 YOLO 检测英雄位置
    
    Args:
        image: BGR 图像 (OpenCV 格式)
        conf_threshold: 置信度阈值
    
    Returns:
        [(x, y, w, h, confidence), ...] 检测框列表
    """
    model = _load_yolo_model()
    if model is None:
        return []
    
    # YOLO 需要 RGB 图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 推理
    results = model(image_rgb, conf=conf_threshold, iou=IOU_THRESHOLD, verbose=False)
    result = results[0]
    
    # 提取检测框
    boxes = []
    if result.boxes is not None:
        bboxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confs = result.boxes.conf.cpu().numpy()
        
        for bbox, conf in zip(bboxes, confs):
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            if w > 20 and h > 20:  # 过滤过小框
                boxes.append((x1, y1, w, h, float(conf)))
    
    # NMS（如果 YOLO 未做）
    if len(boxes) > 1:
        boxes = _nms_boxes_yolo(boxes, iou_threshold=IOU_THRESHOLD)
    
    print(f"[YOLO] 检测到 {len(boxes)} 个英雄候选框")
    return boxes


def _nms_boxes_yolo(boxes: List[Tuple], iou_threshold: float = 0.45) -> List[Tuple]:
    """非极大值抑制（按置信度排序）"""
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    
    for b in boxes:
        x1, y1, w1, h1, conf1 = b
        dominated = False
        
        for k in kept:
            x2, y2, w2, h2, _ = k
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            
            if union > 0 and inter / union > iou_threshold:
                dominated = True
                break
        
        if not dominated:
            kept.append(b)
    
    return kept


# ──────────────────────────────────────────────────────────────
# CLIP 识别
# ──────────────────────────────────────────────────────────────
def identify_champion_clip(crop: np.ndarray, champion_names: Optional[List[str]] = None, 
                           threshold: float = CLIP_THRESHOLD) -> Tuple[str, float]:
    """
    使用 CLIP 识别裁剪的英雄图像
    
    Args:
        crop: BGR 裁剪图像
        champion_names: 候选英雄名称列表
        threshold: 相似度阈值
    
    Returns:
        (champion_name, similarity_score)
    """
    import torch
    
    model, preprocess = _load_clip_model()
    if model is None:
        return "", 0.0
    
    if champion_names is None:
        champion_names = _load_champion_names()
    
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    # 转换为 RGB 并预处理
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    
    # 图像编码
    with torch.no_grad():
        image_input = preprocess(crop_pil).unsqueeze(0).to(next(model.parameters()).device)
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 文本编码
        texts = _prepare_clips_texts(champion_names)
        text_tokens = clip.tokenize(texts).to(next(model.parameters()).device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        
        # 每个英雄取最高分（多个模板）
        num_templates = 4  # 与 templates 数量一致
        best_scores = []
        for i, name in enumerate(champion_names):
            start_idx = i * num_templates
            end_idx = start_idx + num_templates
            best_score = similarity[start_idx:end_idx].max()
            best_scores.append((best_score, name))
        
        # 排序
        best_scores.sort(reverse=True)
        
        if best_scores and best_scores[0][0] >= threshold:
            return best_scores[0][1], float(best_scores[0][0])
        
        return "", float(best_scores[0][0]) if best_scores else 0.0


# ──────────────────────────────────────────────────────────────
# 星级检测（保留原逻辑）
# ──────────────────────────────────────────────────────────────
def detect_star(region: np.ndarray) -> int:
    """检测星点数量（1~3 星）"""
    if region is None or region.size == 0:
        return 1
    
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # 金色星点
    gold_mask = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([38, 255, 255]))
    # 白色星点
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    
    combined = cv2.bitwise_or(gold_mask, white_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = (region.shape[0] * region.shape[1]) * 0.003
    
    star_count = sum(1 for cnt in contours if cv2.contourArea(cnt) >= area_threshold)
    return max(1, min(3, star_count)) if star_count > 0 else 1


# ──────────────────────────────────────────────────────────────
# 装备识别（可选 YOLO 或模板匹配）
# ──────────────────────────────────────────────────────────────
def detect_items(image: np.ndarray, hero_box: Tuple[int, int, int, int]) -> List[str]:
    """
    在英雄框下方区域检测装备
    
    Args:
        image: 完整图像
        hero_box: (x, y, w, h) 英雄框
    
    Returns:
        [item_name, ...] 装备名称列表
    """
    x, y, w, h = hero_box
    
    # 装备通常在英雄框下方
    item_y_start = y + h
    item_y_end = y + h + int(h * 0.6)
    item_x_start = x - int(w * 0.2)
    item_x_end = x + w + int(w * 0.2)
    
    h_img, w_img = image.shape[:2]
    item_region = image[max(0, item_y_start):min(h_img, item_y_end),
                        max(0, item_x_start):min(w_img, item_x_end)]
    
    if item_region.size == 0:
        return []
    
    # TODO: 可使用 YOLO 或 CLIP 识别装备
    # 当前返回空列表，后续可扩展
    return []


# ──────────────────────────────────────────────────────────────
# 主识别流程
# ──────────────────────────────────────────────────────────────
def analyze_screenshot(image_path: str, mode: str = "yolo_clip", 
                       debug: bool = False) -> Dict:
    """
    分析 TFT 截图
    
    Args:
        image_path: 截图路径
        mode: 识别模式 ("yolo_clip", "yolo_only", "clip_only")
        debug: 是否显示调试信息
    
    Returns:
        {
            "heroes": [{"name": str, "confidence": float, "box": [x,y,w,h], "stars": int, "items": []}],
            "mode": str,
            "image_size": [w, h]
        }
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"无法读取图像：{image_path}"}
    
    h, w = image.shape[:2]
    result = {
        "heroes": [],
        "mode": mode,
        "image_size": [w, h]
    }
    
    champion_names = _load_champion_names()
    
    if mode in ["yolo_clip", "yolo_only"]:
        # YOLO 检测
        boxes = detect_heroes_yolo(image)
        
        for i, (x, y, bw, bh, conf) in enumerate(boxes):
            # 裁剪英雄区域
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            crop = image[y1:y2, x1:x2]
            
            hero_info = {
                "name": "unknown",
                "confidence": conf,
                "box": [x, y, bw, bh],
                "stars": 1,
                "items": []
            }
            
            if mode == "yolo_clip":
                # CLIP 识别
                name, score = identify_champion_clip(crop, champion_names)
                if name:
                    hero_info["name"] = name
                    hero_info["confidence"] = score
                
                # 星级检测
                star_region = image[max(0, y - int(bh * 0.3)):y, x1:x2]
                hero_info["stars"] = detect_star(star_region)
                
                # 装备检测
                hero_info["items"] = detect_items(image, (x, y, bw, bh))
            
            result["heroes"].append(hero_info)
    
    elif mode == "clip_only":
        # 全图网格搜索（慢，不推荐）
        print("[WARN] clip_only 模式需要已知英雄位置，建议使用 yolo_clip")
    
    if debug:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


# ──────────────────────────────────────────────────────────────
# 训练辅助函数
# ──────────────────────────────────────────────────────────────
def create_training_yaml(output_path: str = "./yolo_data.yaml"):
    """创建 YOLO 训练配置文件"""
    
    yaml_content = f"""# TFT Champion Detection Dataset
path: {Path(output_path).parent.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(DEFAULT_CHAMPIONS)}
names: {json.dumps(DEFAULT_CHAMPIONS)}
"""
    
    with open(output_path, "w") as f:
        f.write(yaml_content)
    
    print(f"[INFO] 已创建训练配置文件：{output_path}")
    return output_path


def export_dataset_images(image_dir: str, output_dir: str = "./yolo_data"):
    """
    将标注数据转换为 YOLO 格式
    
    假设输入为 VOC XML 或 COCO JSON 格式
    """
    from pathlib import Path
    
    input_path = Path(image_dir)
    output_path = Path(output_dir)
    
    # 创建目录结构
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] 数据集导出目录：{output_path}")
    print("[INFO] 请将标注好的图片和标签文件放入对应目录")
    print("[INFO] 然后运行：python tft_yolo_clip.py --train --data ./yolo_data.yaml")


# ──────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT YOLO+CLIP 识别引擎")
    parser.add_argument("image", nargs="?", help="截图路径")
    parser.add_argument("--mode", choices=["yolo_clip", "yolo_only", "clip_only"], 
                        default="yolo_clip", help="识别模式")
    parser.add_argument("--debug", action="store_true", help="显示调试信息")
    parser.add_argument("--detect-only", action="store_true", help="仅 YOLO 检测")
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--data", type=str, help="训练数据配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--export-dataset", type=str, help="导出数据集到指定目录")
    
    args = parser.parse_args()
    
    if args.export_dataset:
        export_dataset_images(args.export_dataset)
        sys.exit(0)
    
    if args.train:
        model = _load_yolo_model()
        if model is None:
            print("[ERROR] 无法加载 YOLO 模型")
            sys.exit(1)
        
        data_file = args.data or "./yolo_data.yaml"
        if not Path(data_file).exists():
            create_training_yaml(data_file)
            print(f"[INFO] 请准备数据集后再次运行")
            sys.exit(0)
        
        print(f"[TRAIN] 开始训练，数据配置：{data_file}, epochs={args.epochs}")
        results = model.train(data=data_file, epochs=args.epochs, imgsz=640)
        print(f"[TRAIN] 训练完成，模型保存至：{results.save_dir}")
        sys.exit(0)
    
    if not args.image:
        parser.print_help()
        print("\n示例:")
        print("  python tft_yolo_clip.py screenshot.png")
        print("  python tft_yolo_clip.py screenshot.png --mode yolo_clip --debug")
        print("  python tft_yolo_clip.py --train --data ./yolo_data.yaml --epochs 100")
        sys.exit(0)
    
    if args.detect_only:
        args.mode = "yolo_only"
    
    result = analyze_screenshot(args.image, mode=args.mode, debug=args.debug)
    
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(1)
    
    # 输出结果
    print(f"\n=== 识别结果 ({args.mode}) ===")
    print(f"图像尺寸：{result['image_size'][0]}x{result['image_size'][1]}")
    print(f"检测到 {len(result['heroes'])} 个英雄:\n")
    
    for i, hero in enumerate(result["heroes"], 1):
        print(f"{i}. {hero['name']:20} 置信度：{hero['confidence']:.3f}  星级：{hero['stars']}  位置：{hero['box']}")
    
    # 保存 JSON
    output_json = Path(args.image).stem + "_yolo_result.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至：{output_json}")
