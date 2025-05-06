# 仮想環境について
# このスクリプトは、プロジェクトルートの.venv仮想環境で実行することを想定しています。
# 必要なパッケージ: numpy, Pillow, scipy
# 実行方法: source .venv/bin/activate && python scripts/mask_generator.py

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from scipy.ndimage import label, center_of_mass
import math
import os
import logging
import sys
from typing import List, Tuple, Optional, Dict, Set
from scipy.spatial import ConvexHull
from skimage.morphology import convex_hull_image
from skimage.measure import find_contours
from shapely.geometry import Polygon, Point

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mask_generator.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_image(image_path: str) -> bool:
    """画像ファイルの存在と形式を検証"""
    if not os.path.exists(image_path):
        logger.error(f"File not found: {image_path}")
        return False
    
    try:
        with Image.open(image_path) as img:
            if img.format not in ['PNG', 'JPEG']:
                logger.error(f"Unsupported image format: {img.format}")
                return False
            # RGBAの場合はRGBに変換
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ['RGB']:
                logger.error(f"Unsupported color mode: {img.mode}")
                return False
    except Exception as e:
        logger.error(f"Error validating image {image_path}: {str(e)}")
        return False
    
    return True

def flood_fill_background(arr: np.ndarray, bg_color: np.ndarray, threshold: int = 30) -> np.ndarray:
    """背景のFlood Fill処理（HSV色空間を使用して背景色の範囲を決定）"""
    try:
        h, w, c = arr.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        queue = deque()
        
        # 外周ピクセルをキューに追加
        for x in range(w):
            queue.append((0, x))
            queue.append((h-1, x))
        for y in range(h):
            queue.append((y, 0))
            queue.append((y, w-1))
        
        # RGBからHSVに変換
        img_rgb = Image.fromarray(arr)
        img_hsv = img_rgb.convert('HSV')
        arr_hsv = np.array(img_hsv)
        
        # 外周ピクセルのHSV値の統計を計算
        border_pixels = []
        for x in range(w):
            border_pixels.append(arr_hsv[0, x])
            border_pixels.append(arr_hsv[h-1, x])
        for y in range(h):
            border_pixels.append(arr_hsv[y, 0])
            border_pixels.append(arr_hsv[y, w-1])
        
        border_pixels = np.array(border_pixels)
        mean_hsv = np.mean(border_pixels, axis=0)
        std_hsv = np.std(border_pixels, axis=0)
        
        # HSVの各成分の閾値を設定
        # H（色相）は無視、S（彩度）とV（明度）のみで判定
        s_threshold = 30  # 彩度の閾値（低いほど背景と判定）
        v_threshold = 200  # 明度の閾値（高いほど背景と判定）
        
        logger.info(f"Background color statistics (HSV):")
        logger.info(f"Mean HSV: {mean_hsv}")
        logger.info(f"Std HSV: {std_hsv}")
        logger.info(f"Saturation threshold: {s_threshold}")
        logger.info(f"Value threshold: {v_threshold}")
        
        # Flood Fill
        while queue:
            y, x = queue.popleft()
            if visited[y, x]:
                continue
            visited[y, x] = True
            hsv = arr_hsv[y, x]
            # 彩度が低く、明度が高い場合を背景と判定
            if hsv[1] <= s_threshold and hsv[2] >= v_threshold:
                mask[y, x] = 1
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0<=ny<h and 0<=nx<w and not visited[ny, nx]:
                        queue.append((ny, nx))
        return mask
    except Exception as e:
        logger.error(f"Error in flood_fill_background: {str(e)}")
        raise

def create_mask(image_path: str, mask_path: str, threshold: int = 30) -> None:
    """マスク画像の生成"""
    try:
        if not validate_image(image_path):
            return

        img = Image.open(image_path)
        # デバッグ用にRGB変換後の画像を保存（RGBAでもRGBでも保存）
        rgb_img = img.convert('RGB')
        # debug_rgb_path = os.path.splitext(mask_path)[0] + '_rgb.png'
        # rgb_img.save(debug_rgb_path)
        # logger.info(f"Saved debug RGB image: {debug_rgb_path}")
        
        # 以降の処理はRGB画像を使用
        arr_rgb = np.array(rgb_img)

        # 外周ピクセルの平均色を背景色とする
        border = 1
        top = arr_rgb[:border, :, :]
        bottom = arr_rgb[-border:, :, :]
        left = arr_rgb[:, :border, :]
        right = arr_rgb[:, -border:, :]
        outer_pixels = np.concatenate([top, bottom, left, right], axis=None).reshape(-1, arr_rgb.shape[2])
        bg_color = np.mean(outer_pixels, axis=0)

        # Flood Fillで背景領域を特定
        bg_mask = flood_fill_background(arr_rgb, bg_color, threshold=threshold)

        # 背景以外を白（255）、背景を黒（0）
        mask = np.where(bg_mask == 0, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(mask_path)
        logger.info(f"Created mask: {mask_path}")
    except Exception as e:
        logger.error(f"Error in create_mask: {str(e)}")
        raise

def draw_bounding_box(arr_draw, labeled, object_id, color, line_width=2):
    # オブジェクトの座標を取得
    y_coords, x_coords = np.where(labeled == object_id)
    if len(y_coords) == 0:
        return
    
    # バウンディングボックスの座標を計算
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    
    # 枠線を描画
    for w in range(line_width):
        # 上辺
        for x in range(min_x - w, max_x + w + 1):
            if 0 <= min_y - w < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                arr_draw[min_y - w, x] = color
        # 下辺
        for x in range(min_x - w, max_x + w + 1):
            if 0 <= max_y + w < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                arr_draw[max_y + w, x] = color
        # 左辺
        for y in range(min_y - w, max_y + w + 1):
            if 0 <= y < arr_draw.shape[0] and 0 <= min_x - w < arr_draw.shape[1]:
                arr_draw[y, min_x - w] = color
        # 右辺
        for y in range(min_y - w, max_y + w + 1):
            if 0 <= y < arr_draw.shape[0] and 0 <= max_x + w < arr_draw.shape[1]:
                arr_draw[y, max_x + w] = color

def draw_line(arr_draw, start_y, start_x, end_y, end_x, color, width=2):
    """2点間を結ぶ線を描画する"""
    # 線の長さを計算
    length = math.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
    if length == 0:
        return
    
    # 線上の点を計算
    for t in np.linspace(0, 1, int(length * 2)):  # 2倍の密度で点を打つ
        y = int(start_y + (end_y - start_y) * t)
        x = int(start_x + (end_x - start_x) * t)
        
        # 線の太さ分の点を描画
        for dy in range(-width//2, width//2 + 1):
            for dx in range(-width//2, width//2 + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < arr_draw.shape[0] and 0 <= nx < arr_draw.shape[1]:
                    arr_draw[ny, nx] = color

def mark_centers_on_mask(mask_path, output_path, area_threshold=30):
    try:
        print(f"Loading mask: {mask_path}")
        mask_img = Image.open(mask_path).convert('L')
        arr = np.array(mask_img)
        print(f"Mask shape: {arr.shape}, dtype: {arr.dtype}")
        
        # 白いエリア（255）のラベリング
        structure = np.ones((3,3), dtype=int)
        labeled, num_features = label(arr == 255, structure=structure)
        print(f"Number of features: {num_features}")
        
        # 面積フィルタリングのカウンター
        filtered_objects = 0
        
        # バウンディングボックス情報を保存するリスト
        bounding_box_list = []
        valid_box_flags = []
        
        # 各オブジェクトの面積を計算
        areas = []
        chull_dict = {}
        for i in range(1, num_features + 1):
            obj_mask = (labeled == i)
            if np.sum(obj_mask) > 0:
                chull = convex_hull_image(obj_mask)
                area = np.sum(chull)
                chull_dict[i] = chull
                if area > area_threshold:
                    areas.append((i, area))
                    # バウンディングボックスを計算
                    y_coords, x_coords = np.where(obj_mask)
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    bounding_box_list.append((min_x, min_y, max_x, max_y))
                    valid_box_flags.append(True)
            else:
                filtered_objects += 1
        print(f"\nFiltered out {filtered_objects} small objects (area <= {area_threshold})")
        
        # 面積でソート
        areas.sort(key=lambda x: x[1], reverse=True)
        print("\nSorted areas:")
        for obj_id, area in areas:
            print(f"Object {obj_id}: area = {area}")
        
        # 面積が10000以上のオブジェクトIDを主オブジェクトとする
        main_object_ids = set(obj_id for obj_id, area in areas if area >= 10000)
        print(f"\nMain object IDs (area >= 10000): {main_object_ids}")
        
        # 結果を描画するための配列
        arr_draw = arr.copy()
        bounding_img = Image.fromarray(arr_draw).convert('RGBA')  # RGBAに変更
        draw = ImageDraw.Draw(bounding_img)
        
        # 主・副オブジェクトの重心・半径リストを作成
        object_info = []  # (obj_id, cy, cx, radius, is_main, area)
        for obj_id, area in areas:
            y_coords, x_coords = np.where(labeled == obj_id)
            cy, cx = center_of_mass(labeled == obj_id)
            cy, cx = int(cy), int(cx)
            # 凸包の外接円半径（近似）
            if len(x_coords) >= 3:
                points = np.stack([x_coords, y_coords], axis=1)
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                if len(hull_points) >= 2:
                    dists = np.sqrt(np.sum((hull_points[None,:,:] - hull_points[:,None,:])**2, axis=2))
                    max_dist = np.max(dists)
                    radius = int(max_dist/2)
                else:
                    radius = 1
            elif len(x_coords) == 2:
                radius = int(np.linalg.norm([x_coords[0]-x_coords[1], y_coords[0]-y_coords[1]]) / 2)
            else:
                radius = 1
            is_main = obj_id in main_object_ids
            object_info.append((obj_id, cy, cx, radius, is_main, area))

        # 衝突検出（マージ対象IDセットを作成）
        collisions = detect_convex_hull_collisions(labeled, object_info)
        merge_target_ids = set()
        for ids in collisions.values():
            merge_target_ids.update(ids)
        merge_target_ids.update([k for k, v in collisions.items() if v])

        # 半透明青バウンディングボックス用のオーバーレイを作成
        overlay = Image.new('RGBA', bounding_img.size, (0,0,0,0))
        draw_overlay = ImageDraw.Draw(overlay)

        # 各オブジェクトの凸包を描画（マージ対象は黄色、それ以外は赤）
        for obj_id, cy, cx, radius, is_main, area in object_info:
            y_coords, x_coords = np.where(labeled == obj_id)
            points = np.stack([x_coords, y_coords], axis=1)  # (x, y)順
            ellipse_width = 10 if is_main else 5
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                polygon = [tuple(map(int, p)) for p in hull_points]
                polygon.append(polygon[0])
                if obj_id in merge_target_ids:
                    draw.line(polygon, fill=(255,255,0,255), width=ellipse_width)  # 黄色
                else:
                    draw.line(polygon, fill=(255,0,0,255), width=ellipse_width)  # 赤
            elif len(points) == 2:
                p0 = tuple(map(int, points[0]))
                p1 = tuple(map(int, points[1]))
                if obj_id in merge_target_ids:
                    draw.line([p0, p1], fill=(255,255,0,255), width=ellipse_width)
                else:
                    draw.line([p0, p1], fill=(255,0,0,255), width=ellipse_width)
            elif len(points) == 1:
                x, y = map(int, points[0])
                if obj_id in merge_target_ids:
                    draw.ellipse([x-2, y-2, x+2, y+2], outline=(255,255,0,255), width=2)
                else:
                    draw.ellipse([x-2, y-2, x+2, y+2], outline=(255,0,0,255), width=2)

        # overlayのアルファ値を50%にして合成
        alpha = 128
        overlay_alpha = overlay.split()[3].point(lambda p: alpha if p > 0 else 0)
        overlay.putalpha(overlay_alpha)
        bounding_img = Image.alpha_composite(bounding_img, overlay)

        # マージ後の凸包を緑色で描画
        if len(object_info) > 0:
            collisions = detect_convex_hull_collisions(labeled, object_info)
            if any(collisions.values()):
                labeled, object_info = merge_colliding_objects(labeled, object_info, collisions)
                for obj_id, cy, cx, radius, is_main, area in object_info:
                    y_coords, x_coords = np.where(labeled == obj_id)
                    points = np.stack([x_coords, y_coords], axis=1)
                    ellipse_width = 10 if is_main else 5
                    if len(points) >= 3:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        polygon = [tuple(map(int, p)) for p in hull_points]
                        polygon.append(polygon[0])
                        logger.info(f"[MERGED CONVEX HULL] obj_id={obj_id}, vertices={polygon}")
                        draw.line(polygon, fill=(0,255,0,255), width=ellipse_width)  # 緑色
                    elif len(points) == 2:
                        p0 = tuple(map(int, points[0]))
                        p1 = tuple(map(int, points[1]))
                        logger.info(f"[MERGED CONVEX HULL] obj_id={obj_id}, vertices={[p0, p1]}")
                        draw.line([p0, p1], fill=(0,255,0,255), width=ellipse_width)
                    elif len(points) == 1:
                        x, y = map(int, points[0])
                        logger.info(f"[MERGED CONVEX HULL] obj_id={obj_id}, vertices={[(x, y)]}")
                        draw.ellipse([x-2, y-2, x+2, y+2], outline=(0,255,0,255), width=2)

        logger.info(f"Found {len(bounding_box_list)} valid objects")
        
        # 結果を保存
        bounding_img.save(output_path)
        logger.info(f"Saved marked centers image: {output_path}")
        
        return bounding_box_list, valid_box_flags
        
    except Exception as e:
        logger.error(f"Error in mark_centers_on_mask: {str(e)}")
        raise

def extract_bounding_box_segments(
    image_path: str,
    mask_path: str,
    bounding_box_list: List[Tuple[int, int, int, int]],
    valid_box_flags: List[bool],
    output_path_prefix: str
) -> None:
    """バウンディングボックス領域の切り出しと保存"""
    try:
        # 入力画像のみ検証（マスク画像は自作なので検証不要）
        if not validate_image(image_path):
            return

        img = Image.open(image_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L')

        # 出力前に既存の-1.png～-9.pngを削除
        for i in range(1, 10):
            del_path = f"{output_path_prefix}-{i}.png"
            if os.path.exists(del_path):
                os.remove(del_path)
                logger.debug(f"Removed existing file: {del_path}")

        # 有効なボックスのみ抽出
        boxes = [box for box, valid in zip(bounding_box_list, valid_box_flags) if valid]
        if not boxes:
            logger.warning("No valid bounding boxes found")
            return

        # 面積でソートして上位9個を取得
        areas = [(i, (x2-x1)*(y2-y1)) for i, (x1, y1, x2, y2) in enumerate(boxes)]
        top9 = sorted(areas, key=lambda x: x[1], reverse=True)[:9]
        top9_indices = [x[0] for x in top9]
        top9_boxes = [boxes[i] for i in top9_indices]

        # 左上から右下の順（y1,x1昇順）
        top9_boxes = sorted(top9_boxes, key=lambda box: (box[1], box[0]))

        MAX_WIDTH = 370
        MAX_HEIGHT = 320

        for idx, (x1, y1, x2, y2) in enumerate(top9_boxes):
            try:
                # 元画像・マスク画像から切り出し
                segment = img.crop((x1, y1, x2+1, y2+1))
                segment_mask = mask.crop((x1, y1, x2+1, y2+1))

                # リサイズ（アスペクト比保持）
                if segment.width > MAX_WIDTH or segment.height > MAX_HEIGHT:
                    ratio = min(MAX_WIDTH / segment.width, MAX_HEIGHT / segment.height)
                    new_width = int(segment.width * ratio)
                    new_height = int(segment.height * ratio)
                    segment = segment.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    segment_mask = segment_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # マスクの黒部分を透明化
                segment_array = np.array(segment)
                mask_array = np.array(segment_mask)
                segment_array[mask_array == 0, 3] = 0
                segment = Image.fromarray(segment_array)

                # 保存
                output_path = f"{output_path_prefix}-{idx+1}.png"
                segment.save(output_path, "PNG")
                logger.info(f"Saved segment {idx+1}: {output_path}")

            except Exception as e:
                logger.error(f"Error processing segment {idx+1}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in extract_bounding_box_segments: {str(e)}")
        raise

def detect_convex_hull_collisions(labeled: np.ndarray, object_info: List[Tuple]) -> Dict[int, List[int]]:
    """
    凸包同士の衝突を検出する
    
    Args:
        labeled: ラベル付き画像配列
        object_info: オブジェクト情報のリスト (obj_id, cy, cx, radius, is_main, area)
    
    Returns:
        Dict[int, List[int]]: 衝突しているオブジェクトIDのマッピング
    """
    collisions = {}
    
    # 各オブジェクトの凸包を計算
    hulls = {}
    for obj_id, _, _, _, _, _ in object_info:
        y_coords, x_coords = np.where(labeled == obj_id)
        points = np.stack([x_coords, y_coords], axis=1)
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            polygon = Polygon(hull_points)
            hulls[obj_id] = polygon
    
    # 衝突検出
    for obj_id1, hull1 in hulls.items():
        collisions[obj_id1] = []
        for obj_id2, hull2 in hulls.items():
            if obj_id1 != obj_id2 and hull1.intersects(hull2):
                collisions[obj_id1].append(obj_id2)
    
    return collisions

def merge_colliding_objects(labeled: np.ndarray, object_info: List[Tuple], collisions: Dict[int, List[int]]) -> Tuple[np.ndarray, List[Tuple]]:
    """
    衝突しているオブジェクトをマージする
    
    Args:
        labeled: ラベル付き画像配列
        object_info: オブジェクト情報のリスト
        collisions: 衝突情報の辞書
    
    Returns:
        Tuple[np.ndarray, List[Tuple]]: マージ後のラベル付き画像とオブジェクト情報
    """
    # マージするオブジェクトのグループを特定
    merge_groups: List[Set[int]] = []
    processed = set()
    
    for obj_id, colliding in collisions.items():
        if obj_id in processed:
            continue
            
        if not colliding:
            processed.add(obj_id)
            continue
            
        # 新しいマージグループを作成
        group = {obj_id}
        group.update(colliding)
        
        # 既存のグループとマージ
        merged = False
        for existing_group in merge_groups:
            if group & existing_group:
                existing_group.update(group)
                merged = True
                break
        
        if not merged:
            merge_groups.append(group)
        
        processed.update(group)
    
    # マージグループの情報をログ出力
    if merge_groups:
        logger.info("Merging groups:")
        for i, group in enumerate(merge_groups, 1):
            logger.info(f"Group {i}: {sorted(group)}")
    
    # マージグループごとに処理
    new_labeled = labeled.copy()
    new_object_info = []
    next_label = 1
    
    # マージしないオブジェクトを処理
    for obj_id, cy, cx, radius, is_main, area in object_info:
        if not any(obj_id in group for group in merge_groups):
            new_labeled[new_labeled == obj_id] = next_label
            new_object_info.append((next_label, cy, cx, radius, is_main, area))
            logger.info(f"Kept object {obj_id} as new object {next_label}")
            next_label += 1
    
    # マージグループを処理
    for group in merge_groups:
        # グループ内のオブジェクトをマージ
        merged_mask = np.zeros_like(labeled, dtype=bool)
        for obj_id in group:
            merged_mask |= (labeled == obj_id)
        
        # 新しいラベルを割り当て
        new_labeled[merged_mask] = next_label
        
        # マージ後のオブジェクト情報を計算
        y_coords, x_coords = np.where(merged_mask)
        cy, cx = center_of_mass(merged_mask)
        cy, cx = int(cy), int(cx)
        
        # 凸包の計算
        if len(x_coords) >= 3:
            points = np.stack([x_coords, y_coords], axis=1)
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            area = Polygon(hull_points).area
            if len(hull_points) >= 2:
                dists = np.sqrt(np.sum((hull_points[None,:,:] - hull_points[:,None,:])**2, axis=2))
                max_dist = np.max(dists)
                radius = int(max_dist/2)
            else:
                radius = 1
        elif len(x_coords) == 2:
            area = 2
            radius = int(np.linalg.norm([x_coords[0]-x_coords[1], y_coords[0]-y_coords[1]]) / 2)
        else:
            area = 1
            radius = 1
        
        # 主オブジェクトかどうかの判定（面積で判断）
        is_main = area >= 10000
        
        new_object_info.append((next_label, cy, cx, radius, is_main, area))
        logger.info(f"Merged objects {sorted(group)} into new object {next_label} (area: {area:.1f}, {'main' if is_main else 'sub'})")
        next_label += 1
    
    return new_labeled, new_object_info

def analyze_object_hierarchy(mask_path, area_main_threshold=10000):
    """
    マスク画像からオブジェクトの主従関係を整理し、重心座標・主従情報・可視化画像を返す
    入力: mask_path（マスク画像ファイルパス）
    出力: centers（{obj_id: (cy, cx)}）、hierarchy（{obj_id: {'is_main': bool, 'parent': main_id or None}}）、bounding_img（Pillow画像）
    """
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import label, center_of_mass
    import math

    mask_img = Image.open(mask_path).convert('L')
    arr = np.array(mask_img)
    structure = np.ones((3,3), dtype=int)
    labeled, num_features = label(arr == 255, structure=structure)
    logger.info(f"Number of features detected: {num_features}")

    # 面積・重心・半径・主従情報を集計（面積は凸包で計算）
    areas = []
    chull_dict = {}
    filtered_objects = 0
    for i in range(1, num_features + 1):
        obj_mask = (labeled == i)
        if np.sum(obj_mask) > 0:
            chull = convex_hull_image(obj_mask)
            area = np.sum(chull)
            chull_dict[i] = chull
            logger.info(f"Object {i}: raw area = {np.sum(obj_mask)}, convex hull area = {area}")
            if area > 30:
                areas.append((i, area))
                logger.info(f"Object {i} passed area filter")
            else:
                logger.info(f"Object {i} filtered out (area <= 30)")
        else:
            filtered_objects += 1
            logger.info(f"Object {i} has no pixels")
    areas.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Total objects after area filtering: {len(areas)}")

    # 主オブジェクトID（凸包面積で判定）
    main_object_ids = set(obj_id for obj_id, area in areas if area >= area_main_threshold)
    # オブジェクト情報リスト
    object_info = []  # (obj_id, cy, cx, radius, is_main, area)
    centers = {}
    for obj_id, area in areas:
        y_coords, x_coords = np.where(labeled == obj_id)
        cy, cx = center_of_mass(labeled == obj_id)
        cy, cx = int(cy), int(cx)
        # 凸包の外接円半径（近似）
        if len(x_coords) >= 3:
            points = np.stack([x_coords, y_coords], axis=1)
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            if len(hull_points) >= 2:
                dists = np.sqrt(np.sum((hull_points[None,:,:] - hull_points[:,None,:])**2, axis=2))
                max_dist = np.max(dists)
                radius = int(max_dist/2)
            else:
                radius = 1
        elif len(x_coords) == 2:
            radius = int(np.linalg.norm([x_coords[0]-x_coords[1], y_coords[0]-y_coords[1]]) / 2)
        else:
            radius = 1
        is_main = obj_id in main_object_ids
        object_info.append((obj_id, cy, cx, radius, is_main, area))
        centers[obj_id] = (cy, cx)

    # 衝突がなくなるまでマージを繰り返す
    max_iterations = 10  # 無限ループ防止
    for iteration in range(max_iterations):
        # 凸包の衝突を検出
        collisions = detect_convex_hull_collisions(labeled, object_info)
        
        # 衝突がない場合は終了
        if not any(collisions.values()):
            logger.info(f"No more collisions after {iteration} iterations")
            break
            
        # 衝突情報をログに出力
        logger.info(f"Iteration {iteration + 1}:")
        for obj_id, colliding_objects in collisions.items():
            if colliding_objects:
                logger.info(f"Object {obj_id} collides with objects: {colliding_objects}")
        
        # オブジェクトをマージ
        labeled, object_info = merge_colliding_objects(labeled, object_info, collisions)
        
        # 新しいオブジェクト情報からcentersを更新
        centers = {obj_id: (cy, cx) for obj_id, cy, cx, _, _, _ in object_info}
    
    # 主従認識情報
    hierarchy = {}
    for obj_id, cy, cx, radius, is_main, area in object_info:
        hierarchy[obj_id] = {'is_main': is_main, 'parent': None}

    # 結果を描画するための配列
    arr_draw = arr.copy()
    bounding_img = Image.fromarray(arr_draw).convert('RGB')
    draw = ImageDraw.Draw(bounding_img)

    # 各オブジェクトの凸包を赤枠で描画
    for obj_id, cy, cx, radius, is_main, area in object_info:
        # オブジェクトの画素座標を取得
        y_coords, x_coords = np.where(labeled == obj_id)
        points = np.stack([x_coords, y_coords], axis=1)  # (x, y)順
        ellipse_width = 10 if is_main else 5
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            polygon = [tuple(map(int, p)) for p in hull_points]
            # 閉じる
            polygon.append(polygon[0])
            draw.line(polygon, fill=(255, 0, 0), width=ellipse_width)
        elif len(points) == 2:
            p0 = tuple(map(int, points[0]))
            p1 = tuple(map(int, points[1]))
            draw.line([p0, p1], fill=(255, 0, 0), width=ellipse_width)
        elif len(points) == 1:
            x, y = map(int, points[0])
            draw.ellipse([x-2, y-2, x+2, y+2], outline=(255, 0, 0), width=2)

    return centers, hierarchy, bounding_img

if __name__ == '__main__':
    try:
        # ディレクトリ構造の設定
        input_dir = os.path.join('output', 'generated')
        output_dir = 'output'
        mask_dir = os.path.join('output', 'masks')
        bounding_dir = os.path.join('output', 'bounding')

        # 必要なディレクトリを作成
        for dir_path in [mask_dir, bounding_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")

        # 入力ディレクトリの存在確認
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)

        # output/generatedフォルダ内のすべてのPNGファイルを処理
        png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        png_files.sort()  # ファイル名でソート
        if not png_files:
            logger.warning(f"No PNG files found in {input_dir}")
            sys.exit(0)

        for filename in png_files:
            try:
                base = os.path.splitext(filename)[0]
                input_path = os.path.join(input_dir, filename)
                mask_path = os.path.join(mask_dir, f'{base}_mask.png')
                bounding_path = os.path.join(bounding_dir, f'{base}_bounding.png')
                output_prefix = os.path.join(output_dir, base)

                logger.info(f"\nProcessing: {filename}")
                logger.info(f"Input: {input_path}")
                logger.info(f"Mask: {mask_path}")
                logger.info(f"Bounding: {bounding_path}")
                logger.info(f"Output prefix: {output_prefix}")

                # マスク生成
                create_mask(input_path, mask_path)
                # 青枠描画＋座標リスト取得
                bounding_box_list, valid_box_flags = mark_centers_on_mask(mask_path, bounding_path)
                # 青枠領域の切り出し
                extract_bounding_box_segments(input_path, mask_path, bounding_box_list, valid_box_flags, output_prefix)

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 