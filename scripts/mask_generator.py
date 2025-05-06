import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from scipy.ndimage import label, center_of_mass
import math
import os
import logging
import sys
from typing import List, Tuple, Optional

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
            if img.mode not in ['RGB', 'RGBA']:
                logger.error(f"Unsupported color mode: {img.mode}")
                return False
    except Exception as e:
        logger.error(f"Error validating image {image_path}: {str(e)}")
        return False
    
    return True

def flood_fill_background(arr: np.ndarray, bg_color: np.ndarray, alpha: Optional[np.ndarray] = None, threshold: int = 30) -> np.ndarray:
    """背景のFlood Fill処理（アルファ値が低い部分は常に背景とみなす）"""
    try:
        h, w, c = arr.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        queue = deque()
        
        # 外周ピクセルをキューに追加
        for x in range(w):
            if alpha is not None and alpha[0, x] < 128:
                continue
            queue.append((0, x))
            if alpha is not None and alpha[h-1, x] < 128:
                continue
            queue.append((h-1, x))
        for y in range(h):
            if alpha is not None and alpha[y, 0] < 128:
                continue
            queue.append((y, 0))
            if alpha is not None and alpha[y, w-1] < 128:
                continue
            queue.append((y, w-1))
        
        # 外周ピクセルのRGB値のmin-maxを計算
        border_pixels = []
        for x in range(w):
            if alpha is None or alpha[0, x] >= 128:
                border_pixels.append(arr[0, x])
            if alpha is None or alpha[h-1, x] >= 128:
                border_pixels.append(arr[h-1, x])
        for y in range(h):
            if alpha is None or alpha[y, 0] >= 128:
                border_pixels.append(arr[y, 0])
            if alpha is None or alpha[y, w-1] >= 128:
                border_pixels.append(arr[y, w-1])
        
        border_pixels = np.array(border_pixels)
        min_rgb = np.min(border_pixels, axis=0)
        max_rgb = np.max(border_pixels, axis=0)
        
        # Flood Fill
        while queue:
            y, x = queue.popleft()
            if visited[y, x]:
                continue
            visited[y, x] = True
            # アルファ値が低い場合は常に背景
            if alpha is not None and alpha[y, x] < 128:
                mask[y, x] = 1
                continue
            color = arr[y, x]
            if np.all(min_rgb <= color) and np.all(color <= max_rgb):
                mask[y, x] = 1
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0<=ny<h and 0<=nx<w and not visited[ny, nx]:
                        # 進行先もアルファ値128以上のみ
                        if alpha is not None and alpha[ny, nx] < 128:
                            continue
                        queue.append((ny, nx))
        return mask
    except Exception as e:
        logger.error(f"Error in flood_fill_background: {str(e)}")
        raise

def create_mask(image_path: str, mask_path: str, threshold: int = 30) -> None:
    """マスク画像の生成（RGBA画像の透明部分は白で埋めてから処理）"""
    try:
        if not validate_image(image_path):
            return

        img = Image.open(image_path)
        if img.mode == 'RGBA':
            arr = np.array(img)
            alpha = arr[..., 3]
            arr_rgb = arr[..., :3].copy()
            # 透明部分（アルファ値128未満）を白で埋める
            arr_rgb[alpha < 128] = [255, 255, 255]
        else:
            arr_rgb = np.array(img.convert('RGB'))

        # 外周ピクセルの平均色を背景色とする
        border = 1
        top = arr_rgb[:border, :, :]
        bottom = arr_rgb[-border:, :, :]
        left = arr_rgb[:, :border, :]
        right = arr_rgb[:, -border:, :]
        outer_pixels = np.concatenate([top, bottom, left, right], axis=None).reshape(-1, arr_rgb.shape[2])
        bg_color = np.mean(outer_pixels, axis=0)

        # Flood Fillで背景領域を特定
        bg_mask = flood_fill_background(arr_rgb, bg_color, alpha=None, threshold=threshold)

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

def mark_centers_on_mask(mask_path, output_path):
    try:
        print(f"Loading mask: {mask_path}")
        mask_img = Image.open(mask_path).convert('L')
        arr = np.array(mask_img)
        print(f"Mask shape: {arr.shape}, dtype: {arr.dtype}")
        # 白いエリア（255）のラベリング
        structure = np.ones((3,3), dtype=int)
        labeled, num_features = label(arr == 255, structure=structure)
        print(f"Number of features: {num_features}")
        
        # 各オブジェクトの面積を計算
        areas = []
        for i in range(1, num_features + 1):
            area = np.sum(labeled == i)
            areas.append((i, area))  # (オブジェクトID, 面積)のタプルを保存
            print(f"Object {i}: area = {area}")
        
        # 面積でソート
        areas.sort(key=lambda x: x[1], reverse=True)
        print("\nSorted areas:")
        for obj_id, area in areas:
            print(f"Object {obj_id}: area = {area}")
        
        # メインオブジェクトのIDを保存
        main_object_ids = set(obj_id for obj_id, _ in areas[:9])
        print(f"\nMain object IDs: {main_object_ids}")
        
        centers = center_of_mass(arr == 255, labeled, range(1, num_features+1))
        print("\nCenters:")
        for i, (cy, cx) in enumerate(centers, 1):
            print(f"Object {i}: center = ({cy}, {cx})")
        
        # RGB画像に変換
        out_img = mask_img.convert('RGB')
        arr_draw = np.array(out_img)
        
        # 主オブジェクトの中心座標を保存
        main_centers = []
        for i, (cy, cx) in enumerate(centers):
            obj_id = i + 1
            if obj_id in main_object_ids:
                main_centers.append((obj_id, cy, cx))
        
        # 統合されたオブジェクトのマスクを作成
        combined_mask = np.zeros_like(arr)
        
        # 重心を描画
        for i, (cy, cx) in enumerate(centers):
            obj_id = i + 1  # オブジェクトIDは1から始まる
            cy, cx = int(round(cy)), int(round(cx))
            
            # 対応する面積を探す
            area = next(area for obj_id2, area in areas if obj_id2 == obj_id)
            print(f"\nDrawing circle for Object {obj_id}:")
            print(f"  Center: ({cy}, {cx})")
            print(f"  Area: {area}")
            
            # 円の半径を計算（面積が等しくなるように）
            radius = int(math.sqrt(area / math.pi))
            print(f"  Radius: {radius}")
            
            # オブジェクトがメインかどうかで色と線の太さを決定
            is_main = obj_id in main_object_ids
            color = [255, 0, 0]  # 両方とも明るい赤
            line_width = 10 if is_main else 2  # 主は10ピクセル、副は2ピクセル
            
            # 円の輪郭を描画（より細かい間隔で）
            for angle in range(0, 360, 1):  # 1度ずつ描画
                rad = math.radians(angle)
                x = int(cx + radius * math.cos(rad))
                y = int(cy + radius * math.sin(rad))
                if 0 <= y < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                    # 線の太さを調整
                    for dy in range(-line_width//2, line_width//2 + 1):
                        for dx in range(-line_width//2, line_width//2 + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < arr_draw.shape[0] and 0 <= nx < arr_draw.shape[1]:
                                arr_draw[ny, nx] = color
                                combined_mask[ny, nx] = 255  # 赤い円をマスクに追加
            
            # 白いオブジェクトをマスクに追加
            combined_mask[labeled == obj_id] = 255
            
            # 副オブジェクトの場合、最も近い主オブジェクトまでの線を描画
            if not is_main:
                min_dist = float('inf')
                closest_main = None
                
                # 最も近い主オブジェクトを探す
                for main_id, main_cy, main_cx in main_centers:
                    # 主オブジェクトの円周上の最も近い点を計算
                    dx = cx - main_cx
                    dy = cy - main_cy
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
                        closest_main = (main_id, main_cy, main_cx)
                
                if closest_main:
                    main_id, main_cy, main_cx = closest_main
                    # 主オブジェクトの円周上の点を計算
                    dx = cx - main_cx
                    dy = cy - main_cy
                    angle = math.atan2(dy, dx)
                    main_radius = int(math.sqrt(next(area for obj_id2, area in areas if obj_id2 == main_id) / math.pi))
                    end_x = int(main_cx + main_radius * math.cos(angle))
                    end_y = int(main_cy + main_radius * math.sin(angle))
                    
                    # 線を描画
                    draw_line(arr_draw, cy, cx, end_y, end_x, [255, 0, 0], width=2)
                    # 線をマスクに追加
                    draw_line(combined_mask, cy, cx, end_y, end_x, 255, width=2)
        
        # 統合されたオブジェクトをラベリング
        combined_labeled, combined_num = label(combined_mask == 255, structure=structure)
        
        # 枠線描画後にImage.fromarrayで画像を生成し、テキストのみImageDrawで描画
        out_img = Image.fromarray(arr_draw)
        draw = ImageDraw.Draw(out_img)
        
        # フォントサイズを設定（より大きく）
        font_size = 32
        try:
            # フォントサイズを大きく設定
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except:
            # デフォルトフォントの場合は、サイズを大きくするためにスケーリング
            font = ImageFont.load_default()
            font = font.font_variant(size=font_size)
        
        # 青い枠を描画（統合されたオブジェクトに対して）
        bounding_box_list = []
        valid_box_flags = []
        # まず全てのバウンディングボックスをリストアップし、valid判定も行う
        for i in range(1, combined_num + 1):
            y_coords, x_coords = np.where(combined_labeled == i)
            if len(y_coords) == 0:
                continue
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            line_width = 4
            padding = 4
            box = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
            bounding_box_list.append(box)
            xw = box[2] - box[0]
            yw = box[3] - box[1]
            ratio = xw / yw if yw != 0 else 0
            valid = 0.7 <= ratio <= 1.3 and xw <= 500 and yw <= 500
            valid_box_flags.append(valid)
        # 青枠を全て描画
        for box in bounding_box_list:
            min_x, min_y, max_x, max_y = box
            line_width = 4
            # 青枠
            for w in range(line_width):
                for x in range(min_x - w, max_x + w + 1):
                    if 0 <= min_y - w < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                        arr_draw[min_y - w, x] = [0, 0, 255]
                    if 0 <= max_y + w < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                        arr_draw[max_y + w, x] = [0, 0, 255]
                for y in range(min_y - w, max_y + w + 1):
                    if 0 <= y < arr_draw.shape[0] and 0 <= min_x - w < arr_draw.shape[1]:
                        arr_draw[y, min_x - w] = [0, 0, 255]
                    if 0 <= y < arr_draw.shape[0] and 0 <= max_x + w < arr_draw.shape[1]:
                        arr_draw[y, max_x + w] = [0, 0, 255]
        # validなものだけ水色枠を重ねて描画
        for box, valid in zip(bounding_box_list, valid_box_flags):
            if valid:
                min_x, min_y, max_x, max_y = box
                for x in range(min_x, max_x+1):
                    if 0 <= min_y < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                        arr_draw[min_y, x] = [255, 255, 0]  # 黄色
                    if 0 <= max_y < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                        arr_draw[max_y, x] = [255, 255, 0]  # 黄色
                for y in range(min_y, max_y+1):
                    if 0 <= y < arr_draw.shape[0] and 0 <= min_x < arr_draw.shape[1]:
                        arr_draw[y, min_x] = [255, 255, 0]  # 黄色
                    if 0 <= y < arr_draw.shape[0] and 0 <= max_x < arr_draw.shape[1]:
                        arr_draw[y, max_x] = [255, 255, 0]  # 黄色
        # 枠線描画後にImage.fromarrayで画像を生成し、テキストのみImageDrawで描画
        out_img = Image.fromarray(arr_draw)
        draw = ImageDraw.Draw(out_img)
        # テキスト描画（青枠の中心にサイズ表示）
        font_size = 32
        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except:
            font = ImageFont.load_default()
            font = font.font_variant(size=font_size)
        for box in bounding_box_list:
            min_x, min_y, max_x, max_y = box
            width = max_x - min_x
            height = max_y - min_y
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            text = f"({width},{height})"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = center_x - text_width // 2
            text_y = center_y - text_height // 2
            draw.text((text_x, text_y), text, fill=(0, 0, 255), font=font)
        
        out_img.save(output_path)
        print(f'\nSaved: {output_path}')
        return bounding_box_list, valid_box_flags
    except Exception as e:
        print(f"Error in mark_centers_on_mask: {e}")

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