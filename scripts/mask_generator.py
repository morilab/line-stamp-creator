import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from scipy.ndimage import label, center_of_mass
import math
import os

def flood_fill_background(arr, bg_color, threshold=30):
    h, w, c = arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # 0: 未処理, 1: 背景
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    # 外周ピクセルをキューに追加
    for x in range(w):
        queue.append((0, x))
        queue.append((h-1, x))
    for y in range(h):
        queue.append((y, 0))
        queue.append((y, w-1))
    # Flood Fill
    while queue:
        y, x = queue.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True
        color = arr[y, x]
        if np.linalg.norm(color - bg_color) <= threshold:
            mask[y, x] = 1  # 背景
            # 4近傍
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0<=ny<h and 0<=nx<w and not visited[ny, nx]:
                    queue.append((ny, nx))
    return mask

def create_mask(image_path, mask_path, threshold=30):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    # 外周ピクセルの平均色を背景色とする
    border = 1
    top = arr[:border, :, :]
    bottom = arr[-border:, :, :]
    left = arr[:, :border, :]
    right = arr[:, -border:, :]
    outer_pixels = np.concatenate([top, bottom, left, right], axis=None).reshape(-1, arr.shape[2])
    bg_color = np.mean(outer_pixels, axis=0)
    # Flood Fillで背景領域を特定
    bg_mask = flood_fill_background(arr, bg_color, threshold)
    # 背景以外を白（255）、背景を黒（0）
    mask = np.where(bg_mask==1, 0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(mask_path)

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
        
        # PILのImageDrawを使用してテキストを描画
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
        for i in range(1, combined_num + 1):
            # 統合されたオブジェクトの座標を取得
            y_coords, x_coords = np.where(combined_labeled == i)
            if len(y_coords) == 0:
                continue
            
            # バウンディングボックスの座標を計算
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            
            # 枠線の太さと余白
            line_width = 4
            padding = 4  # オブジェクトからの余白
            
            # パディング込みのバウンディングボックス座標を記録
            box = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
            bounding_box_list.append(box)
            
            # 青枠のサイズを計算
            width = (max_x + padding) - (min_x - padding)
            height = (max_y + padding) - (min_y - padding)
            
            # 青枠の中心位置を計算
            center_y = (min_y - padding + max_y + padding) // 2
            center_x = (min_x - padding + max_x + padding) // 2
            
            # サイズを中心位置に表示
            text = f"({width},{height})"
            # テキストのサイズを取得
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # テキストの位置を計算（中央揃え）
            text_x = center_x - text_width // 2
            text_y = center_y - text_height // 2
            
            # テキストを描画（青色に変更）
            draw.text((text_x, text_y), text, fill=(0, 0, 255), font=font)
            
            # 枠線を描画（青）
            for w in range(line_width):
                # 上辺
                for x in range(min_x - padding - w, max_x + padding + w + 1):
                    if 0 <= min_y - padding - w < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                        arr_draw[min_y - padding - w, x] = [0, 0, 255]
                # 下辺
                for x in range(min_x - padding - w, max_x + padding + w + 1):
                    if 0 <= max_y + padding + w < arr_draw.shape[0] and 0 <= x < arr_draw.shape[1]:
                        arr_draw[max_y + padding + w, x] = [0, 0, 255]
                # 左辺
                for y in range(min_y - padding - w, max_y + padding + w + 1):
                    if 0 <= y < arr_draw.shape[0] and 0 <= min_x - padding - w < arr_draw.shape[1]:
                        arr_draw[y, min_x - padding - w] = [0, 0, 255]
                # 右辺
                for y in range(min_y - padding - w, max_y + padding + w + 1):
                    if 0 <= y < arr_draw.shape[0] and 0 <= max_x + padding + w < arr_draw.shape[1]:
                        arr_draw[y, max_x + padding + w] = [0, 0, 255]
            
            # 枠線を描画した後、テキストを再描画
            out_img = Image.fromarray(arr_draw)
            draw = ImageDraw.Draw(out_img)
            draw.text((text_x, text_y), text, fill=(0, 0, 255), font=font)
            arr_draw = np.array(out_img)
        
        out_img.save(output_path)
        print(f'\nSaved: {output_path}')
        return bounding_box_list
    except Exception as e:
        print(f"Error in mark_centers_on_mask: {e}")

def extract_bounding_box_segments(image_path, mask_path, bounding_box_list, output_path_prefix):
    """
    青枠のバウンディングボックス座標リストを受け取り、
    切り出し→リサイズ→透明化→保存（左上から右下の順）
    """
    img = Image.open(image_path).convert('RGBA')
    mask = Image.open(mask_path).convert('L')
    boxes = bounding_box_list
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
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    # inputフォルダ内のすべてのPNGファイルを処理
    input_dir = 'input'
    output_dir = 'output'
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            base = os.path.splitext(filename)[0]
            input_path = os.path.join(input_dir, filename)
            mask_path = os.path.join(output_dir, f'{base}_mask.png')
            bounding_path = os.path.join(output_dir, f'{base}_bounding.png')
            output_prefix = os.path.join(output_dir, base)
            # マスク生成
            create_mask(input_path, mask_path)
            # 青枠描画＋座標リスト取得
            bounding_box_list = mark_centers_on_mask(mask_path, bounding_path)
            # 青枠領域の切り出し
            extract_bounding_box_segments(input_path, mask_path, bounding_box_list, output_prefix) 