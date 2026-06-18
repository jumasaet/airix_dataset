import os
import json

def coco_to_yolo_bbox(img_w, img_h, coco_bbox):
    # Forzar que el ancho y alto de la imagen sean flotantes
    img_w = float(img_w)
    img_h = float(img_h)
    
    # COCO bbox: [x_min, y_min, width, height]
    # Forzamos a que cada elemento de la lista sea un float
    x_min, y_min, w, h = [float(val) for val in coco_bbox]
    
    # YOLO requiere: x_center, y_center, width, height (normalizados 0-1)
    x_center = x_min + (w / 2.0)
    y_center = y_min + (h / 2.0)
    
    # Normalizar
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    
    return x_center, y_center, w, h
    
def convert_datasets(base_path):
    subdirs = [f.path for f in os.scandir(base_path) if f.is_dir() and not f.name.startswith('.')]
    
    for subdir in subdirs:
        train_dir = os.path.join(subdir, 'train')
        json_path = os.path.join(train_dir, '_annotations.coco.json')
        
        if not os.path.exists(json_path):
            continue
            
        print(f"Convirtiendo labels en: {os.path.basename(subdir)}...")
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        # Crear mapeo de image_id -> (file_name, width, height)
        images_map = {}
        for img in coco_data['images']:
            images_map[img['id']] = {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }
            
        # Agrupar anotaciones por imagen
        annotations_by_img = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_img:
                annotations_by_img[img_id] = []
            annotations_by_img[img_id].append(ann)
            
        # Crear los archivos .txt de YOLO
        for img_id, img_info in images_map.items():
            file_name = img_info['file_name']
            base_name, _ = os.path.splitext(file_name)
            txt_path = os.path.join(train_dir, f"{base_name}.txt")
            
            anns = annotations_by_img.get(img_id, [])
            
            with open(txt_path, 'w') as txt_file:
                for ann in anns:
                    cat_id = ann['category_id'] # Verifica si Roboflow los indexó 0, 1, 2
                    coco_bbox = ann['bbox']
                    
                    # Convertir coordenadas
                    x, y, w, h = coco_to_yolo_bbox(img_info['width'], img_info['height'], coco_bbox)
                    
                    # Escribir en formato YOLO
                    txt_file.write(f"{cat_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    ruta_base = '/media/john/Windows-SSD/Dataset_Cacaos'
    convert_datasets(ruta_base)