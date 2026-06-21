import os
import json
import glob
# Puedes instalarlo con: pip install imagesize
import imagesize 

def yolo_to_coco_subset(base_path, subset, categories):
    images_dir = os.path.join(base_path, subset, 'images')
    labels_dir = os.path.join(base_path, subset, 'labels')
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Extensiones de imagen comunes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        
    annotation_id = 1
    image_id = 1
    
    print(f"Procesando {subset}: se encontraron {len(image_paths)} imágenes.")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        # Buscar el archivo .txt correspondiente
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(labels_dir, lbl_name)
        
        # Obtener dimensiones reales de la imagen sin cargarla en memoria pesada
        try:
            width, height = imagesize.get(img_path)
        except Exception as e:
            print(f"Error al leer dimensiones de {img_name}: {e}")
            continue

        # Agregar info de la imagen al estilo COCO
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_name, # MMDetection buscará este nombre dentro de la carpeta del subset
            "width": width,
            "height": height
        })
        
        # Si existe el archivo de anotación YOLO, procesarlo
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id, x_center, y_center, bbox_w, bbox_h = map(float, parts)
                class_id = int(class_id)
                
                # De normalizado (YOLO) a Píxeles absolutos (COCO)
                # YOLO: x_center, y_center, w, h (0 a 1)
                # COCO: x_min, y_min, w, h (en píxeles)
                w_px = bbox_w * width
                h_px = bbox_h * height
                x_min_px = (x_center * width) - (w_px / 2)
                y_min_px = (y_center * height) - (h_px / 2)
                
                # Asegurar que no se salgan de los bordes por errores de redondeo
                x_min_px = max(0, x_min_px)
                y_min_px = max(0, y_min_px)
                w_px = min(w_px, width - x_min_px)
                h_px = min(h_px, height - y_min_px)
                
                area = w_px * h_px
                
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,  # COCO suele empezar en 1, pero MMDetection acepta desde 0 si coincide con tu config.
                    "bbox": [round(x_min_px, 2), round(y_min_px, 2), round(w_px, 2), round(h_px, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                annotation_id += 1
                
        image_id += 1

    # Guardar el JSON resultante en la carpeta del subset
    output_json = os.path.join(base_path, subset, f'labels_coco.json')
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"¡Listo! Archivo guardado en: {output_json}\n")

if __name__ == "__main__":
    # 1. DEFINE TUS CLASES AQUÍ (revisa tu archivo data.yaml para verificar los nombres)
    # Ejemplo si tu data.yaml dice: names: ['cacao_bueno', 'cacao_enfermo']
    # MMDetection maneja mejor los IDs si mantienes el mismo orden que en YOLO (empezando desde 0 o 1). 
    # Lo dejaremos desde 0 para hacer match directo con los TXT de YOLO.
    
    CATEGORIES = [
        {"id": 0, "name": "disease"},
        {"id": 1, "name": "healthy"},
        # Agrega más clases si tu dataset de cacao tiene más categorías
    ]
    
    BASE_DIR = os.path.dirname("/home/admin-cidis/airix_dataset/airix_final/")
    
    # Instalar imagesize si no está disponible
    try:
        import imagesize
    except ImportError:
        os.system('pip install imagesize')
        import imagesize

    # Ejecutar para los tres subsets
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        print("Checking " + subset)
        if os.path.exists(os.path.join(BASE_DIR, subset)):
            yolo_to_coco_subset(BASE_DIR, subset, CATEGORIES)