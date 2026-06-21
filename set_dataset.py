import os
import re
import shutil
import zipfile
import cv2
import tempfile
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
ZIP_PATH = "/home/admin-cidis/airix_dataset/Cocoa_Detection.yolo26.zip" # <-- Actualizado
OUTPUT_DIR = "/home/admin-cidis/airix_dataset/airix_final"              # <-- Actualizado
MAX_SIZE = 1024

# Nombres de tus clases en el orden exacto
CLASES_YOLO = ['disease', 'healthy'] # <-- Actualizado

MAPEO_NOMBRES = {
    "nenesVert": "airix_01",
    "IMM": "airix_02",
    "trainhoriz": "airix_03",
    "nesnes_horiz": "airix_04",
    "valnes": "airix_05",
    "train_vert": "airix_06"
}

MAPEO_SPLIT = {
    'airix_03': 'train', 'airix_06': 'train', 'airix_04': 'train',
    'San': 'train', 'airix_01': 'train', 'plaga': 'train', 'train': 'train',
    'airix_05': 'test',
    'airix_02': 'val', '312': 'val'
}

# ==========================================
# 🚀 LÓGICA DE PROCESAMIENTO
# ==========================================
def procesar_dataset_directo(zip_path, output_dir, max_size, mapeo_nombres, mapeo_split, clases):
    out_path = Path(output_dir)
    
    # 1. Crear estructura YOLO de destino
    for split in ['train', 'val', 'test']:
        (out_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'test': 0, 'omitidos': 0, 'redimensionados': 0}
    
    # Diccionario para guardar el reporte detallado del txt
    stats_per_prefix = {}
    
    print(f"📦 Descomprimiendo {zip_path}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        temp_path = Path(temp_dir)
        imagenes = list(temp_path.rglob('*.jpg')) + list(temp_path.rglob('*.png')) + list(temp_path.rglob('*.jpeg'))
        
        print(f"🔍 Se encontraron {len(imagenes)} imágenes. Iniciando procesamiento en 1 pasada...\n")
        
        for img_path in imagenes:
            lbl_name = img_path.stem + ".txt"
            lbl_path = img_path.parent.parent / 'labels' / lbl_name
            if not lbl_path.exists():
                lbl_path = img_path.with_suffix('.txt')
            
            # --- LIMPIEZA DE NOMBRE ---
            base_name = img_path.name.split('.rf.')[0]
            base_name = re.sub(r'_(jpg|png|jpeg)$', '', base_name, flags=re.IGNORECASE)
            
            match = re.search(r'^(.*?)_(\d+)$', base_name)
            if match:
                old_prefix = match.group(1)
                number = int(match.group(2))
            else:
                old_prefix = base_name
                number = None
                
            # --- MAPEO DE NOMBRE ---
            new_prefix = mapeo_nombres.get(old_prefix, old_prefix)
            
            if number is not None:
                new_base = f"{new_prefix}_{number:05d}"
            else:
                new_base = new_prefix
                
            # --- MAPEO DE SPLIT ---
            destino_split = mapeo_split.get(new_prefix)
            if not destino_split:
                if new_base.startswith('Sin_Prefijo'):
                    destino_split = 'train'
                else:
                    print(f"⚠️ Prefijo '{new_prefix}' no mapeado en el split. Omitiendo {img_path.name}")
                    stats['omitidos'] += 1
                    continue
            
            # --- REGISTRAR EN EL REPORTE DINÁMICO ---
            if new_prefix not in stats_per_prefix:
                stats_per_prefix[new_prefix] = {
                    'old_prefix': old_prefix,
                    'split': destino_split,
                    'count': 0
                }
            stats_per_prefix[new_prefix]['count'] += 1
            
            # --- RUTAS FINALES ---
            dest_img = out_path / destino_split / 'images' / f"{new_base}{img_path.suffix}"
            dest_lbl = out_path / destino_split / 'labels' / f"{new_base}.txt"
            
            # --- LECTURA, RESIZE Y GUARDADO ---
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            lado_largo = max(h, w)
            
            if lado_largo > max_size:
                factor = max_size / lado_largo
                nuevo_w, nuevo_h = int(w * factor), int(h * factor)
                img = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
                stats['redimensionados'] += 1
                
            cv2.imwrite(str(dest_img), img)
            
            if lbl_path.exists():
                shutil.copy2(lbl_path, dest_lbl)
                
            stats[destino_split] += 1

    # 2. Generar el archivo log de equivalencias, cantidades y split
    log_path = out_path / "airix_dataset_mapping.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE SUB-DATASETS, CANTIDADES Y SPLITS\n")
        f.write("="*75 + "\n")
        f.write(f"{'PREFIJO ACTUAL':<20} | {'PREFIJO ORIGINAL':<20} | {'CANTIDAD':<10} | {'SPLIT ASIGNADO'}\n")
        f.write("-" * 75 + "\n")
        
        # Ordenar alfabéticamente por prefijo nuevo
        for new_pref, data in sorted(stats_per_prefix.items()):
            f.write(f"{new_pref:<20} | {data['old_prefix']:<20} | {data['count']:<10} | {data['split'].upper()}\n")

    # 3. Generar archivo data.yaml para YOLO
    yaml_path = out_path / "data.yaml"
    yaml_content = f"""train: {out_path / 'train' / 'images'}
val: {out_path / 'val' / 'images'}
test: {out_path / 'test' / 'images'}

nc: {len(clases)}
names: {clases}
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # --- RESUMEN FINAL ---
    total = stats['train'] + stats['val'] + stats['test']
    print("✅ PROCESO FINALIZADO CON ÉXITO")
    print("-" * 45)
    print(f"📍 Destino: {out_path}")
    print(f"📄 Log creado: airix_dataset_mapping.txt (¡Ahora con cantidades!)")
    print(f"🛠️ Archivo YOLO creado: data.yaml")
    print(f"🔄 Imágenes redimensionadas: {stats['redimensionados']}")
    print("-" * 45)
    if total > 0:
        print(f"Train: {stats['train']} imágenes ({(stats['train']/total)*100:.1f}%)")
        print(f"Val:   {stats['val']} imágenes ({(stats['val']/total)*100:.1f}%)")
        print(f"Test:  {stats['test']} imágenes ({(stats['test']/total)*100:.1f}%)")
    print("-" * 45)
    print(f"Total procesado final: {total}")
    if stats['omitidos'] > 0:
        print(f"Archivos omitidos: {stats['omitidos']}")

# --- EJECUCIÓN ---
procesar_dataset_directo(ZIP_PATH, OUTPUT_DIR, MAX_SIZE, MAPEO_NOMBRES, MAPEO_SPLIT, CLASES_YOLO)