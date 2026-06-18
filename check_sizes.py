import os
from collections import Counter
from PIL import Image
from tqdm import tqdm

def analyze_all_datasets(base_path, top_n=10):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    global_sizes = []
    
    # Buscar todas las carpetas 'train' dentro de los directorios .coco
    # O simplemente cualquier subcarpeta que contenga imágenes
    subdirs = [f.path for f in os.scandir(base_path) if f.is_dir() and not f.name.startswith('.')]
    
    print(f"Se encontraron {len(subdirs)} subdirectorios potenciales para analizar.\n")
    
    for subdir in subdirs:
        # Apuntar a la carpeta 'train' si existe, si no, analizar el subdirectorio raíz
        target_dir = os.path.join(subdir, 'train')
        if not os.path.exists(target_dir):
            target_dir = subdir # Por si alguna carpeta como '15' tiene imágenes directo
            
        # Listar imágenes válidas
        try:
            files = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
        except Exception:
            continue # Si no se puede acceder o no es un directorio válido
            
        if not files:
            continue

        print(f"Analizando subdataset: {os.path.basename(subdir)} ({len(files)} imágenes)...")
        local_sizes = []

        for filename in tqdm(files, desc=os.path.basename(subdir)[:20]):
            img_path = os.path.join(target_dir, filename)
            try:
                with Image.open(img_path) as img:
                    # (width, height)
                    local_sizes.append(img.size)
            except Exception as e:
                print(f" -> Error en {filename}: {e}")

        # Añadir al acumulador global
        global_sizes.extend(local_sizes)
        
        # Mostrar top local
        local_counts = Counter(local_sizes)
        print(f"-> Tamaños más comunes en {os.path.basename(subdir)}:")
        for size, count in local_counts.most_common(3):
            print(f"   {str(size):<15} : {count} imágenes")
        print("-" * 60)

    # --- Resumen Global ---
    if not global_sizes:
        print("No se encontraron imágenes válidas en ningún subdirectorio.")
        return

    global_counts = Counter(global_sizes)
    most_common_global = global_counts.most_common(top_n)

    print("\n" + "="*20 + " RESUMEN GLOBAL DEL DATASET " + "="*20)
    print(f"Total de imágenes analizadas: {len(global_sizes)}")
    print(f"Total de resoluciones distintas: {len(global_counts)}")
    print("\n--- Top Tamaños Globales ---")
    print(f"{'Ranking':<10} | {'Resolución (W x H)':<20} | {'Cantidad'} | {'Porcentaje'}")
    print("-" * 65)
    
    for i, (size, count) in enumerate(most_common_global, 1):
        percentage = (count / len(global_sizes)) * 100
        print(f"{i:<10} | {str(size):<20} | {count:<13} | {percentage:.2f}%")

if __name__ == "__main__":
    # Tu nueva ruta en la partición montada de Linux
    ruta_dataset = '/media/john/Windows-SSD/Dataset_Cacaos'
    analyze_all_datasets(ruta_dataset)