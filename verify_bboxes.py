import os
import cv2
import glob

def draw_yolo_bbox(img, label_path, base_path_to_categories=None):
    h_img, w_img, _ = img.shape
    
    # --- CÁLCULO DINÁMICO DE TAMAÑOS ---
    # Usamos la diagonal o el ancho para escalar
    scale_factor = img.shape[1] / 1000 # Escalar basado en un ancho de 1000px
    line_thickness = max(1, int(4 * scale_factor))  # Grosor de línea base 4
    font_scale = max(0.4, 1.2 * scale_factor)      # Escala de fuente base 1.2
    font_thickness = max(1, int(3 * scale_factor)) # Grosor de fuente base 3
    # -------------------------------------

    if not os.path.exists(label_path):
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Colores para diferentes clases (hasta 5, luego repite)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]

    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue
            
        # El formato YOLO es: class_id x_center y_center width height (normalizados)
        try:
            cat_id_int = int(data[0])
            cat_id, x_c, y_c, w, h = cat_id_int, float(data[1]), float(data[2]), float(data[3]), float(data[4])
        except ValueError:
            print(f"Error parseando linea en {label_path}: {line}")
            continue

        # Desnormalizar a píxeles absolutos
        x_center = x_c * w_img
        y_center = y_c * h_img
        width = w * w_img
        height = h * h_img

        # Calcular esquinas (x_min, y_min) y (x_max, y_max)
        x1 = int(x_center - (width / 2.0))
        y1 = int(y_center - (height / 2.0))
        x2 = int(x_center + (width / 2.0))
        y2 = int(y_center + (height / 2.0))

        color = colors[cat_id_int % len(colors)]

        # Dibujar el rectángulo con GROSOR ADAPTATIVO
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Dibujar la etiqueta de clase con TAMAÑO ADAPTATIVO
        label_text = f"Clase: {cat_id}"
        
        # Fondo para el texto (para mejorar legibilidad)
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - int(1.3 * text_height)), (x1 + text_width, y1), color, -1) # -1 llena el rectángulo

        cv2.putText(img, label_text, (x1, y1 - int(0.3 * text_height)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return img

def verify_datasets(base_path, max_images=5):
    # Buscar todos los subdirectorios
    subdirs = [f.path for f in os.scandir(base_path) if f.is_dir() and not f.name.startswith('.')]
    
    cv2.namedWindow("Verificacion de BBox adaptive", cv2.WINDOW_NORMAL) # Ventana redimensionable

    for subdir in subdirs:
        train_dir = os.path.join(subdir, 'train')
        if not os.path.exists(train_dir):
            continue
            
        print(f"\n--- Mostrando hasta {max_images} imágenes de: {os.path.basename(subdir)} ---")
        print("Controles:")
        print("  [CUALQUIER TECLA] -> Siguiente imagen")
        print("  [q]             -> Salir del script")
        print("  [s]             -> Saltar este sub-dataset")
        
        # Buscar imágenes (soporta jpg, jpeg y png)
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in img_extensions:
            image_paths.extend(glob.glob(os.path.join(train_dir, ext)))
            
        # Tomar solo las primeras N imágenes
        sampled_images = image_paths[:max_images]
        
        if not sampled_images:
            print(f"No se encontraron imágenes en {train_dir}")
            continue

        for img_path in sampled_images:
            # Obtener el archivo .txt correspondiente
            base_name, _ = os.path.splitext(img_path)
            txt_path = f"{base_name}.txt"
            
            # Leer imagen
            img = cv2.imread(img_path)
            if img is None:
                print(f"No se pudo cargar la imagen: {img_path}")
                continue
                
            # Dibujar las cajas CON TAMAÑOS ADAPTATIVOS
            img_with_boxes = draw_yolo_bbox(img, txt_path)
            
            # Mostrar ventana (cv2.WINDOW_NORMAL permite que se ajuste sola)
            cv2.imshow("Verificacion de BBox adaptive", img_with_boxes)
            
            # Esperar tecla
            key = cv2.waitKey(0) & 0xFF # 0xFF para compatibilidad
            
            # Controles
            if key == ord('q'):
                print("Verificación cancelada por el usuario.")
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                print(f"Saltando dataset: {os.path.basename(subdir)}")
                break # Salta al siguiente subdir

    cv2.destroyAllWindows()
    print("\n¡Verificación completada para todos los datasets!")

if __name__ == "__main__":
    ruta_base = '/media/john/Windows-SSD/Dataset_Cacaos'
    verify_datasets(ruta_base, max_images=5)