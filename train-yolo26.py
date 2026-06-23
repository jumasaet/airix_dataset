from ultralytics import YOLO

# 1. Carga el modelo pre-entrenado de RT-DETR
# Se recomienda empezar con rtdetr-l.pt (Large)
model = YOLO("yolo26l.pt")

IMGZ=1024

# 2. Entrena el modelo con tu archivo .yaml de datos
# Asegúrate de que 'data.yaml' tenga la estructura correcta (path, train, val, nc, names)
model.train(
    data="/home/admin-cidis/airix_dataset/airix_final/data.yaml", 
    epochs=100, 
    imgsz=IMGZ, 
    batch=8, 
    project="/home/admin-cidis/airix_dataset/yolo26l", 
    name=str(IMGZ) + "_px",
    optimizer='AdamW',         # Define el optimizador explícitamente
    lr0=0.0001,                # <<-- MÁS IMPORTANTE: Baja el LR drásticamente
    lrf=0.01,                  # Final learning rate
    amp=True,                  # Mantén esto activo
    device="0"
)