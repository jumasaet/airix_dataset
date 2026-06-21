# AIRIX

## 1. Descargar dataset

Se debe descargar el dataset directamente de roboflow, del proyecto "segairix" exportando con el formato YOLO en formato ZIP, una vez hecho esto correr el script "set_dataset.py" colocando la ruta del zip dentro de la carpeta, esto tendrá como salida el dataset con los splits y los resizes ya hechos.

## 2. Seteo para MMDetection

Al parecer esta herramienta se suele usar para hacer benchmarks ya que entrena con las mismas condiciones a todos los detectores, mismos aumentos, trato de tamaño de imágenes, etc. Esta herramienta usa los labels en formato COCO así que después de ajustar las rutas debidas se debe correr el "yolo_to_coco.py", hecho esto ya se puede hacer entrenamientos.

## 3. MMDetection
Seguir las instrucciones de mmdet.txt