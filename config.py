# ==============================================================================
# config.py
# Módulo central de inicialización del sistema. Carga todos los parámetros del
# archivo config.yml y construye los modelos de IA (YOLO + MiDaS) listos para
# ser importados por el resto de los módulos del sistema.
#
# Este archivo actúa como la única fuente de verdad de configuración:
# ningún otro módulo debe abrir config.yml directamente.
# ==============================================================================

import yaml
import cv2

# ------------------------------------------------------------------------------
# Carga del archivo de configuración YAML.
# yaml.safe_load parsea el archivo sin ejecutar código arbitrario (seguro).
# ------------------------------------------------------------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


# ==============================================================================
# SECCIÓN 1 — EXTRACCIÓN DE RUTAS DE MODELOS
# Todas las rutas a archivos .cfg, .weights, .names y .onnx provienen del YAML.
# Esto permite cambiar modelos sin tocar código Python.
# ==============================================================================

# Archivos de etiquetas (.names) para cada tipo de detector
coco_names          = config["model_path"]["coco_names"]           # Etiquetas COCO (80 clases generales, ej: cup, person)
vehicle_names       = config["model_path"]["vehicle_names"]        # Etiquetas de vehículos personalizados
licenseplate_names  = config["model_path"]["licenseplate_names"]   # Etiquetas de matrículas (uso alternativo)

# Archivos de arquitectura y pesos del modelo COCO (objeto target general)
coco_yolo_cfg       = config["model_path"]["coco_yolo_cfg"]        # Arquitectura YOLOv3 para COCO
coco_yolo_weights   = config["model_path"]["coco_yolo_weights"]    # Pesos pre-entrenados YOLOv3 COCO

# Archivos de arquitectura y pesos del modelo de vehículos (objetos circundantes)
vehicle_yolo_cfg     = config["model_path"]["vehicle_yolo_cfg"]    # Arquitectura YOLOv4 entrenado en vehículos
vehicle_yolo_weights = config["model_path"]["vehicle_yolo_weights"] # Pesos custom (checkpoint iter 10000)

# Archivos del modelo alternativo de matrículas (comentado en uso activo)
licenseplate_yolo_cfg     = config["model_path"]["licenseplate_yolo_cfg"]
licenseplate_yolo_weights = config["model_path"]["licenseplate_yolo_weights"]

# Ruta al modelo MiDaS en formato ONNX para estimación de profundidad monocular
MDE_model_path = config["model_path"]["MDE_model_path"]


# ==============================================================================
# SECCIÓN 2 — PARÁMETROS DE CÁMARA
# Usados por LensOpticCalculator para convertir píxeles a distancia real (mm).
# Estos valores deben corresponder a las especificaciones físicas de la cámara
# utilizada (en este caso, Honor Magic 7 Lite).
# ==============================================================================

sensor_height_mm = config["camera_information"]["sensor_height_mm"]
# Altura física del sensor en milímetros. Para sensor 1/1.67": ~5.75 mm.
# Representa cuánto espacio físico ocupa el sensor en la cámara.

sensor_height_px = config["camera_information"]["sensor_height_px"]
# Resolución vertical del sensor en píxeles (altura del frame capturado).
# Debe coincidir con la resolución real de captura (ej: 1080p → 1080 px).

focal_length = config["camera_information"]["focal_length"]
# Distancia focal del lente en milímetros.
# Determina el ángulo de visión y la magnitud del zoom óptico.
# Valor típico para el sensor del Honor Magic 7 Lite: ~5.25 mm.


# ==============================================================================
# SECCIÓN 3 — INFORMACIÓN DEL OBJETO TARGET
# Define qué objeto será el punto de referencia de distancia en cada sesión.
# ==============================================================================

real_object_height = config["target_object"]["real_object_height"]
# Altura física real del objeto target en milímetros.
# Valor crítico para LensOpticCalculator: a mayor precisión aquí,
# mayor precisión en la estimación de distancia. Ej: taza de café → 90 mm.

target = config["target_object"]["target"]
# Nombre del objeto target tal como aparece en el archivo .names de YOLO.
# Ej: "cup" para una taza, "Vehicle registration plate" para matrículas.


# ==============================================================================
# SECCIÓN 4 — CARGA DE ETIQUETAS (LABELS)
# Cada modelo YOLO requiere su propio archivo .names con las clases que detecta.
# Se leen como listas de strings, una clase por línea.
# ==============================================================================

# Etiquetas de objetos circundantes (vehículos personalizados)
# Usado por yolo_model (detector del entorno)
class_names = []
with open(vehicle_names, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Etiquetas del objeto target.
# Actualmente apunta a COCO (para detectar "cup" y otros objetos generales).
# Cambiar a licenseplate_names para el modo de detección de matrículas.
target_names = []
with open(coco_names, 'rt') as f:
    target_names = f.read().rstrip('\n').split('\n')

# Alternativa comentada: etiquetas para matrículas de vehículos
# with open(licenseplate_names, 'rt') as f:
#     target_names = f.read().rstrip('\n').split('\n')


# ==============================================================================
# SECCIÓN 5 — INICIALIZACIÓN DE MODELOS DNN
# Los tres modelos se cargan una sola vez aquí y se reutilizan en todo el sistema.
# Usar CPU como backend es compatible con cualquier hardware sin GPU dedicada.
# ==============================================================================

# ------------------------------------------------------------------------------
# Modelo 1: Detector de objetos circundantes (Vehículos — YOLOv4 custom)
# Se usa en FindObjects() para detectar todo el entorno fuera del ROI.
# ------------------------------------------------------------------------------
yolo_model = cv2.dnn.readNetFromDarknet(vehicle_yolo_cfg, vehicle_yolo_weights)
yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)   # Backend OpenCV optimizado para CPU
yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)         # Inferencia en CPU

# ------------------------------------------------------------------------------
# Modelo 2: Detector del objeto target (COCO — YOLOv3)
# Se usa en FindTargetObject() para localizar el objeto de referencia dentro del ROI.
# Modo alternativo: descomentando las líneas de licenseplate para usar ese modelo.
# ------------------------------------------------------------------------------
yolo_target_model = cv2.dnn.readNetFromDarknet(coco_yolo_cfg, coco_yolo_weights)
yolo_target_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_target_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Alternativa comentada: modelo de matrículas como target
# yolo_target_model = cv2.dnn.readNetFromDarknet(licenseplate_yolo_cfg, licenseplate_yolo_weights)
# yolo_target_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# yolo_target_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ------------------------------------------------------------------------------
# Modelo 3: Estimación de Profundidad Monocular — MiDaS (formato ONNX)
# Se usa en MonocularEstimator() para generar el mapa de profundidad de cada frame.
# El modelo "small" es más rápido; el "large" es más preciso pero más lento.
# ------------------------------------------------------------------------------
mde_model = cv2.dnn.readNet(MDE_model_path)
mde_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
mde_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Confirmación de inicialización exitosa en consola
print('Yolo Initialization Successful')
print('Depth Estimation Model Initialization Successful')