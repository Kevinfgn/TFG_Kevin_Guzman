# Contains the initialization of necessary variables, models, and etc comming from the config.yml file

import yaml
import cv2

# Load the YAML config file
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# --- Extracción de Rutas desde YAML ---
coco_names = config["model_path"]["coco_names"]
vehicle_names = config["model_path"]["vehicle_names"]
licenseplate_names = config["model_path"]["licenseplate_names"]

coco_yolo_cfg = config["model_path"]["coco_yolo_cfg"]
coco_yolo_weights = config["model_path"]["coco_yolo_weights"]

vehicle_yolo_cfg = config["model_path"]["vehicle_yolo_cfg"]
vehicle_yolo_weights = config["model_path"]["vehicle_yolo_weights"]

licenseplate_yolo_cfg = config["model_path"]["licenseplate_yolo_cfg"]
licenseplate_yolo_weights = config["model_path"]["licenseplate_yolo_weights"]

MDE_model_path = config["model_path"]["MDE_model_path"]

# --- Información de Cámara y Target ---
sensor_height_mm = config["camera_information"]["sensor_height_mm"] 
sensor_height_px = config["camera_information"]["sensor_height_px"] 
focal_length = config["camera_information"]["focal_length"]

real_object_height = config["target_object"]["real_object_height"]
target = config["target_object"]["target"]

# --- CARGA DE ETIQUETAS (LABELS) ---

# Etiquetas para objetos del entorno (Generalmente vehículos)
class_names = []
with open(vehicle_names, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Etiquetas para el objeto objetivo (Target)
# Cambiamos a coco_names porque tu target actual es 'cup'
target_names = []
with open(coco_names, 'rt') as f: 
    target_names = f.read().rstrip('\n').split('\n') 

# --- OPCIÓN ORIGINAL (COMENTADA POR SI DESEAS VOLVER A MATRÍCULAS) ---
# with open(licenseplate_names, 'rt') as f:
#     target_names = f.read().rstrip('\n').split('\n') 


# --- INICIALIZACIÓN DE MODELOS DNN ---

# 1. Modelo de Objetos Circundantes (Vehículos)
yolo_model = cv2.dnn.readNetFromDarknet(vehicle_yolo_cfg, vehicle_yolo_weights)
yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 2. Modelo del Objeto Target (COCO)
# Nota: Aquí se usa coco_yolo_cfg y coco_yolo_weights
yolo_target_model = cv2.dnn.readNetFromDarknet(coco_yolo_cfg, coco_yolo_weights)
yolo_target_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_target_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# --- OPCIÓN PARA MATRÍCULAS ---
# yolo_target_model = cv2.dnn.readNetFromDarknet(licenseplate_yolo_cfg, licenseplate_yolo_weights)
# yolo_target_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# yolo_target_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# 3. Inicialización del modelo MDE (Monocular Depth Estimation)
mde_model = cv2.dnn.readNet(MDE_model_path)
mde_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
mde_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Checkpoint de éxito
print('Yolo Initialization Successful')
print('Depth Estimation Model Initialization Successful')