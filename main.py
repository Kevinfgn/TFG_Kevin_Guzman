# Initialization of Libraries and Dependencies

import cv2
import os
import numpy as np

from config import target
from modules.SceneLocator import FindObjects
from modules.ROIConfigurator import ROIConfigurator
from modules.MonocularEstimator import MonocularEstimator
from modules.TargetObjectLocator import FindTargetObject

# --- OPCIONES DE ENTRADA (DESCOMENTA LA QUE NECESITES) ---

#cam = cv2.VideoCapture(0)                                                                                                       # To test webacm use "cv2.VideoCapture(0)" 0 is the default value, change whenever necessary.
#cam = cv2.VideoCapture('/Users/espiedeguzman/Desktop/Untitled.mp4')                                                              # To test on pre recorded videos replace "0" with the file path and file.
cam = cv2.imread('img/cup.jpg')                                                                                                 # To test on images replace "cv2.VideoCapture(0)" with "cv2.imread('file path and file name')".

# Identificar si es video o imagen para la lógica del bucle
is_video = isinstance(cam, cv2.VideoCapture)
impreso_una_vez = False # Control para la consola

ROIConfigurator(cam)

while True:
    if is_video:
        success, img = cam.read()
        if not success:
            break
    else:
        img = cam.copy() # Si es imagen, usamos una copia para cada ciclo

    # Procesamiento de imagen
    img_height, img_width, channels = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convierte a RGB para los modelos

    # Ejecución de Modelos
    monocular_depth_val = MonocularEstimator(img_rgb)                                                                           # Executes the MDE Model.                                                   
    target_object_depth_val = FindTargetObject(img_rgb, target, monocular_depth_val)                                            # YOLO Object Detection execution for Target Object.
    
    # --- LÓGICA DE IMPRESIÓN EN CONSOLA ---
    if target_object_depth_val is not None:
        distancia, valor_midas = target_object_depth_val
        
        # Si es imagen, imprime solo una vez. Si es video, imprime siempre.
        if not is_video and not impreso_una_vez:
            print(f"\n[INFO] Target: {target} | Distancia: {distancia / 10:.2f} mm | Valor Midas: {valor_midas}")
            impreso_una_vez = True
        elif is_video:
            # En video imprimimos con retorno de carro (\r) para no llenar la consola hacia abajo
            print(f"[LIVE] Target: {target} | Distancia: {distancia / 10:.2f} mm | Midas: {valor_midas}      ", end="\r")

    FindObjects(img_rgb, target_object_depth_val, monocular_depth_val)                                                          # YOLO Object Detection execution for Surrounding Object and Lens Optic Calculation.

    # Volver a BGR para mostrar en ventanas de OpenCV
    img_final = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Depth Map', monocular_depth_val)                                                                                # Shows MDE Generated Depth Map.
    cv2.imshow('Monocular Depth Estimation', img_final)                                                                         # Video Feed with Results.

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Limpieza final
if is_video:
    cam.release()
cv2.destroyAllWindows()