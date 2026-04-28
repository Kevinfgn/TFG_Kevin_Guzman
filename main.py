# Initialization of Libraries and Dependencies

import cv2
import os
import numpy as np

from config import target
from modules.SceneLocator import FindObjects
from modules.ROIConfigurator import ROIConfigurator
from modules.MonocularEstimator import MonocularEstimator
from modules.TargetObjectLocator import FindTargetObject

# --- OPCIONES DE ENTRADA ---

# cam = cv2.VideoCapture(0)                                                                                                       
# cam = cv2.VideoCapture('vid/demo3.mp4')                                                                                             
cam = cv2.imread('img/cuph2.jpeg')                                                                                                 

if cam is None:
    print("Error: No se pudo cargar la imagen o fuente de video. Revisa la ruta.")
    exit()

# Identificar si es video o imagen
is_video = isinstance(cam, cv2.VideoCapture)
impreso_una_vez = False 

# --- FUNCIÓN DE REDIMENSIONAMIENTO OPTIMIZADA ---
def optimizar_imagen(frame, altura_objetivo=720):
    """
    Redimensiona la imagen para que el procesamiento sea fluido.
    720px es el estándar que definimos en config.yml para el Honor Magic 7 Lite.
    """
    h, w = frame.shape[:2]
    if h > altura_objetivo or is_video:
        ratio = altura_objetivo / h
        nueva_anchura = int(w * ratio)
        # INTER_AREA es el mejor método para reducir tamaño sin perder nitidez técnica
        return cv2.resize(frame, (nueva_anchura, altura_objetivo), interpolation=cv2.INTER_AREA)
    return frame

# Si es imagen, la optimizamos UNA SOLA VEZ antes de entrar al bucle
if not is_video:
    cam = optimizar_imagen(cam)

# Inicializamos el configurador de ROI con la imagen optimizada
ROIConfigurator(cam)

while True:
    if is_video:
        success, img_raw = cam.read()
        if not success:
            break
        img = optimizar_imagen(img_raw)
    else:
        img = cam.copy() 

    # 1. Preparación de dimensiones y color
    img_height, img_width, channels = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # 2. Ejecución de Modelos de IA
    # MonocularEstimator genera el mapa de calor de profundidad
    monocular_depth_val = MonocularEstimator(img_rgb)                                                   
    
    # FindTargetObject localiza el objeto (ej. 'cup') y calcula la distancia
    target_object_depth_val = FindTargetObject(img_rgb, target, monocular_depth_val)                                            
    
    # 3. Lógica de Impresión de resultados
    if target_object_depth_val is not None:
        distancia, valor_midas = target_object_depth_val
        
        # IMPORTANTE
        if not is_video and not impreso_una_vez:
            print("\n" + "="*40)
            print(f"DETECCION EXITOSA: {target.upper()}")
            print(f"Distancia al objeto: {distancia / 10:.2f} cm")
            print(f"Confianza de profundidad (MiDaS): {valor_midas:.4f}")
            print("="*40 + "\n")
            impreso_una_vez = True
            
        elif is_video:
            # En video usamos \r para actualizar la misma línea
            print(f"[LIVE] Target: {target} | Distancia: {distancia / 10:.2f} cm | Midas: {valor_midas:.3f}    ", end="\r")

    # 4. Detección de objetos circundantes (Entorno)
    FindObjects(img_rgb, target_object_depth_val, monocular_depth_val)                                                          

    # 5. Visualización de resultados
    # Convertimos de nuevo a BGR para que OpenCV muestre los colores correctamente
    img_final = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('1. Mapa de Calor (Depth Map)', monocular_depth_val)                                                                                
    cv2.imshow('2. Deteccion y Distancia (TFG)', img_final)                                                                         

    # Salir con la tecla 'q'
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Limpieza de recursos
if is_video:
    cam.release()
cv2.destroyAllWindows()