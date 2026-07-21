# ==============================================================================
# main.py
# Punto de entrada principal del sistema de detección de objetos y estimación
# de distancia en tiempo real. Orquesta el pipeline completo:
#
#   Entrada (imagen/video)
#     → Preprocesamiento y redimensionado
#       → Mapa de profundidad MiDaS
#         → Detección y distancia del objeto target (ROI)
#           → Detección de objetos circundantes
#             → Visualización de resultados
#
# Soporta tres modos de entrada: cámara en vivo, archivo de video, imagen estática.
# ==============================================================================

import cv2
import os
import time
import numpy as np
from collections import deque

from config import target
from modules.SceneLocator import FindObjects
from modules.ROIConfigurator import ROIConfigurator
from modules.MonocularEstimator import MonocularEstimator
from modules.TargetObjectLocator import FindTargetObject


# ==============================================================================
# SECCIÓN 1 — CONFIGURACIÓN DE ENTRADA
# Solo una fuente debe estar activa. Comentar/descomentar según el modo deseado.
# ==============================================================================

# cam = cv2.VideoCapture(0)               # Modo: cámara en vivo (índice 0 = cámara principal)
cam = cv2.VideoCapture('vid/cup.mp4') # Modo: archivo de video
# cam = cv2.imread('img/cup90.jpeg')       # Modo: imagen estática (procesamiento en loop)

# Validación: si la fuente no se pudo cargar, se aborta el programa.
if cam is None:
    print("Error: No se pudo cargar la imagen o fuente de video. Revisa la ruta.")
    exit()

# Detecta el tipo de fuente para adaptar el comportamiento del loop principal.
# VideoCapture → video en vivo o archivo. ndarray → imagen estática.
is_video = isinstance(cam, cv2.VideoCapture)

# Bandera para imprimir resultados solo una vez en modo imagen (evita spam en consola).
impreso_una_vez = False


# ==============================================================================
# SECCIÓN 2 — FUNCIÓN DE REDIMENSIONAMIENTO
# Normaliza la resolución de entrada a 720px de alto para garantizar rendimiento
# consistente en el procesamiento de IA, independiente de la resolución original.
# ==============================================================================
def optimizar_imagen(frame, altura_objetivo=720):
    """
    Redimensiona un frame manteniendo la relación de aspecto.
    Usa INTER_AREA, el método óptimo para reducción de tamaño sin artefactos.

    Parámetros:
        frame (numpy.ndarray) → Frame BGR a redimensionar.
        altura_objetivo (int) → Alto deseado en píxeles (default: 720).

    Retorna:
        frame redimensionado (numpy.ndarray) o el frame original si ya es menor.
    """
    h, w = frame.shape[:2]
    if h > altura_objetivo or is_video:
        ratio = altura_objetivo / h              # Factor de escala proporcional
        nueva_anchura = int(w * ratio)
        return cv2.resize(frame, (nueva_anchura, altura_objetivo), interpolation=cv2.INTER_AREA)
    return frame  # Si el frame ya es pequeño (imagen), no se modifica


# ==============================================================================
# SECCIÓN 3 — PREPROCESAMIENTO INICIAL (solo imágenes estáticas)
# En modo imagen, el redimensionado se hace UNA SOLA VEZ antes del loop,
# ya que el frame no cambia entre iteraciones.
# ==============================================================================
if not is_video:
    cam = optimizar_imagen(cam)

# Inicializa la ventana de sliders ROI con las dimensiones del frame de entrada.
# En modo video usa el objeto VideoCapture; en modo imagen, el array ya redimensionado.
ROIConfigurator(cam)


# ==============================================================================
# SECCIÓN 3.5 — INICIALIZACIÓN DE GRABACIÓN DE VIDEO
# Solo se inicializa cuando la fuente es video (cámara en vivo o archivo).
# En modo imagen estática no se crean ni escriben archivos de grabación.
# ==============================================================================
if is_video:
    # Obtiene las dimensiones reales del frame procesado mediante un frame de prueba
    success_test, frame_test_raw = cam.read()
    if not success_test:
        print("Error: No se pudo leer un frame de prueba para inicializar la grabación.")
        exit()
    frame_test = optimizar_imagen(frame_test_raw)
    # Regresa el video al inicio para que el loop principal empiece desde el frame 0
    cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_h, frame_w = frame_test.shape[:2]

    # Codec XVID — compatible con .avi en la mayoría de sistemas sin dependencias extra
    fourcc        = cv2.VideoWriter_fourcc(*'XVID')
    fps_grabacion = 20  # FPS objetivo para los archivos de video grabados

    # VideoWriter para el feed principal anotado (detecciones + distancias)
    writer_main  = cv2.VideoWriter('output_deteccion.avi', fourcc, fps_grabacion, (frame_w, frame_h))

    # VideoWriter para el mapa de profundidad MiDaS.
    # El depthmap es un array float32 de un solo canal [0,1]; se convierte a uint8 BGR
    # antes de escribir (OpenCV VideoWriter requiere frames BGR uint8).
    writer_depth = cv2.VideoWriter('output_depthmap.avi',  fourcc, fps_grabacion, (frame_w, frame_h))

    print(f"[REC] Grabando en: output_deteccion.avi | output_depthmap.avi  ({frame_w}x{frame_h} @ {fps_grabacion} fps objetivo)")

# Variables para el cálculo de FPS reales en tiempo de ejecución
fps_actual = 0.0

# Buffer para suavizado temporal de la distancia del target
distancia_buffer = deque(maxlen=8)  # Promedia los últimos 8 frames


# ==============================================================================
# SECCIÓN 4 — LOOP PRINCIPAL DE PROCESAMIENTO
# Cada iteración procesa un frame completo a través del pipeline de IA.
# En modo imagen, el mismo frame se reprocesa en cada iteración del loop
# (permite ajustar la ROI interactivamente con los sliders).
# ==============================================================================
while True:

    # Marca de tiempo al inicio del frame para calcular FPS reales
    frame_start = time.time()

    # --------------------------------------------------------------------------
    # 4.1 — Adquisición del frame
    # --------------------------------------------------------------------------
    if is_video:
        success, img_raw = cam.read()   # Lee el siguiente frame del video/cámara
        if not success:
            break                       # Fin del video o error de captura → salir
        img = optimizar_imagen(img_raw) # Normaliza resolución en cada frame de video
    else:
        img = cam.copy()                # En modo imagen, copia el frame estático
                                        # (evita acumulación de dibujos entre iteraciones)

    # --------------------------------------------------------------------------
    # 4.2 — Preparación del frame
    # --------------------------------------------------------------------------
    img_height, img_width, channels = img.shape

    # Convierte BGR → RGB porque MiDaS y YOLO fueron entrenados con imágenes RGB.
    # OpenCV carga en BGR por defecto; la conversión es necesaria para inferencia correcta.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------------------------
    # 4.3 — Generación del mapa de profundidad (MiDaS)
    # Produce un array 2D normalizado [0,1] donde cada píxel representa
    # la profundidad relativa: 1 = más cerca, 0 = más lejos.
    # --------------------------------------------------------------------------
    monocular_depth_val = MonocularEstimator(img_rgb)

    # --------------------------------------------------------------------------
    # 4.4 — Detección del objeto target dentro del ROI
    # Localiza el objeto de referencia (ej: "cup") en la región central del frame,
    # calcula su distancia real con LensOpticCalculator y extrae su valor MiDaS.
    # Retorna (distancia_mm, valor_midas) o None si el target no está presente.
    # --------------------------------------------------------------------------
    target_object_depth_val = FindTargetObject(img_rgb, target, monocular_depth_val)

    # --------------------------------------------------------------------------
    # 4.5 — Impresión de resultados en consola
    # Comportamiento diferenciado: imagen → imprime una vez; video → actualiza en línea.
    # --------------------------------------------------------------------------
    if target_object_depth_val is not None:
        distancia, valor_midas = target_object_depth_val

        if not is_video and not impreso_una_vez:
            # Modo imagen: imprime bloque detallado una sola vez
            print("\n" + "="*40)
            print(f"DETECCION EXITOSA: {target.upper()}")
            print(f"Distancia al objeto: {distancia / 10:.2f} cm")    # mm → cm
            print(f"Confianza de profundidad (MiDaS): {valor_midas:.4f}")
            print("="*40 + "\n")
            impreso_una_vez = True

        elif is_video:
            # Modo video: actualiza la misma línea de consola en tiempo real
            print(
                f"[LIVE] Target: {target} | "
                f"Distancia: {distancia / 10:.2f} cm | "
                f"Midas: {valor_midas:.3f}  \n  ",
                end="\r"
            )

    # --------------------------------------------------------------------------
    # 4.5.1 — Dibuja la distancia del target sobre el frame principal
    # Se dibuja sobre img_rgb para que quede en la grabación y en la ventana.
    # --------------------------------------------------------------------------
    if target_object_depth_val is not None:
        distancia, valor_midas = target_object_depth_val

        # Agrega la distancia actual al buffer y calcula el promedio suavizado
        distancia_buffer.append(distancia)
        distancia_suavizada = sum(distancia_buffer) / len(distancia_buffer)

        texto_distancia = f"Target: {target.upper()}"
        cv2.putText(img_rgb, texto_distancia, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # Limpia el buffer cuando el target desaparece para evitar valores obsoletos
        distancia_buffer.clear()
        cv2.putText(img_rgb, "Target no detectado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --------------------------------------------------------------------------
    # 4.6 — Detección de objetos circundantes (entorno)
    # Usa el valor del target como referencia para estimar distancias de otros
    # objetos detectados por el modelo de vehículos en el frame completo.
    # Si target_object_depth_val es None, muestra mensaje de "coloca el target".
    # --------------------------------------------------------------------------
    FindObjects(img_rgb, target_object_depth_val, monocular_depth_val)

    # --------------------------------------------------------------------------
    # 4.7 — Visualización de resultados
    # Se reconvierte RGB → BGR para que OpenCV muestre los colores correctamente
    # en las ventanas (los dibujos se hicieron sobre img_rgb durante el pipeline).
    # --------------------------------------------------------------------------
    img_final = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Redimensiona las ventanas de visualización sin afectar la grabación
    display_scale  = 0.6
    img_display    = cv2.resize(img_final,           (int(img_width  * display_scale), int(img_height * display_scale)))
    depth_display  = cv2.resize(monocular_depth_val, (int(img_width  * display_scale), int(img_height * display_scale)))

    cv2.imshow('1. Mapa de Calor (Depth Map)', depth_display)   # Mapa de profundidad en escala de grises
    cv2.imshow('2. Deteccion y Distancia (TFG)', img_display)   # Frame anotado con detecciones y distancias

    # --------------------------------------------------------------------------
    # 4.7.5 — Grabación de ambos feeds en sus respectivos archivos de video
    # Solo graba cuando la fuente es video; en modo imagen no se escribe nada.
    # El depthmap (float32, 1 canal, rango [0,1]) se convierte a uint8 BGR antes
    # de escribirlo para cumplir el formato que VideoWriter espera.
    # --------------------------------------------------------------------------
    if is_video:
        writer_main.write(img_final)   # Frame principal anotado (ya en BGR uint8)

        depth_uint8  = (monocular_depth_val * 255).astype(np.uint8)   # Escala [0,1] → [0,255]
        depth_bgr    = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)  # 1 canal → 3 canales BGR
        writer_depth.write(depth_bgr)                                  # Escribe frame del depthmap

    # --------------------------------------------------------------------------
    # 4.7.6 — Cálculo e impresión de FPS reales en consola
    # Se mide el tiempo transcurrido desde el inicio del frame hasta este punto,
    # cubriendo todo el pipeline (inferencia + detección + anotación + grabación).
    # --------------------------------------------------------------------------
    frame_end  = time.time()
    fps_actual = 1.0 / (frame_end - frame_start) if (frame_end - frame_start) > 0 else 0.0
    print(f"[FPS] {fps_actual:.1f} fps", end="\r")

    # --------------------------------------------------------------------------
    # 4.8 — Control de salida
    # waitKey(1): espera 1 ms entre frames (necesario para que OpenCV renderice).
    # Tecla 'q' termina el loop limpiamente.
    # --------------------------------------------------------------------------
    key = cv2.waitKey(1)
    if key == ord("q"):
        break


# ==============================================================================
# SECCIÓN 5 — LIMPIEZA DE RECURSOS
# Libera la cámara y los archivos de grabación solo si se usó modo video.
# En modo imagen solo se cierran las ventanas de visualización.
# ==============================================================================
if is_video:
    cam.release()          # Libera el handle del archivo de video o cámara física
    writer_main.release()  # Finaliza y cierra el archivo output_deteccion.avi
    writer_depth.release() # Finaliza y cierra el archivo output_depthmap.avi
    print(f"\n[REC] Grabacion finalizada: output_deteccion.avi | output_depthmap.avi")

cv2.destroyAllWindows()    # Cierra todas las ventanas de visualización abiertas