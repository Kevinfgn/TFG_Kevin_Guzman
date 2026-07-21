# ==============================================================================
# roi_configurator.py
# Módulo para crear una ventana interactiva con sliders (trackbars) que permiten
# configurar en tiempo real el tamaño de la Región de Interés (ROI) recortada
# del feed de video o de una imagen estática.
# ==============================================================================

import cv2
import numpy as np


# ------------------------------------------------------------------------------
# ROIConfigurator
# Crea una ventana de OpenCV con dos trackbars interactivos que permiten al
# usuario ajustar visualmente el recorte horizontal y vertical de la ROI,
# sin necesidad de modificar valores en el código.
#
# La función acepta tanto un objeto VideoCapture (cámara en vivo) como un
# array NumPy (imagen estática), detectando automáticamente cuál es.
#
# Parámetros:
#   cam (cv2.VideoCapture | numpy.ndarray) → Fuente de video o imagen de referencia
#                                            usada para determinar las dimensiones del frame.
#
# Trackbars creados en la ventana "ROI Size":
#   "Left-Right Crop" → Controla el recorte horizontal (ancho de la ROI).
#                        Rango: 0 a width. Valor inicial: 40% del ancho total.
#   "Top-Bottom Crop" → Controla el recorte vertical (alto de la ROI).
#                        Rango: 0 a height. Valor inicial: 80% del alto total.
#
# Retorna:
#   None → Los valores de los trackbars se leen externamente con cv2.getTrackbarPos().
# ------------------------------------------------------------------------------
def ROIConfigurator(cam):

    # Callback vacío requerido por OpenCV para createTrackbar.
    # No realiza ninguna acción; la lectura del valor se hace manualmente
    # en el loop principal con cv2.getTrackbarPos().
    def on_trackbar(val):
        pass

    # Crea la ventana con nombre fijo donde se alojarán los sliders.
    cv2.namedWindow("ROI Size")

    # -------------------------------------------------------------------------
    # Detección del tipo de entrada para extraer dimensiones del frame.
    # Se distingue entre una cámara en vivo y una imagen estática,
    # ya que cada una expone sus dimensiones de forma diferente.
    # -------------------------------------------------------------------------
    if isinstance(cam, cv2.VideoCapture):
        # Si es una cámara activa, se consultan las propiedades del stream.
        width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # Si es un array NumPy (imagen cargada con cv2.imread o similar),
        # shape devuelve (alto, ancho, canales) → se extraen los dos primeros.
        height, width = cam.shape[:2]

    # -------------------------------------------------------------------------
    # Valores iniciales de los trackbars expresados como porcentaje del frame.
    # 40% del ancho  → punto de partida razonable para recorte lateral.
    # 80% del alto   → conserva la mayor parte del frame verticalmente.
    # -------------------------------------------------------------------------
    initial_widthcrop  = width    # Recorte horizontal inicial (40% del ancho)
    initial_heightcrop = height   # Recorte vertical inicial   (80% del alto)

    # Crea el trackbar de recorte horizontal.
    # Rango: 0 (sin recorte lateral) a width (recorte total).
    cv2.createTrackbar("Left-Right Crop", "ROI Size", initial_widthcrop, width,  on_trackbar)

    # Crea el trackbar de recorte vertical.
    # Rango: 0 (sin recorte vertical) a height (recorte total).
    cv2.createTrackbar("Top-Bottom Crop", "ROI Size", initial_heightcrop, height, on_trackbar)