# Containts the function responsible for creating configurable sliders for the cropped ROI region.

import cv2
import numpy as np

def ROIConfigurator(cam):
    
    def on_trackbar(val):
        pass  

    cv2.namedWindow("ROI Size")

    # --- CORRECCIÓN AQUÍ ---
    # Verificamos si 'cam' es una cámara (VideoCapture) o una imagen (numpy array)
    if isinstance(cam, cv2.VideoCapture):
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # Si es una imagen (numpy.ndarray), usamos .shape
        # shape devuelve (alto, ancho, canales)
        height, width = cam.shape[:2]
    # -----------------------

    initial_widthcrop = int(width*.4)   # 40% del ancho inicial
    initial_heightcrop = int(height*.8) # 80% del alto inicial

    # Crear trackbars
    cv2.createTrackbar("Left-Right Crop", "ROI Size", initial_widthcrop, width, on_trackbar)
    cv2.createTrackbar("Top-Bottom Crop", "ROI Size", initial_heightcrop, height, on_trackbar)