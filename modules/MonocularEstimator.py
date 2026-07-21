# ==============================================================================
# depth_estimator.py
# Módulo para generar el Mapa de Profundidad (Depth Map) del feed de entrada
# usando el modelo MiDaS de Estimación de Profundidad Monocular (MDE).
# ==============================================================================

import cv2
from config import mde_model
# mde_model → Modelo MiDaS pre-cargado desde config (red neuronal DNN de OpenCV).


# ------------------------------------------------------------------------------
# MonocularEstimator
# Genera un mapa de profundidad normalizado a partir de un frame de imagen,
# usando el modelo MiDaS (Monocular Depth Estimation) ejecutado vía OpenCV DNN.
#
# El mapa de profundidad asigna a cada píxel un valor relativo de profundidad:
#   - Valores cercanos a 1 → objetos más CERCANOS a la cámara.
#   - Valores cercanos a 0 → objetos más LEJANOS a la cámara.
#
# Parámetros:
#   img (numpy.ndarray) → Frame de imagen BGR capturado del video feed.
#
# Retorna:
#   monocular_output (numpy.ndarray) → Mapa de profundidad 2D normalizado (0.0 a 1.0),
#                                      con las mismas dimensiones que el frame de entrada.
#
# Nota sobre el modelo:
#   - Modelo pequeño (default): blob de entrada (256×256).
#   - Modelo grande (mayor precisión): cambiar blob a (384×384) en blobFromImage.
# ------------------------------------------------------------------------------
def MonocularEstimator(img):

    # Extrae las dimensiones del frame original para restaurarlas después.
    img_height, img_width, channels = img.shape

    # Convierte el frame en un blob compatible con la red neuronal.
    # Parámetros de blobFromImage:
    #   img              → Frame de entrada en BGR.
    #   1/255.           → Factor de escala: normaliza valores de píxel de [0,255] a [0,1].
    #   (256,256)        → Tamaño de entrada requerido por MiDaS (small). Usar (384,384) para large.
    #   (123.675, 116.28, 103.53) → Valores de media ImageNet para sustracción por canal (R, G, B).
    #   True             → swapRB=True: convierte BGR → RGB antes de procesar.
    #   False            → crop=False: no recorta, solo redimensiona al tamaño indicado.
    blob = cv2.dnn.blobFromImage(
        img, 1/255., (256, 256),
        (123.675, 116.28, 103.53),
        True, False
    )

    # Alimenta el blob al modelo MiDaS cargado en memoria.
    mde_model.setInput(blob)

    # Ejecuta el forward pass de la red y obtiene el mapa de profundidad crudo.
    # La salida tiene forma (1, H_red, W_red), es decir, un batch de 1 con el mapa 2D.
    monocular_output = mde_model.forward()

    # Elimina la dimensión de batch → queda un array 2D (H_red, W_red).
    monocular_output = monocular_output[0, :, :]

    # Redimensiona el mapa de profundidad al tamaño original del frame de entrada,
    # ya que MiDaS lo procesa internamente en 256×256 (o 384×384).
    monocular_output = cv2.resize(monocular_output, (img_width, img_height))

    # Normaliza los valores del mapa al rango [0.0, 1.0] usando min-max normalization.
    # Esto hace los valores comparables entre frames y útiles para cálculos de proporción.
    monocular_output = cv2.normalize(
        monocular_output, None,
        0, 1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F
    )

    # Retorna el mapa de profundidad 2D normalizado listo para procesamiento posterior.
    return monocular_output