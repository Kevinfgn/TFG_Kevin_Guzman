# ==============================================================================
# optics_utils.py
# Módulo de utilidades para cálculo óptico de lentes, proporciones de distancia,
# niveles de seguridad y limitación de coordenadas. Se basa en el Modelo de Lente
# Delgada (Thin Lens Model) para estimar distancias de objetos detectados en video.
# ==============================================================================

from config import sensor_height_mm, sensor_height_px, focal_length, real_object_height
# sensor_height_mm   → Altura física del sensor de la cámara en milímetros.
# sensor_height_px   → Resolución vertical del sensor en píxeles.
# focal_length       → Distancia focal del lente en milímetros.
# real_object_height → Altura real conocida del objeto de referencia en milímetros.


# ------------------------------------------------------------------------------
# LensOpticCalculator
# Estima la distancia real entre la cámara y un objeto detectado,
# usando el Modelo de Lente Delgada (Thin Lens Model).
#
# Fórmula aplicada:
#   distancia = (H_real × f × sensor_px) / (h_px × sensor_mm)
#
# Parámetros:
#   px_height (float) → Altura del objeto detectado medida en píxeles dentro del frame.
#
# Retorna:
#   computed_object_distance (float) → Distancia estimada al objeto en milímetros.
#
# Nota: Si px_height es 0, se producirá una división por cero. Se recomienda
#       validar la entrada antes de llamar a esta función.
# ------------------------------------------------------------------------------
def LensOpticCalculator(px_height):
    computed_object_distance = (
        real_object_height * focal_length * sensor_height_px
    ) / (
        px_height * sensor_height_mm
    )
    return computed_object_distance


# ------------------------------------------------------------------------------
# RatioProportionCalculator
# Estima la distancia de objetos secundarios (no de referencia) usando
# proporción inversa respecto a un objeto de referencia cuya distancia
# real ya fue calculada con LensOpticCalculator.
#
# Principio:
#   A mayor valor en el mapa de profundidad (depthmap), el objeto está más
#   cerca. Se usa proporcionalidad inversa para estimar su distancia real.
#
#   distancia_objeto = (distancia_ref × punto_ref) / valor_depthmap_objeto
#
# Parámetros:
#   object_depthmap_Val (float) → Valor del mapa de profundidad del objeto secundario.
#   reference_distance  (float) → Distancia real calculada del objeto de referencia (mm).
#   reference_point     (float) → Valor del depthmap del objeto de referencia.
#
# Retorna:
#   computed_distance (float) → Distancia estimada del objeto secundario en milímetros.
# ------------------------------------------------------------------------------
def RatioProportionCalculator(object_depthmap_Val, reference_distance, reference_point):
    computed_distance = (reference_distance * reference_point) / object_depthmap_Val
    return computed_distance


# ------------------------------------------------------------------------------
# SafetyLevel
# Clasifica qué tan seguro es un objeto detectado según su distancia estimada,
# y asigna un color al bounding box de la detección para visualización en video.
#
# Umbral definido:
#   > 3000 mm (3 m) → Seguro       → Bounding box VERDE (0, 153, 76)
#   ≤ 3000 mm (3 m) → Peligro pot. → Bounding box ROJO  (153, 0, 0)
#
# Parámetros:
#   distance (float) → Distancia estimada al objeto en milímetros.
#
# Retorna:
#   dist       (str)   → Etiqueta de seguridad: "Safe" o "Potential Danger".
#   bbox_color (tuple) → Color BGR para dibujar el bounding box en OpenCV.
# ------------------------------------------------------------------------------
def SafetyLevel(distance):
    if distance > 3000:
        dist = "Safe"                 # Objeto a más de 3 metros → seguro
        bbox_color = (0, 153, 76)     # Verde en formato BGR (OpenCV)
    else:
        dist = "Potential Danger"     # Objeto a 3 metros o menos → peligro potencial
        bbox_color = (153, 0, 0)      # Rojo en formato BGR (OpenCV)
    return dist, bbox_color


# ------------------------------------------------------------------------------
# LimitVal
# Limita una coordenada para que no exceda los límites del frame de video.
# Evita errores de indexación cuando el centroide de una detección queda
# parcialmente fuera del área visible del feed de cámara.
#
# Lógica:
#   Si la coordenada está dentro del rango válido (<  val) → se mantiene igual.
#   Si la coordenada excede el límite             (>= val) → se recorta a val - 1.
#
# Parámetros:
#   coord (int) → Coordenada a validar (eje X o Y del centroide detectado).
#   val   (int) → Límite máximo permitido (ancho o alto del frame en píxeles).
#
# Retorna:
#   coord (int) → Coordenada corregida y segura para indexar el frame.
# ------------------------------------------------------------------------------
def LimitVal(coord, val):
    if coord < val:
        coord = coord      # La coordenada es válida, no se modifica
    else:
        coord = val - 1    # Se recorta al último índice válido del frame
    return coord