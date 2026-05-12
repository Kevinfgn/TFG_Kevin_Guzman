# ==============================================================================
# TargetObjectLocator.py  (FindTargetObject)
# Módulo encargado de localizar el objeto target (objeto de referencia) dentro
# de la Región de Interés (ROI) central del frame, calcular su distancia real
# con la ecuación óptica de lente delgada, y extraer su valor del mapa MiDaS.
#
# Es el primer paso del pipeline de estimación: sin un target detectado,
# FindObjects no puede calcular distancias de objetos circundantes.
# ==============================================================================

import cv2
import numpy as np
from config import yolo_target_model, target_names
from modules.LensOpticCalculator import LensOpticCalculator, LimitVal


# ------------------------------------------------------------------------------
# FindTargetObject
# Detecta el objeto target dentro del ROI configurado por el usuario, calcula
# su distancia real en mm usando la fórmula de lente delgada, y retorna tanto
# esa distancia como su valor en el mapa de profundidad MiDaS.
#
# Parámetros:
#   img       (numpy.ndarray) → Frame RGB completo. Se usa para dibujar el ROI
#                               y el centroide del target sobre el feed principal.
#   target    (str)           → Nombre de la clase target tal como aparece en
#                               el archivo .names (ej: "cup").
#   mde_Model (numpy.ndarray) → Mapa de profundidad 2D normalizado [0,1]
#                               generado por MonocularEstimator.
#
# Retorna:
#   (target_computed_depthmap_val, target_midas_val) (tuple) si el target fue
#   detectado: distancia real en mm y valor MiDaS en el centroide del target.
#
#   None (implícito) si el target no fue encontrado en el ROI.
# ------------------------------------------------------------------------------
def FindTargetObject(img, target, mde_Model):

    # Crea una copia limpia del frame para la inferencia YOLO dentro del ROI.
    # Evita que anotaciones previas sobre img alteren la detección.
    img_copy = img.copy()
    img_height, img_width, channels = img_copy.shape

    # --------------------------------------------------------------------------
    # Lectura de los valores actuales de los sliders ROI.
    # Los trackbars son leídos en tiempo real cada frame, permitiendo al operador
    # ajustar la ROI interactivamente sin reiniciar el sistema.
    # --------------------------------------------------------------------------
    top_bottom_crop = cv2.getTrackbarPos("Top-Bottom Crop", "ROI Size")  # Recorte vertical en px
    left_right_crop = cv2.getTrackbarPos("Left-Right Crop", "ROI Size")  # Recorte horizontal en px

    # --------------------------------------------------------------------------
    # Cálculo de coordenadas de la ROI centrada en el frame.
    # La ROI siempre se centra en el frame; los sliders controlan su tamaño.
    # x1, x2: límites horizontales. y1, y2: límites verticales.
    # --------------------------------------------------------------------------
    x1 = int(img_width  / 2 - left_right_crop / 2)   # Borde izquierdo de la ROI
    x2 = int(img_width  / 2 + left_right_crop / 2)   # Borde derecho de la ROI
    y1 = int(img_height / 2 - top_bottom_crop / 2)   # Borde superior de la ROI
    y2 = int(img_height / 2 + top_bottom_crop / 2)   # Borde inferior de la ROI

    # Recorta la ROI del frame copiado para pasársela a YOLO.
    # YOLO solo verá esta región, reduciendo falsas detecciones fuera del área de interés.
    img_roi = img_copy[y1:y2, x1:x2]

    # Dibuja el rectángulo de la ROI sobre el frame principal (negro) como referencia visual.
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # --------------------------------------------------------------------------
    # Construcción del diccionario de indexación de etiquetas YOLO.
    # YOLO devuelve class_id como índice entero; este diccionario permite buscar
    # el índice de una clase por su nombre (ej: "cup" → 41 en COCO).
    # --------------------------------------------------------------------------
    yolo_labels_indexing = {label: index for index, label in enumerate(target_names)}

    # Offsets para traducir coordenadas detectadas dentro del ROI al frame completo.
    # Sin estos offsets, los puntos quedarían referenciados al origen del ROI (0,0),
    # no al origen del frame global.
    xoff = x1   # Desplazamiento horizontal: inicio del ROI en el frame completo
    yoff = y1   # Desplazamiento vertical: inicio del ROI en el frame completo

    # --------------------------------------------------------------------------
    # Umbrales de detección YOLO (iguales a FindObjects para consistencia).
    # --------------------------------------------------------------------------
    conf_threshold = 0.7   # Confianza mínima del 70% para aceptar una detección
    nms_treshold   = 0.4   # NMS: menor valor → más agresivo → menos cajas duplicadas

    # --------------------------------------------------------------------------
    # Preprocesamiento de la ROI como blob para YOLOv3.
    # Se pasa img_roi (no img_copy) para que YOLO solo analice la región de interés.
    # --------------------------------------------------------------------------
    blob = cv2.dnn.blobFromImage(img_roi, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    yolo_target_model.setInput(blob)

    # Nombres de las capas de salida del modelo target (múltiples escalas de YOLO).
    output_names = yolo_target_model.getUnconnectedOutLayersNames()

    # Ejecuta inferencia sobre el blob de la ROI.
    detection = yolo_target_model.forward(output_names)

    # Dimensiones de la ROI para escalar coordenadas normalizadas de YOLO.
    hT, wT, cT = img_roi.shape

    # Listas acumuladoras de detecciones para post-procesamiento con NMS.
    bbox      = []   # Bounding boxes en píxeles de la ROI
    class_ids = []   # Índices de clase de cada detección
    confs     = []   # Scores de confianza de cada detección

    # --------------------------------------------------------------------------
    # Bucle de parseo: igual que en FindObjects pero sobre la ROI recortada.
    # Convierte las coordenadas normalizadas de YOLO a píxeles de la ROI.
    # --------------------------------------------------------------------------
    for output in detection:
        for det in output:
            scores     = det[5:]
            class_id   = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                w = int(det[2] * wT)
                h = int(det[3] * hT)
                x = int((det[0] * wT) - w / 2)
                y = int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    # Aplica NMS para eliminar bounding boxes redundantes del mismo objeto.
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_treshold)
    indices = np.array(indices).flatten()

    # --------------------------------------------------------------------------
    # Búsqueda del objeto target entre las detecciones supervivientes al NMS.
    # Se busca por class_id numérico, no por nombre, usando el diccionario de
    # indexación construido anteriormente.
    # --------------------------------------------------------------------------
    target_index = yolo_labels_indexing[target]   # Índice numérico del target en .names

    if target_index in class_ids:
        # Localiza todas las detecciones que coinciden con el target y toma la primera.
        # Si hay múltiples detecciones del mismo objeto, se usa la de mayor confianza
        # (que YOLO ya ordena internamente).
        indexes = np.where(np.array(class_ids) == target_index)[0]
        box1    = bbox[indexes[0]]
        x, y, w, h = box1[0], box1[1], box1[2], box1[3]

        # Calcula el centroide del bounding box dentro del sistema de coordenadas ROI.
        xcoord = (x + (x + w)) / 2
        ycoord = (y + (y + h)) / 2

        # Traslada el centroide al sistema de coordenadas del frame completo
        # sumando los offsets del ROI.
        xoffset_coord = xcoord + xoff
        yoffset_coord = ycoord + yoff

        # Aplica LimitVal para garantizar que el centroide no excede los bordes
        # del frame completo al indexar el mapa de profundidad.
        xoffset_coord = LimitVal(xoffset_coord, img_width)
        yoffset_coord = LimitVal(yoffset_coord, img_height)

        # Convierte la ROI de RGB a BGR para que OpenCV muestre los colores
        # correctamente en la ventana "ROI" (OpenCV espera BGR).
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2BGR)

        # Dibuja el bounding box del target en verde sobre la ventana ROI.
        cv2.rectangle(img_roi, (x, y), (x + w, y + h), (0, 153, 76), 2)

        # Dibuja un círculo negro en el centroide del target sobre el frame completo,
        # referenciado en coordenadas globales con los offsets aplicados.
        cv2.circle(img, (int(xoffset_coord), int(yoffset_coord)), 3, (0, 0, 0), 2)

        # Recupera el valor de profundidad MiDaS en el centroide del target.
        # Este valor [0,1] será el reference_point para RatioProportionCalculator.
        target_midas_val = mde_Model[int(yoffset_coord), int(xoffset_coord)]

        # Calcula la distancia real del target en mm usando la fórmula de lente delgada.
        # Se pasa 'h' (altura del bounding box en píxeles dentro de la ROI) como
        # px_height: a mayor h en píxeles, menor distancia estimada.
        target_computed_depthmap_val = LensOpticCalculator(h)

        # Muestra la ventana de la ROI con el bounding box del target dibujado.
        cv2.imshow('ROI', img_roi)

        # Retorna la tupla que FindObjects usará como referencia de escala.
        # [0] = distancia real en mm (de LensOpticCalculator)
        # [1] = valor MiDaS normalizado en el centroide (de mde_Model)
        return (target_computed_depthmap_val, target_midas_val)

    # Si el target no fue encontrado en el ROI, la función retorna None implícitamente.
    # FindObjects detectará esto con bool(None) → False y mostrará el mensaje de aviso.