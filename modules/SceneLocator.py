# ==============================================================================
# SceneLocator.py  (FindObjects)
# Módulo encargado de detectar todos los objetos circundantes presentes en el
# frame completo (fuera del ROI), estimar su distancia usando proporción inversa
# respecto al objeto target, y clasificarlos según su nivel de seguridad.
#
# Depende de que FindTargetObject ya haya localizado el target y retornado
# su distancia real y valor MiDaS, que sirven como referencia de escala.
# ==============================================================================

import cv2
import numpy as np
from config import yolo_model, class_names
from modules.LensOpticCalculator import RatioProportionCalculator, LimitVal, SafetyLevel


# ------------------------------------------------------------------------------
# FindObjects
# Detecta objetos circundantes en el frame completo usando YOLOv4 (modelo de
# vehículos), estima su distancia por proporción inversa con MiDaS, y dibuja
# bounding boxes con color según nivel de seguridad (verde/rojo).
#
# Parámetros:
#   img                    (numpy.ndarray) → Frame RGB anotable donde se dibujan
#                                            los resultados de detección.
#   target_object_depth_val (tuple | None) → Tupla (distancia_mm, valor_midas)
#                                            del objeto target. None si no hay target.
#   monocular_depth_val    (numpy.ndarray) → Mapa de profundidad 2D normalizado [0,1]
#                                            generado por MonocularEstimator.
#
# Retorna:
#   None → Todos los resultados se dibujan directamente sobre img y monocular_depth_val.
#
# Comportamiento:
#   - Si target_object_depth_val es válido (target presente): ejecuta detección completa.
#   - Si target_object_depth_val es None/vacío (target ausente): muestra mensaje de aviso.
# ------------------------------------------------------------------------------
def FindObjects(img, target_object_depth_val, monocular_depth_val):

    # Condición de entrada: solo se ejecuta detección si el target fue encontrado.
    # bool() sobre una tupla es True si no está vacía; False si es None o vacía.
    if bool(target_object_depth_val):

        # Crea una copia limpia del frame para la inferencia YOLO.
        # Se usa img_copy para la detección y img para el dibujo de resultados,
        # evitando que las anotaciones previas interfieran con la red neuronal.
        img_copy = img.copy()

        # Extrae los valores de referencia del objeto target para los cálculos
        # de proporción inversa en RatioProportionCalculator.
        reference_distance = target_object_depth_val[0]   # Distancia real del target en mm
        reference_point    = target_object_depth_val[1]   # Valor MiDaS del target [0,1]

        # ----------------------------------------------------------------------
        # Umbrales de detección YOLO
        # ----------------------------------------------------------------------
        conf_threshold = 0.7   # Confianza mínima para aceptar una detección (70%).
                               # Detecciones con score < 0.7 se descartan.
        nms_threshold  = 0.4   # Umbral de Non-Maximum Suppression.
                               # Valor bajo → más agresivo → menos bounding boxes duplicados.

        # ----------------------------------------------------------------------
        # Preprocesamiento: conversión del frame a blob para YOLOv4.
        # Parámetros idénticos a FindTargetObject para consistencia.
        # Input size (320×320): balance entre velocidad y precisión en CPU.
        # ----------------------------------------------------------------------
        blob = cv2.dnn.blobFromImage(img_copy, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
        yolo_model.setInput(blob)

        # Obtiene los nombres de las capas de salida (cabezas de detección de YOLO).
        # YOLO tiene múltiples escalas de salida para detectar objetos de distinto tamaño.
        output_names = yolo_model.getUnconnectedOutLayersNames()

        # Ejecuta el forward pass del modelo y obtiene todas las detecciones.
        detection = yolo_model.forward(output_names)

        # Dimensiones del frame de trabajo para escalar coordenadas normalizadas de YOLO.
        hT, wT, cT = img_copy.shape

        # Listas acumuladoras para post-procesamiento con NMS.
        bbox      = []   # Bounding boxes crudos [x, y, w, h] en píxeles
        class_ids = []   # Índice de clase detectada por cada bbox
        confs     = []   # Puntuación de confianza de cada detección

        # ----------------------------------------------------------------------
        # Bucle de parseo de detecciones YOLO.
        # Cada 'output' es una escala de detección; cada 'det' es un objeto candidato.
        # Los primeros 4 valores de 'det' son coordenadas normalizadas (cx, cy, w, h).
        # Los valores desde índice 5 en adelante son scores por clase.
        # ----------------------------------------------------------------------
        for output in detection:
            for det in output:
                scores     = det[5:]                    # Vector de scores para cada clase
                class_id   = np.argmax(scores)          # Clase con el score más alto
                confidence = scores[class_id]           # Score de esa clase

                if confidence > conf_threshold:
                    # Convierte coordenadas normalizadas YOLO → píxeles del frame.
                    # YOLO reporta centro (cx, cy) y tamaño (w, h) normalizados.
                    w = int(det[2] * wT)
                    h = int(det[3] * hT)
                    x = int((det[0] * wT) - w / 2)     # Esquina superior izquierda X
                    y = int((det[1] * hT) - h / 2)     # Esquina superior izquierda Y

                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

            # NMS: elimina bounding boxes redundantes que detectan el mismo objeto.
            # Se aplica dentro del loop por escala para respetar la estructura de YOLO.
            indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)
            indices = np.array(indices).flatten()   # Convierte a array 1D para iterar

        # ----------------------------------------------------------------------
        # Bucle de anotación: procesa solo los bounding boxes supervivientes al NMS.
        # ----------------------------------------------------------------------
        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Calcula el centroide del bounding box (punto central del objeto).
            xcoord = (x + (x + w)) // 2
            ycoord = (y + (y + h)) // 2

            # Aplica LimitVal para asegurar que el centroide no excede los bordes
            # del frame, evitando errores de indexación en el mapa de profundidad.
            xcoord = LimitVal(xcoord, wT)
            ycoord = LimitVal(ycoord, hT)

            # Recupera el valor de profundidad relativa del objeto circundante
            # del mapa MiDaS en la posición exacta de su centroide.
            object_depthmap_val = monocular_depth_val[int(ycoord), int(xcoord)]

            # Dibuja un punto magenta en el centroide sobre el mapa de profundidad
            # para visualización de qué píxel se está muestreando.
            cv2.circle(monocular_depth_val, (int(xcoord), int(ycoord)), 3, (255, 0, 255), -1)

            # Muestra el valor MiDaS crudo del objeto sobre el frame anotado.
            # Útil para depuración y calibración del sistema.
            cv2.putText(img, str(round(object_depthmap_val, 5)),
                        (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Estima la distancia real del objeto circundante usando proporción inversa.
            # Usa el target como referencia: distance_obj = (dist_ref × midas_ref) / midas_obj
            output_face = RatioProportionCalculator(
                object_depthmap_val, reference_distance, reference_point
            )

            # Determina el nivel de seguridad y el color del bounding box.
            # safety[0] = "Safe" / "Potential Danger"
            # safety[1] = color BGR (verde si seguro, rojo si peligro)
            safety = SafetyLevel(output_face)

            # Línea comentada: muestra la etiqueta de seguridad textual sobre el frame.
            # cv2.putText(img, "Distance Safety Level: " + f'{safety[0]}', ...)

            # Dibuja el bounding box con el color de seguridad correspondiente.
            cv2.rectangle(img, (x, y), (x + w, y + h), (safety[1]), 2)

            # Muestra nombre de clase (en mayúsculas) y distancia estimada en mm
            # junto al borde superior del bounding box.
            cv2.putText(img,
                        f'{class_names[class_ids[i]].upper()}- ' + str(round(output_face, 2)),
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (safety[1]), 2)

            # Dibuja un punto con el color de seguridad en el centroide del objeto
            # sobre el frame principal para referencia visual de qué se muestreó.
            cv2.circle(img, (int(xcoord), int(ycoord)), 3, (safety[1]), -1)

    # --------------------------------------------------------------------------
    # Rama alternativa: target no presente en el frame.
    # Se muestra un mensaje guía al operador para que coloque el objeto de referencia.
    # --------------------------------------------------------------------------
    else:
        cv2.putText(img, "Place Target Object within the ROI",
                    (300, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)