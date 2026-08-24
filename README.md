# Módulo de Detección de Objetos y Estimación de Distancia Monocular en Video para Sistemas de Análisis Vial

Los avances en robótica y sistemas autónomos han incrementado la necesidad de soluciones precisas de medición de profundidad, especialmente en tareas de detección y evitación de obstáculos para navegación autónoma. Tradicionalmente, la medición de profundidad se realiza mediante sensores de distancia, escáneres 3D, LIDAR o cámaras estéreo; sin embargo, su costo elevado limita su adopción en aplicaciones de bajo presupuesto. Esto ha impulsado el interés en soluciones monoculares de estimación de profundidad, que utilizan modelos de aprendizaje profundo para generar mapas de disparidad a partir de una única cámara.

**El reto de las soluciones monoculares de estimación de profundidad (MDE) es su confiabilidad en aplicaciones que requieren precisión.** Los resultados de los modelos MDE suelen representarse como valores normalizados en lugar de medidas físicas reales, lo que limita su interpretación y aplicación directa sin un sistema de control adicional.

**Este repositorio contiene el código fuente del sistema desarrollado como Trabajo Final de Graduación (TFG) en el CIVCO (Centro de Investigación en Vivienda y Construcción)**, cuyo objetivo es actualizar los valores del mapa de profundidad generado a valores de distancia reales, aplicado a un sistema de análisis vial. El método relaciona conceptos de óptica física (modelo de cámara estenopeica/pinhole) con detección de objetos YOLO para calcular distancias reales en escena.

## Módulo de Estimación de Profundidad Monocular

A lo largo de este proyecto se utilizó **MiDaS ONNX (small)**, un modelo MDE entrenado mezclando múltiples datasets para cubrir entornos diversos, lo que lo hace robusto para distintas aplicaciones y adecuado para ejecución en tiempo real sobre CPU.

Más información en: https://github.com/isl-org/MiDaS

## Método de Actualización y Conceptos Físicos Utilizados

### Modelo de cámara pinhole

Para transformar los valores del mapa de profundidad en medidas físicas (mm), se infirieron relaciones a partir del modelo de cámara estenopeica (pinhole), combinando la información de la cámara con la altura en píxeles de un objeto de referencia con altura física conocida. Esta relación se usa para calibrar dinámicamente el sistema en cada ejecución.

### Limitaciones

El método de actualización está limitado por:
- La precisión de la información de calibración de la cámara.
- La necesidad de un **Objeto de Referencia** (objeto con altura física conocida) presente en la escena.
- Que dicho objeto de referencia se mantenga perpendicular a la cámara.
- Vulnerabilidad ante manipulación del objeto de referencia, lo cual genera un error proporcional sistemático que se propaga a toda la escena (se proponen mitigaciones como validación de relación de aspecto y validación cruzada con doble referencia).

### Detección de objetos YOLO

Se utilizaron dos instancias de YOLO:
- Una instancia **pre-entrenada en COCO**, usada para detectar el objeto de referencia.
- Una instancia **YOLO personalizada** (entrenada con datos propios etiquetados en Roboflow) para detectar vehículos circundantes.

Las salidas de bounding box de YOLO se utilizan para obtener la altura en píxeles del objeto de referencia y para ubicar los centroides de los objetos circundantes dentro del mapa de profundidad.

### Relación MiDaS - YOLO

Una vez calculada la distancia real del objeto de referencia y ubicado su centroide en el mapa de profundidad, esta distancia sirve como valor de anclaje para relacionar, por proporción inversa, las distancias del resto de objetos detectados en la escena (los valores del mapa de profundidad de MiDaS son normalizados entre 0 y 1, donde valores cercanos a 0 representan mayor lejanía).

### Suavizado de distancia

Se implementó un filtro de promedio móvil basado en `deque` para suavizar las estimaciones de distancia y reducir la variabilidad frame a frame.

### Nivel de seguridad (SafetyLevel)

El sistema clasifica los objetos detectados según un umbral binario de **3 metros**: los objetos por debajo de este umbral se consideran de riesgo potencial (bounding box roja) y los que están por encima se consideran seguros (bounding box verde). Este umbral coincide con la zona de mayor error de estimación, identificada como el punto crítico de fallo del sistema; se proponen como mejoras futuras zonas de histéresis, umbrales dinámicos y calibración intrínseca formal de la cámara.

## Configuración

El sistema utiliza un archivo de configuración **YAML externalizado**, permitiendo actualizar rutas de modelos, información de la cámara (altura física y en píxeles del sensor, distancia focal) y el objeto de referencia sin modificar el código fuente.

## Desempeño

- Desarrollado y probado en un laptop con procesador Intel Core i7-1065G7 (GPU limitada); el procesamiento en CPU fue una decisión de diseño deliberada, no una limitación del proyecto.
- Rendimiento promedio de ~12 fps en CPU, cumpliendo el objetivo de 10 fps en la mayoría de los casos.
- Error relativo medio menor al 8% en mediciones controladas.

## Trabajo Futuro

1. Validar el método propuesto en escenarios dinámicos adicionales.
2. Comparar resultados contra otro dispositivo de medición de distancia, como LIDAR o cámara estéreo.
3. Integrar detección 3D con bounding boxes 3D para corregir la imprecisión en detecciones no perpendiculares del objeto de referencia.
4. Entrenar modelos con datasets personalizados para detectar vehículos específicos de la región.
5. Explorar modelos MDE y arquitecturas de detección más recientes.

## Uso del código fuente

Dependencias requeridas:
1. Python
2. OpenCV
3. Numpy
4. yaml

Pasos para ejecutar el programa localmente:
1. Clonar este repositorio.
2. Abrir el repositorio en el IDE de su preferencia.
3. Editar el archivo `config.yml` y actualizar las rutas de los modelos.
4. Desde `config.yml`, actualizar la información de la cámara: altura física y en píxeles del sensor, y distancia focal. Cada cámara tiene especificaciones distintas; ingresar información incorrecta afectará la precisión del sistema.
5. Actualizar `real_object_height` con la altura física (en mm) del objeto de referencia a utilizar en su región.

   ```yaml
   target_object:
     target: <nombre_objeto>
     real_object_height: <altura_en_mm>
   ```

6. Verificar en `config.py` que los archivos de YOLO (names, cfg, weights) del objeto de referencia y de los objetos circundantes estén correctamente inicializados.
7. Ejecutar el programa corriendo el archivo `Main.py`.

## Ejecutar el programa con otros modelos YOLO

Para usar modelos YOLO diferentes, agregar las rutas correspondientes en `config.yml`:

```yaml
model_path:
  model_names: ruta/modelo.names
  model_yolo_cfg: ruta/modelo.cfg
  model_yolo_weights: ruta/modelo.weights
```

Y agregar las líneas de inicialización correspondientes en `config.py`, actualizando los nombres de clase de forma consistente con el modelo especificado.


