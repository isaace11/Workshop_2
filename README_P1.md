# Problema 1 — Detección de Fatiga Muscular con EMG

##  Descripción del problema

En este problema se busca clasificar si un sujeto se encuentra en estado de **fatiga muscular** o **no fatiga**, a partir de señales de electromiografía (EMG) registradas en 8 músculos de la pierna.

La EMG mide la actividad eléctrica del músculo, y su comportamiento cambia a medida que aparece la fatiga.

---

##  Pipeline del proyecto

El flujo completo implementado fue:

## Cómo ejecutar el proyecto:

Nota: El archivo "emg_data.csv" no se incluye en el repositorio porque es grande, y estuvo generando problemas para subirlo a github. 
1. Se puede generar ejecutanto "get_data.ipynb" para descargar el dataset y generar "emg_data"
2. Ejecutar "P1_01_preparacion_datos.ipynb" para:
   - segmentar la señal en ventanas de 1 segundo
   - extraer features
   - generar "data_procesada/train.csv", "val.csv" y "test.csv"

3. Ejecutar "P1_02_entrenamiento.ipynb" para:
   - entrenar los modelos
   - evaluar el desempeño

## Notebook 1 — Preparación de datos

En este notebook se realizaron los siguientes pasos:

- Carga del dataset EMG
- Conversión del target a clasificación binaria:
  - 0 = no fatiga
  - 1 = fatiga
- Exploración visual de las señales
- Segmentación en ventanas de 1 segundo con 50% de solapamiento
- Extracción de features por canal:

### Features utilizadas

| Feature | Significado |
|--------|------------|
| MAV | Amplitud media |
| RMS | Energía de la señal |
| STD | Variabilidad |
| WL | Complejidad de la señal |
| ZC | Frecuencia de cambios de signo |

---

## Justificación de features

Las features seleccionadas son ampliamente utilizadas en el análisis de EMG:

- **MAV y RMS** capturan la amplitud y energía de la señal, que suelen aumentar con la fatiga.
- **STD** mide la variabilidad.
- **WL** refleja la complejidad de la señal.
- **ZC** está relacionado con cambios en la frecuencia.

Estas permiten transformar señales crudas en variables útiles para clasificación.


---



---

## Observación importante

Se observó una diferencia significativa entre los resultados de validación y test, especialmente en Gradient Boosting.

Esto puede deberse a:

- variaciones temporales en la señal
- diferencias en la distribución de los datos entre splits
- efectos de la segmentación

Esto sugiere que el modelo podría beneficiarse de una validación más robusta.

---

## Conclusión

El problema demuestra que es posible detectar fatiga muscular a partir de señales EMG mediante técnicas de machine learning.

El uso de:

- segmentación temporal
- extracción de features
- modelos supervisados

permite transformar señales complejas en un problema de clasificación estructurado.