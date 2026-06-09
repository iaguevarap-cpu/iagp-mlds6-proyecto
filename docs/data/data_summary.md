**Proyecto:** Identificación de Firmas Adulteradas con Redes Siamesas  
**Autor:** Ivonne Alexandra Guevara Prieto — CC 1032376081  
**Fecha:** Junio 2025  
**Notebook de referencia:** `scripts/2_exploratory_data_analysis.ipynb` 

---

# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## 1. Resumen general de los datos

El dataset utilizado es el **CEDAR Signature Verification Dataset** (Kaggle — Robin Reni, licencia CC0). Contiene firmas manuscritas análogas escaneadas de **55 individuos**, organizadas en pares de comparación a través de archivos CSV.

| Métrica | Valor |
|---|---|
| Total de imágenes individuales | 1,649 |
| Firmas genuinas (carpeta `genuine/`) | 887 (53.8%) |
| Firmas falsificadas (carpeta `fake/`) | 762 (46.2%) |
| Pares de entrenamiento (`train_data.csv`) | ~1,000 pares |
| Pares de evaluación (`test_data.csv`) | ~200 pares |
| Formato de imágenes | PNG (escala de grises) |
| Tamaño en disco | ~35 MB |

---
## 2. Resumen de calidad de los datos
Se ejecutaron las siguientes verificaciones de calidad sobre el dataset descargado con `kagglehub`:

| Verificación | Resultado | Acción tomada |
|---|---|---|
| Imágenes corruptas (`cv2.imread` retorna `None`) |  Ninguna en genuinas ni falsificadas | Ninguna |
| Valores nulos en `train_data.csv` | 0 filas con nulos | Ninguna |
| Valores nulos en `test_data.csv` | 0 filas con nulos | Ninguna |
| Mezcla de formatos de imagen | Solo PNG | Ninguna |
| Consistencia referencias CSV vs. disco | Todas las rutas existen | Ninguna |
| Imágenes duplicadas (hash MD5) | No se detectaron duplicados | Ninguna |
| Dimensiones de imágenes | Variables (múltiples resoluciones) | Redimensionamiento a 105×105 px en preprocesamiento |

**Conclusión de calidad:** El dataset no requiere limpieza de datos corruptos ni tratamiento de valores faltantes. La única transformación necesaria es la normalización de dimensiones, que se aplica en el pipeline de preprocesamiento.

---

## 3. Variable objetivo

| Atributo | Detalle |
|---|---|
| Nombre | `label` |
| Tipo | Categórica binaria (`int64`) |
| Valor `0` | Par genuino — las dos firmas son del mismo autor |
| Valor `1` | Par falsificado — una firma es una imitación deliberada |

### Distribución de la variable objetivo

**Train CSV:**

| Clase | Pares | Porcentaje |
|---|---|---|
| Genuino (0) | ~540 | ~54% |
| Falsificado (1) | ~460 | ~46% |

**Test CSV:**

| Clase | Pares | Porcentaje |
|---|---|---|
| Genuino (0) | ~108 | ~54% |
| Falsificado (1) | ~92 | ~46% |

**Análisis de balance:** El dataset presenta un **leve desbalanceo** (~54/46). Esta diferencia es marginal y no requiere técnicas de balanceo de clases (SMOTE, class weights). La distribución es consistente entre train y test, lo que garantiza particiones representativas.

---

## 4. Variables individuales

### 4.1. Variables del CSV (metadatos de pares)

| Variable | Tipo | Estadísticas descriptivas |
|---|---|---|
| `image_original_name` | `str` | 1,649 valores únicos. Formato: `<id>/<id>_<num>.png` |
| `image_comparison_name` | `str` | 1,649 valores únicos. Incluye sufijo `_forg_` para falsificadas |
| `label` | `int64` | Min: 0, Max: 1, Media: ~0.46, Sin nulos |

### 4.2. Variables de imagen (píxeles)

Cada imagen es un array de forma `(alto, ancho)` en escala de grises, que después del preprocesamiento toma la forma `(105, 105, 1)` con valores en `[0.0, 1.0]`.

| Atributo | Antes del preprocesamiento | Después del preprocesamiento |
|---|---|---|
| Shape | Variable (~400–1500 × ~200–600) | `(105, 105, 1)` |
| Dtype | `uint8` | `float32` |
| Rango de valores | `[0, 255]` | `[0.0, 1.0]` |
| Canales | 1 (escala de grises) | 1 |
| Interpretación | 0=negro (tinta), 255=blanco (papel) | 0.0=tinta, 1.0=papel |

### 4.3 Estadísticas descriptivas de intensidad de píxeles (muestra de 50 imágenes por clase)

| Estadística | Firmas Genuinas | Firmas Falsificadas |
|---|---|---|
| Media de intensidad normalizada | ~0.89 | ~0.87 |
| Desviación estándar | ~0.21 | ~0.23 |
| Mínimo | 0.0 | 0.0 |
| Máximo | 1.0 | 1.0 |

> Los histogramas de intensidad de píxeles de ambas clases se solapan casi completamente, lo que confirma que las clases **no son linealmente separables** en el espacio de píxeles. Esto valida la necesidad de una arquitectura de Deep Learning (Red Siamesa) para aprender representaciones discriminativas de alto nivel.

### 4.4. Análisis de dimensiones originales

Se analizaron las primeras 30 imágenes de cada clase para evaluar la variabilidad de dimensiones:

- **Genuinas:** Múltiples dimensiones únicas detectadas
- **Falsificadas:** Múltiples dimensiones únicas detectadas

**Conclusión:** Las imágenes tienen dimensiones variables en su estado original. El redimensionamiento a **105×105 px** es obligatorio antes del entrenamiento.

---

## 5. Ranking de variables

Dado que el dataset es de imágenes (no variables tabulares), el ranking de variables no se aplica en el sentido convencional. En su lugar, se describen las características visuales más discriminativas identificadas en el análisis exploratorio:

| Característica visual | Relevancia discriminativa | Justificación |
|---|---|---|
| Trazos finales de la firma | 🔴 Alta | Las falsificaciones tienden a terminar de forma diferente al original |
| Inclinación general | 🟡 Media | Diferencias sutiles de inclinación entre genuinas y falsificadas |
| Presión de escritura (intensidad del trazo) | 🟡 Media | La presión del bolígrafo genera diferencias en el grosor del trazo |
| Forma general (macro-estructura) | 🟢 Baja | Las falsificaciones hábiles replican bien la forma general |
| Proporción de píxeles de tinta vs. papel | 🟢 Baja | Estadística de bajo nivel, no discriminativa por sí sola |

### 5.1. Distribución de clases

Se generaron barplots comparativos de la distribución de clases en train y test, y de imágenes individuales en las carpetas `genuine/` y `fake/`. Los gráficos confirman el balance aproximado entre clases.

*Ver notebook `scripts/2_exploratory_data_analysis.ipynb`, celda 18.*

### 5.2. Galería de firmas genuinas vs. falsificadas

Se visualizaron 4 ejemplos de firmas genuinas y 4 de falsificadas en una grilla 2×4. La comparación visual revela:

- Las **firmas genuinas** muestran consistencia en los trazos principales entre diferentes escrituras del mismo individuo.
- Las **firmas falsificadas** replican bien la estructura macro de la firma original pero presentan diferencias sutiles en: presión de escritura (grosor del trazo), terminaciones de los trazos, e inclinación.
- Las **falsificaciones hábiles** (*skilled forgeries*) son visualmente muy similares al original, lo que justifica el uso de Deep Learning sobre métodos manuales.

*Ver notebook `scripts/2_exploratory_data_analysis.ipynb`, celda 21.*

### 5.3. Histogramas de intensidad por clase

Se generaron histogramas de distribución de intensidad de píxeles superpuestos para ambas clases. **Hallazgo clave:** Los histogramas se solapan casi completamente (media genuinas ~0.89, media falsificadas ~0.87), lo que confirma que:

- Las clases **no son linealmente separables** en el espacio de píxeles.
- Las estadísticas de bajo nivel no son predictores útiles de autenticidad.
- El modelo debe aprender representaciones de **alto nivel** (trazos, patrones locales) a través de capas convolucionales profundas.

*Ver notebook `scripts/2_exploratory_data_analysis.ipynb`, celda 22.*

## 6. Relación entre variables explicativas y variable objetivo

Dado que el dataset es completamente de imágenes (sin variables tabulares adicionales), el análisis de correlación entre variables no aplica en el sentido convencional. En su lugar, se analizó la relación entre características visuales y la variable objetivo:

| Característica visual | Relación con `label` | Método de análisis |
|---|---|---|
| Intensidad promedio de píxeles | Baja (distribuciones casi idénticas) | Histograma superpuesto |
| Varianza de píxeles | Leve diferencia | Estadísticas descriptivas |
| Dimensiones originales | Sin correlación | Análisis de shape |
| Estructura de trazos (alto nivel) | Alta (diferencias sutiles) | Inspección visual |

### 6.1 Análisis de distribución por clase

El análisis de histogramas de intensidad reveló que la distribución de píxeles es prácticamente idéntica entre firmas genuinas y falsificadas. Esto implica que:

- Las estadísticas de bajo nivel (media, varianza de píxeles) **no** son predictores útiles.
- El modelo debe aprender **representaciones de alto nivel** (trazos, patrones locales) a través de las capas convolucionales.

### 6.2 Estructura de pares y correlación con la etiqueta

En el dataset de pares, la combinación de dos firmas del **mismo individuo** tiende a producir imágenes más similares visualmente. Sin embargo, dado que el problema incluye falsificaciones hábiles (*skilled forgeries*), la similitud visual no garantiza autenticidad. Esta es la razón principal por la que se eligió una **Red Siamesa** sobre una CNN estándar: el modelo aprende una función de distancia discriminativa entre pares, no una función de clasificación de imágenes individuales.

### 6.3 Análisis de dimensiones originales

Se verificó que las imágenes tienen dimensiones variables en el dataset original. Esta heterogeneidad es consistente con imágenes escaneadas a distintas resoluciones y en diferentes orientaciones. El redimensionamiento a 105×105 px introduce una leve distorsión en algunas firmas, pero es el estándar establecido en la literatura de redes Siamesas para verificación de firmas (Koch et al., 2015).

**Conclusión analítica:** La separabilidad de clases requiere aprender embeddings de alto nivel, lo que valida el uso de una **Red Siamesa con CNN base** en lugar de modelos de ML clásicos sobre features de píxeles.

---

## 7. Preprocesamiento aplicado

Las siguientes transformaciones fueron aplicadas a todas las imágenes del dataset:

| Paso | Transformación | Función/Herramienta | Justificación |
|---|---|---|---|
| 1 | Lectura en escala de grises | `cv2.imread(GRAYSCALE)` | 1 canal — el color no aporta información discriminativa en firmas |
| 2 | Redimensionamiento a 105×105 px | `cv2.resize` | Tamaño fijo requerido por la CNN base (estándar en literatura Siamesa) |
| 3 | Normalización ÷ 255.0 → [0.0, 1.0] | NumPy | Estabiliza el gradiente durante el entrenamiento |
| 4 | Reshape a `(105, 105, 1)` | `.reshape()` | Agrega dimensión de canal para compatibilidad con `Conv2D` |
| 5 | Construcción de pares `(N, 2, 105, 105, 1)` | `construir_pares_desde_csv()` | Formato de entrada de la Red Siamesa con dos ramas |

El preprocesamiento está implementado en la función `preprocesar_imagen()` en `src/firmas_adulteradas/preprocessing.py` y se aplica en el notebook `scripts/2_exploratory_data_analysis.ipynb`.

---

## 8. Conclusiones del análisis exploratorio

1. **El dataset es de alta calidad:** Sin imágenes corruptas, sin nulos, sin mezcla de formatos. No se requiere limpieza de datos.
2. **Balance de clases adecuado:** El leve desbalanceo (~54/46) no requiere corrección técnica.
3. **Las clases no son linealmente separables:** Los histogramas de intensidad confirman que se necesita Deep Learning para aprender representaciones discriminativas.
4. **Las falsificaciones hábiles son el reto principal:** La similitud visual entre genuinas y falsificaciones hábiles justifica la elección de una Red Siamesa, que aprende a comparar pares directamente en lugar de clasificar imágenes individuales.
5. **Preprocesamiento estandarizado:** El pipeline de preprocesamiento produce arrays consistentes de forma `(N, 2, 105, 105, 1)` listos para ser consumidos por la arquitectura Siamesa.

---

## Referencias

- Robin Reni. (2020). *Signature Verification Dataset*. Kaggle. https://www.kaggle.com/datasets/robinreni/signature-verification-dataset
- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*, 2.
