# Definición de los datos

## 1. Origen de los datos

| Campo | Detalle |
|---|---|
| **Nombre** | Signature Verification Dataset (CEDAR) |
| **Fuente** | Kaggle — Robin Reni |
| **URL** | https://www.kaggle.com/datasets/robinreni/signature-verification-dataset |
| **Descarga** | `kagglehub.dataset_download("robinreni/signature-verification-dataset")` |
| **Licencia** | CC0: Public Domain |
| **Versión usada** | 1 (última disponible al momento del proyecto) |
| **Origen original** | CEDAR (Center of Excellence for Document Analysis and Recognition) — University at Buffalo |
| **Publicación original** | Kumar, R., et al. (2010). Signature Verification using a Siamese Time Delay Neural Network. |

## 2. Especificación de los scripts para la carga de datos

- scripts\business_data_load.ipynb

## 3. Referencias a rutas o bases de datos origen y destino

- [ ] Especificar las rutas o bases de datos de origen y destino para los datos.

## 4. Rutas de origen de datos

- [ ] Especificar la ubicación de los archivos de origen de los datos.

### 4.2 Descripción General del Dataset

El dataset CEDAR contiene **firmas manuscritas análogas escaneadas** de 55 individuos diferentes. Para cada individuo se dispone de firmas genuinas (escritas por el propio individuo) y firmas falsificadas (*skilled forgeries*, imitadas deliberadamente por otro individuo).

| Estadística | Valor |
|---|---|
| Total de imágenes individuales | 1,649 |
| Firmas genuinas | 887 |
| Firmas falsificadas | 762 |
| Individuos representados | 55 |
| Formato de imágenes | PNG (escala de grises) |
| Resolución original | Variable (aprox. 952 × 358 px promedio) |
| Resolución después del preprocesamiento | 105 × 105 px |
| Tamaño en disco | ~35 MB |
| Balance de clases (imágenes individuales) | 53.8% genuinas / 46.2% falsificadas |

---

###  4.3. Estructura del Dataset

El dataset tiene dos niveles de organización:

### 4.3.1 Carpetas de imágenes individuales

```
sign_data/
├── train/
│   ├── <id_persona>/
│   │   ├── <id_persona>_<num>.png        ← firmas genuinas
│   │   └── <id_persona>_forg_<num>.png   ← firmas falsificadas
│   └── ...
└── test/
    └── (misma estructura)
```

### 4.3.2 Archivos CSV de pares (estructura principal del modelo)

El modelo de verificación opera sobre **pares de imágenes**, no sobre imágenes individuales. Los pares están definidos en dos archivos CSV:

| Archivo | Pares totales | Descripción |
|---|---|---|
| `sign_data/train_data.csv` | ~1,000 | Pares de entrenamiento |
| `sign_data/test_data.csv` | ~200 | Pares de evaluación final |

---
### 4.3.3 Calidad del Dataset

| Aspecto | Resultado | Detalle |
|---|---|---|
| Imágenes corruptas |  Ninguna | Todas las imágenes cargaron correctamente con `cv2.imread` |
| Valores nulos en CSV | Ninguno | Los tres campos están completos en todas las filas |
| Mezcla de formatos | No | Todas las imágenes son PNG |
| Referencias CSV vs. disco | Consistente | Todas las rutas del CSV existen en las carpetas correspondientes |
| Imágenes duplicadas | Sin detectar | No se realizó análisis de hash por el tamaño del dataset |
| Sesgo de individuo | Posible | Las firmas falsificadas fueron realizadas por imitadores que vieron la firma original, lo cual puede introducir sesgo visual |

---

### 4.3.4 Estructura del Array de Pares (Input del Modelo)

Después del preprocesamiento, los datos se transforman en arreglos de pares con la siguiente estructura:

| Variable | Shape | Dtype | Descripción |
|---|---|---|---|
| `pares_train` | `(N_train, 2, 105, 105, 1)` | `float32` | Pares de entrenamiento. Dim 1: índice 0 = imagen referencia, índice 1 = imagen comparación |
| `labels_train` | `(N_train,)` | `int64` | Etiquetas de entrenamiento: 0 = genuino, 1 = falsificado |
| `pares_test` | `(N_test, 2, 105, 105, 1)` | `float32` | Pares de evaluación final |
| `labels_test` | `(N_test,)` | `int64` | Etiquetas de evaluación final |

---

### 4.3.5 Transformaciones Aplicadas

| Paso | Transformación | Justificación |
|---|---|---|
| 1 | Lectura en escala de grises (`cv2.IMREAD_GRAYSCALE`) | La Red Siamesa original opera sobre 1 canal; el color no aporta información discriminativa en firmas |
| 2 | Redimensionamiento a 105 × 105 px | Tamaño fijo requerido por la arquitectura CNN base; valor estándar en la literatura de Siamese Networks |
| 3 | Normalización ÷ 255.0 → [0, 1] | Estabiliza el gradiente durante el entrenamiento |
| 4 | Reshape a `(105, 105, 1)` | Agrega dimensión de canal para compatibilidad con `Conv2D` de Keras |
| 5 | Construcción de pares `(N, 2, 105, 105, 1)` | Formato requerido por la arquitectura Siamesa con dos entradas |

---

### 4.3.6 Referencias del Dataset

- Robin Reni. (2020). *Signature Verification Dataset*. Kaggle. https://www.kaggle.com/datasets/robinreni/signature-verification-dataset
- CEDAR Signature Database. University at Buffalo. https://cedar.buffalo.edu/NIJ/data/
- Kumar, R., Kundu, M., & Sharma, A. (2010). Signature verification using a Siamese Time Delay Neural Network. *International Journal of Pattern Recognition and Artificial Intelligence*.

### Base de datos de destino

- [ ] Especificar la base de datos de destino para los datos.
- [ ] Especificar la estructura de la base de datos de destino.
- [ ] Describir los procedimientos de carga y transformación de los datos en la base de datos de destino.
