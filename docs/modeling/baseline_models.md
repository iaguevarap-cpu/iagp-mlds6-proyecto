# Reporte del Modelo Baseline

**Proyecto:** Identificación de Firmas Adulteradas con Redes Siamesas
**Autor:** Ivonne Alexandra Guevara Prieto — CC 1032376081
**Fecha:** Junio 2025
**Notebooks de referencia:** `scripts/3_experimental_set_up.ipynb` · `scripts/4_modeling.ipynb`

Este documento contiene los resultados del modelo baseline.

## 1. Descripción del modelo

l problema es de **clasificación binaria sobre pares de imágenes**: dado un par de firmas $(A, B)$, el modelo debe predecir si son del mismo autor (`0 = genuino`) o si una es una imitación deliberada (`1 = falsificado`). Ya que el problema es de análisis de imagen, se trabajó con **redes neuronales convolucionales siamesas**.

Se exploraron dos modelos baseline con la misma estructura Siamesa pero diferente red extractora de características:

### 1.1 Modelo 1 — Red Siamesa con CNN base

Una Red Siamesa consiste en dos ramas con **pesos compartidos** que procesan simultáneamente las dos firmas del par. Cada rama aplica la misma CNN para extraer un vector de características (*embedding*) de cada firma. Luego, la distancia L1 entre los dos embeddings alimenta una capa densa final con activación sigmoide que produce el score de similitud.

**1.1.1 Justificación:**
- El problema es de **comparación por pares**, no de clasificación individual — las Siamese Networks están diseñadas específicamente para esto.
- Las redes Siamesas son eficientes con **pocos ejemplos por clase** gracias al aprendizaje de métricas de similitud.
- El uso de **pesos compartidos** garantiza que ambas firmas sean procesadas con la misma función de embedding, haciendo la comparación simétrica y consistente.
- La **distancia L1** entre embeddings captura diferencias sutiles en trazos, presión e inclinación — las señales discriminativas en firmas manuscritas.

**1.1.2 Arquitectura de la CNN base:**

| Capa | Configuración | Activación |
|---|---|---|
| Data Augmentation | RandomFlip + RandomRotation(0.05) + RandomZoom(0.05) | — |
| Conv2D 1 | 64 filtros, kernel 10×10 | ReLU |
| MaxPooling2D + Dropout | pool (2,2), rate=0.3 | — |
| Conv2D 2 | 128 filtros, kernel 7×7 | ReLU |
| MaxPooling2D + Dropout | pool (2,2), rate=0.3 | — |
| Conv2D 3 | 128 filtros, kernel 4×4 | ReLU |
| MaxPooling2D | pool (2,2) | — |
| Flatten → Dense | 4096 unidades | Sigmoid |

> Las transformaciones de augmentation fueron elegidas cuidadosamente para el dominio de firmas: se usa rotación leve (±5%) y zoom (±5%) porque las firmas pueden tener pequeñas variaciones de inclinación y escala entre escrituras del mismo autor. Se evitan rotaciones grandes o flip vertical porque distorsionarían la estructura natural de la firma.

> La capa de embedding de 4096 dimensiones con activación sigmoide comprime toda la información visual en un vector denso. La activación sigmoide acota cada dimensión en [0,1], lo que hace que la distancia L1 resultante sea siempre no negativa y acotada — condición deseable para un score de similitud.

### 1.2 Modelo 2 — Red Siamesa con EfficientNetB0 (Transfer Learning)

Como modelo de comparación, se reemplazó la CNN base por un extractor **EfficientNetB0 preentrenado en ImageNet** con capas congeladas. Las imágenes de firmas (1 canal) se convierten a 3 canales mediante un adaptador `Conv2D(3, 1×1)` para compatibilidad con EfficientNetB0.

Esta arquitectura mantiene la estructura del primer modelo de red siamesa con base CNN, pero al incluir una sección de preentrenamiento, es más eficiente aunque restringida a los resultados del preentrenamiento.

**1.2.1 Partición del dataset:**

| Conjunto | Origen | Proporción | Uso |
|---|---|---|---|
| Train | `train_data.csv` (80%) | ~64% del total | Ajuste de pesos del modelo |
| Validación | `train_data.csv` (20%) | ~16% del total | Selección de hiperparámetros y EarlyStopping |
| Test | `test_data.csv` (100%) | ~20% del total | Evaluación final con datos no vistos |

La partición es **estratificada** (`stratify=labels`) para preservar la distribución original de clases en cada split. El balance es consistente entre train, validación y test, garantizando que las métricas de evaluación sean representativas del comportamiento real del modelo.

---


## 2. Variables de entrada
Los modelos reciben como entrada **pares de imágenes** preprocesadas:

| Variable | Shape | Dtype | Descripción |
|---|---|---|---|
| `firma_referencia` | `(None, 105, 105, 1)` | `float32` | Firma de referencia normalizada a [0,1] |
| `firma_sospechosa` | `(None, 105, 105, 1)` | `float32` | Firma a comparar, normalizada a [0,1] |

No se utilizan variables tabulares adicionales. Toda la información discriminativa está contenida en los píxeles de las dos imágenes del par.

---

## 3. Variable objetivo

| Atributo | Detalle |
|---|---|
| Nombre | `label` |
| Tipo | Categórica binaria (`int64`) |
| Valor `0` | Par genuino — las dos firmas son del mismo autor |
| Valor `1` | Par falsificado — una firma es una imitación deliberada |

---

## 4. Evaluación del modelo

### 4.1 Métricas de evaluación

Se utilizaron las siguientes métricas de clasificación, evaluadas sobre el **conjunto de test** (datos no vistos durante el entrenamiento):

- **AUC-ROC:** Métrica principal. Mide la capacidad discriminativa del modelo a todos los umbrales posibles. Rango [0,1], donde 1.0 es perfecto y 0.5 es aleatorio.
- **Accuracy:** Proporción de predicciones correctas.
- **Precision:** De los casos predichos como clase X, cuántos son realmente clase X.
- **Recall:** De todos los casos reales de clase X, cuántos detectó el modelo.
- **F1-score:** Media armónica entre precision y recall.

Se compararon dos configuraciones de *learning rate* (grid search manual) para seleccionar la mejor:

| Configuración | Learning rate | Descripción |
|---|---|---|
| **Config A** | `1e-4` | Valor del artículo original de Siamese Networks |
| **Config B** | `1e-5` | Learning rate más conservador para evitar sobreajuste |

Hiperparámetros fijos en ambas configuraciones:

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| `batch_size` | 32 | Balance gradiente/memoria GPU T4 Colab |
| `epochs` | 30 | Con EarlyStopping activo |
| `optimizer` | Adam | Convergencia rápida y adaptación del LR |
| `loss` | binary_crossentropy | Estándar para clasificación binaria |
| EarlyStopping | `patience=5`, `monitor='val_auc'` | Restaura mejores pesos |
| ReduceLROnPlateau | `factor=0.5`, `patience=3` | Reduce LR en zonas planas |

### 4.2 Resultados de evaluación

**Comparación de configuraciones — Modelo 1 (CNN base):**

| Configuración | LR | val_AUC | val_Acc |
|---|---|---|---|
| Config A | `1e-4` | Mejor entre ambas | — |
| Config B | `1e-5` | — | — |
| **→ Mejor seleccionada** | `1e-4` (Config A) | | |

> El learning rate fue reducido a `1e-4` tras la comparación de configuraciones, confirmando que el optimizador encontró regiones planas en la función de pérdida y necesitó pasos más pequeños para escapar.

**Comparación final sobre conjunto de test:**

| Modelo | AUC-ROC | Estrategia |
|---|---|---|
| Modelo 1 — Red Siamesa CNN base (Config A) | **0.8184** | Entrenamiento desde cero |
| Modelo 2 — Red Siamesa EfficientNetB0 | ~0.50 | Transfer Learning (ImageNet) |
| Clasificador aleatorio (referencia) | 0.50 | — |

**Classification report — Modelo 1 (ganador):**

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Genuino (0) | 79% | 20% | 32% |
| Falsificado (1) | 56% | 95% | 71% |
| **Accuracy general** | | | **58%** |

---

## 5. Análisis de los resultados

**Precisión:** Valores de 79% para firmas genuinas y 56% para firmas falsificadas muestran que hay pocas alarmas de firmas genuinas incorrectamente clasificadas; sin embargo, el resultado de firmas falsificadas incorrectamente clasificadas es alarmante y debe mantenerse presente en la evaluación de la matriz de confusión.

**Recall:** Valores de 20% para firmas genuinas y 95% para firmas falsificadas indica que el modelo es capaz de encontrar la mayoría de firmas falsificadas, pero que no es preciso al encontrar las firmas genuinas.

**F1-score:** Esta media armónica indica un desbalance entre precision y recall para las firmas genuinas. Por el contrario, el modelo presenta buen equilibrio entre esas métricas para el caso de firmas falsificadas.

**AUC = 0.8184:** El modelo distingue correctamente entre firma genuina y adulterada en el 81.84% de los casos, considerando todos los posibles umbrales de decisión. Supera ampliamente el clasificador aleatorio (0.5) pero no alcanzó el umbral objetivo de 0.95. El clasificador puede servir de soporte a un proceso de clasificación que no tenga costos altos en caso de falsos positivos o falsos negativos.

**Curvas de entrenamiento — Modelo 1:** Las curvas de train y validación no convergen de manera consistente, indicando que el modelo se dificulta aprender patrones generalizables y parece memorizar los datos de entrenamiento. La curva de train continúa mejorando en accuracy mientras que la de validación se estanca — sobreajuste claro. Las gráficas indican que el EarlyStopping óptimo es de tan solo **2 épocas**.

**Curvas de entrenamiento — Modelo 2:** Los valores de accuracy, loss y AUC indican clasificación prácticamente aleatoria. Las características aprendidas por EfficientNetB0 en ImageNet (texturas naturales, bordes de objetos) son estructuralmente distintas de los trazos monocromáticos de firmas manuscritas. Al tener el extractor congelado, la red solo ajusta el adaptador de canal y las capas densas superiores, lo que limita su capacidad de adaptación al dominio.

**Casos fallidos:** Los errores se concentran en falsificaciones hábiles (*skilled forgeries*) con trazos muy similares al original. Los scores de los pares mal clasificados tienden a estar cerca del umbral de 0.5, indicando baja confianza del modelo — consistente con que son los casos más difíciles. En producción, los pares con score en la zona 0.40–0.60 deben derivarse a revisión humana.

---

## 6. Conclusiones

1. La **Red Siamesa con CNN base** es el modelo ganador con AUC = 0.8184, superando ampliamente el clasificador aleatorio y el modelo de Transfer Learning.
2. La arquitectura Siamesa es apropiada para el problema de comparación de pares, pero el tamaño limitado del dataset CEDAR (~800 pares de entrenamiento) limita la capacidad de generalización.
3. El **overfitting detectado** (EarlyStopping óptimo: 2 épocas) sugiere que el modelo aprende rápidamente pero no generaliza bien con más épocas. Se recomienda aumentar el dataset, incrementar el Dropout a 0.4–0.5 y explorar Contrastive Loss como función de pérdida alternativa.
4. La hipótesis experimental fue **confirmada**: el Transfer Learning desde ImageNet no es viable para el dominio de firmas manuscritas monocromáticas.
5. El sistema de tres niveles de alerta (`< 0.40` → genuina, `0.40–0.60` → incierto, `> 0.60` → falsificada) permite calibrar el trade-off FP/FN según la política de riesgo del negocio.

---

## 7. Referencias

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*, 2.
- Hafemann, L. G., Sabourin, R., & Oliveira, L. S. (2017). Learning features for offline handwritten signature verification using deep convolutional neural networks. *Pattern Recognition*, 70, 163-176.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
