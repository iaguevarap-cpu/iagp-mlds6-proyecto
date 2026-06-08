# Reporte del Modelo Final

**Proyecto:** Identificación de Firmas Adulteradas con Redes Siamesas
**Autor:** Ivonne Alexandra Guevara Prieto — CC 1032376081
**Fecha:** Junio 2025
**Notebooks de referencia:** `scripts/modeling/3_experimental_set_up.ipynb` · `scripts/modeling/4_modeling.ipynb`

---

## 1. Resumen Ejecutivo

Se desarrolló un sistema de verificación automática de firmas manuscritas basado en una **Red Siamesa con CNN base**. El modelo compara pares de imágenes de firmas y produce un score de similitud que indica si el par es genuino (`< 0.40`) o falsificado (`> 0.60`). El modelo final alcanzó un **AUC-ROC de 0.8184** sobre el conjunto de test, superando ampliamente el clasificador aleatorio (AUC = 0.50). Aunque no se alcanzó el objetivo inicial de AUC ≥ 0.95, el sistema es funcional como herramienta de apoyo a la detección de fraude, especialmente dado su **recall del 95% en la clase falsificada** — la métrica más crítica en el contexto bancario.

| Criterio | Objetivo | Resultado |
|---|---|---|
| AUC-ROC | ≥ 0.95 | **0.8184** |
| Arquitectura | Siamese Network + CNN | Implementado |
| Comparación de modelos | ≥ 2 configuraciones | CNN base vs. EfficientNetB0 |
| Recall clase falsificada | ≥ 85% | **95%** |
| Interpretación FP/FN | Contexto bancario | Analizado |
| Función de predicción | Score + nivel de alerta | `verificar_firma()` |

---

## 2. Descripción del Problema

La falsificación de firmas manuscritas es una de las fuentes de fraude más comunes en operaciones financieras y legales, con pérdidas globales estimadas en US$26 mil millones en 2023. El proceso de verificación manual es lento, subjetivo e inviable a escala institucional.

El proyecto busca transformar este proceso en un sistema automatizado que, dado un par de imágenes de firmas $(A, B)$, determine si son del mismo autor (`0 = genuino`) o si una es una imitación deliberada (`1 = falsificado`). El modelo opera sobre el **dataset CEDAR** (Kaggle, licencia CC0), con 1,649 imágenes de firmas de 55 individuos organizadas en ~1,200 pares de comparación etiquetados.

---

## 3. Descripción del Modelo

### 3.1 Modelo seleccionado: Red Siamesa con CNN base

El **Modelo 1 — Red Siamesa con CNN base** fue seleccionado como modelo final tras comparación experimental con un segundo modelo de Transfer Learning (EfficientNetB0). Se seleccionó por:

- AUC superior (**0.8184** vs. ~0.50 de EfficientNetB0).
- Confirmación empírica de que el Transfer Learning desde ImageNet no es transferible al dominio de trazos monocromáticos de firmas manuscritas.
- Aprendizaje de representaciones propias del dominio de firmas desde cero.

### 3.2 Arquitectura

```
Firma A (105,105,1) ──► CNN Base (pesos compartidos) ──►
                                                          |emb_A − emb_B| ──► Dense(1, sigmoid) ──► score
Firma B (105,105,1) ──► CNN Base (pesos compartidos) ──►
```

**Capas de la CNN Base:**

| Capa | Configuración | Salida |
|---|---|---|
| Data Augmentation | RandomFlip + RandomRotation(0.05) + RandomZoom(0.05) | (105,105,1) |
| Conv2D 1 + MaxPool + Dropout(0.3) | 64 filtros, 10×10, ReLU | (48,48,64) |
| Conv2D 2 + MaxPool + Dropout(0.3) | 128 filtros, 7×7, ReLU | (21,21,128) |
| Conv2D 3 + MaxPool | 128 filtros, 4×4, ReLU | (9,9,128) |
| Flatten → Dense (embedding) | 4096 unidades, Sigmoid | (4096,) |
| Distancia L1 | `\|emb_A − emb_B\|` | (4096,) |
| Dense (clasificador) | 1 unidad, Sigmoid | score ∈ [0,1] |

### 3.3 Hiperparámetros del modelo final

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Learning rate | `1e-4` (Config A) | Mejor val_AUC en comparación con `1e-5` |
| Batch size | 32 | Balance gradiente / memoria GPU T4 Colab |
| Optimizer | Adam | Adaptación del LR por parámetro |
| Loss | binary_crossentropy | Estándar para clasificación binaria |
| EarlyStopping | `patience=5`, `monitor='val_auc'`, `restore_best_weights=True` | Mitiga overfitting |
| ReduceLROnPlateau | `factor=0.5`, `patience=3`, `min_lr=1e-7` | Escapa zonas planas en la pérdida |

### 3.4 Técnicas de extracción de características

El modelo implementa las siguientes técnicas de extracción de características para el dominio de firmas:

1. **Data augmentation integrado:** RandomFlip, RandomRotation(±5°) y RandomZoom(±5%) se aplican solo durante el entrenamiento como capa del modelo.
2. **Embeddings CNN profundos:** Tres bloques convolucionales aprenden representaciones jerárquicas de los trazos de la firma (bordes → patrones locales → estructura global).
3. **Distancia L1 entre embeddings:** Mide la diferencia absoluta dimensión a dimensión entre los vectores de 4096 componentes de las dos firmas. La activación sigmoide del embedding acota la distancia en un rango [0,1], deseable para un score de similitud estable.

---

## 4. Evaluación del Modelo

### 4.1 Métricas sobre el conjunto de test

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Genuino (0) | 79% | 20% | 32% |
| Falsificado (1) | 56% | 95% | 71% |
| **Accuracy general** | | | **58%** |
| **AUC-ROC** | | | **0.8184** |

### 4.2 Interpretación de métricas

**Precisión (79%/56%):** Hay pocas alarmas de firmas genuinas incorrectamente clasificadas (79% de precisión en genuinas). Sin embargo, el resultado de firmas falsificadas incorrectamente clasificadas es alarmante (56%), lo que debe mantenerse presente en el análisis de la matriz de confusión.

**Recall (20%/95%):** El modelo detecta la gran mayoría de firmas falsificadas (95%), pero no es preciso al identificar las firmas genuinas (20%). En el contexto bancario, el recall de la clase falsificada es la métrica más crítica: los **falsos negativos** (fraudes no detectados) tienen un costo operativo mucho mayor que los **falsos positivos** (rechazos de firmas legítimas).

**F1-score:** Indica un desbalance claro entre precision y recall para firmas genuinas (F1=32%). El modelo presenta buen equilibrio para firmas falsificadas (F1=71%).

**AUC-ROC = 0.8184:** El modelo distingue correctamente entre firma genuina y adulterada en el 81.84% de los casos, considerando todos los posibles umbrales de decisión. Supera ampliamente el clasificador aleatorio (0.5). El clasificador no se puede considerar exitoso en términos absolutos, pero puede servir de soporte a un proceso de verificación que no tenga costos altos en caso de falsos positivos o falsos negativos.

**Umbral ajustable:** El umbral óptimo de decisión no necesariamente es 0.5. Para minimizar falsos negativos (fraudes no detectados), se puede bajar el umbral a ~0.35, aumentando el recall de la clase falsificada a expensas de más falsos positivos. Esta calibración es una decisión de negocio, no técnica.

### 4.3 Sistema de niveles de alerta

| Score del modelo | Resultado | Acción recomendada |
|---|---|---|
| < 0.40 | GENUINO | Aprobación automática |
| 0.40 – 0.60 | INCIERTO | Derivar a revisión humana |
| > 0.60 | FALSIFICADO | Rechazo y alerta de fraude |

### 4.4 Análisis de casos fallidos

Los errores se concentran en **falsificaciones hábiles** (*skilled forgeries*): imitaciones deliberadas con trazos muy similares a la firma original. Esta es una limitación conocida también para los peritos humanos. Los scores de los pares mal clasificados tienden a estar cerca del umbral de 0.5, indicando baja confianza del modelo. En producción, los pares con score en la zona de incertidumbre (0.40–0.60) deben derivarse a revisión humana.

**Falsos Negativos (FN):** Casos donde el modelo predijo *genuino* pero la firma era *falsificada*. Aunque el modelo detecta la mayoría de fraudes, los FN ocurren con mayor frecuencia de lo deseable para un sistema bancario de producción.

**Falsos Positivos (FP):** Casos donde el modelo predijo *falsificado* pero la firma era *genuina*. Generan rechazos indebidos de transacciones legítimas, afectando la experiencia del cliente.

---

## 5. Conclusiones y Recomendaciones

### 5.1 Conclusiones

1. La **Red Siamesa con CNN base** demostró ser el modelo más efectivo para la verificación automática de firmas manuscritas, aprendiendo embeddings discriminativos desde cero en el dominio específico de firmas.
2. La comparación con **EfficientNetB0 confirma empíricamente** que el Transfer Learning desde ImageNet no es efectivo para el dominio de firmas: el modelo clasificó prácticamente de manera aleatoria, validando la hipótesis planteada en el diseño experimental. Es más apropiado aprender representaciones propias del dominio de firmas, dado que las características visuales de imágenes naturales difieren estructuralmente de los trazos monocromáticos.
3. Los **falsos negativos** (fraudes no detectados) son el error más crítico en el contexto bancario. La función `verificar_firma()` implementa una zona de incertidumbre (score 0.40–0.60) que deriva estos casos a revisión humana, reduciendo el riesgo operativo.
4. El sistema transformó la verificación de firmas de un proceso manual y subjetivo a uno automatizado, objetivo y escalable. Sin embargo, el AUC de 0.8184 indica que aún existe margen de mejora antes de un despliegue en producción.

### 5.2 Recomendaciones de mejora

| Limitación identificada | Recomendación |
|---|---|
| Overfitting (EarlyStopping óptimo: 2 épocas) | Reducir `patience` a 2–3 en producción |
| Dataset pequeño (~800 pares de entrenamiento) | Ampliar con CEDAR Extended o GPDS-960 |
| Data augmentation leve | Agregar variaciones de perspectiva, ruido y dilatación de trazos |
| Dropout moderado (0.3) | Incrementar a 0.4–0.5 |
| Binary Cross-Entropy como loss | Explorar **Contrastive Loss** o Triplet Loss, diseñadas para redes Siamesas |
| Umbral fijo de 0.5 | Calibrar umbral según política de riesgo del cliente (~0.35 para máximo recall) |

---

## 6. Referencias

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*, 2.
- Hafemann, L. G., Sabourin, R., & Oliveira, L. S. (2017). Learning features for offline handwritten signature verification using deep convolutional neural networks. *Pattern Recognition*, 70, 163-176.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
- Bromley, J., et al. (1993). Signature verification using a siamese time delay neural network. *NIPS*, 6.
