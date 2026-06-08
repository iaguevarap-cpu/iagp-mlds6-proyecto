# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

**Proyecto:** Identificación de Firmas Adulteradas con Redes Siamesas  
**Programa:** Machine Learning and Data Science (MLDS) — Módulo 6  
**Institución:** Universidad Nacional de Colombia — Facultad de Ingeniería  
**Autor:** Ivonne Alexandra Guevara Prieto — CC 1032376081  
**Fecha:** Junio 2025  
**Metodología:** TDSP (Team Data Science Process) + CRISP-DM  

---

## 1. Contexto del Negocio

### 1.1 Dominio de aplicación

El proyecto se ubica en el dominio de la **seguridad financiera y documental**. La falsificación de firmas es una de las fuentes de fraude más comunes en operaciones financieras y legales. En 2023, las pérdidas globales relacionadas con este tipo de fraude rondaron los **US$26 mil millones** [1]. El incremento de falsificaciones digitales se estima en un 244% para 2025 [2], y el 94% de las entidades bancarias han sido afectadas con costos promedio de US$0.5 millones por incidente.


### 1.2 Problema de negocio

Los problemas centrales que este proyecto busca resolver son:

- **Fraude documental** — La falsificación genera pérdidas millonarias anuales en los sectores financiero y legal.
- **Limitaciones del proceso manual** — La verificación humana es lenta, costosa, subjetiva y propensa a errores bajo alto volumen.
- **Falta de escalabilidad** — Las instituciones procesan miles de documentos diariamente, lo que hace inviable la revisión pericial en tiempo real.
- **Ausencia de trazabilidad** — Se requiere un sistema que genere registros auditables de cada decisión tomada.
- **Desequilibrio FP/FN** — Un sistema bien entrenado puede equilibrar mejor la sensibilidad y especificidad que un revisor humano bajo presión.

---

## 2. Objetivo del Proyecto

### 2.1 Objetivo general

Desarrollar e implementar un sistema de verificación automática de firmas manuscritas basado en una **Red Siamesa con arquitectura CNN**, capaz de determinar si un par de firmas es genuino o falsificado, con el fin de apoyar la detección de fraude documental en instituciones financieras y legales.

### 2.2 Objetivo en términos de Machine Learning

Entrenar una Red Siamesa que, dado un par de imágenes de firmas $(A, B)$, produzca un score de similitud $s \in [0, 1]$:

$$s = \sigma(W \cdot |f(A) - f(B)|)$$

donde $f(\cdot)$ es la función de embedding aprendida por la CNN y $\sigma$ es la función sigmoide.

| Parámetro | Valor objetivo |
|---|---|
| Tipo de problema | Clasificación binaria por pares |
| Arquitectura principal | Siamese Network + CNN base |
| Modelo de comparación | Siamese Network + EfficientNetB0 |
| Función de pérdida | Binary Cross-Entropy |
| Métrica principal | AUC-ROC ≥ 0.95 |
| Métrica secundaria | Accuracy, F1-score, EER ≤ 5% |
| Latencia máxima | < 2 segundos por par |

### 2.3 Criterios de éxito del proyecto

| Criterio | Umbral mínimo | Umbral ideal |
|---|---|---|
| AUC-ROC en test | ≥ 0.90 | ≥ 0.95 |
| Equal Error Rate (EER) | ≤ 10% | ≤ 5% |
| Recall clase falsificada | ≥ 85% | ≥ 95% |
| Tiempo de inferencia | < 5 s | < 2 s |
| Repositorio GitHub completo | Requerido | Con módulos Python |

---

## 3. Alcance

### 3.1 Incluido en el alcance

- Verificación de firmas manuscritas análogas digitalizadas.
- Detección de falsificaciones hábiles (*skilled forgery*) y aleatorias (*random forgery*).
- Generación de un score de confianza con tres niveles de alerta por par verificado.
- Comparación experimental: Red Siamesa CNN base vs. EfficientNetB0 con Transfer Learning.
- Despliegue de una aplicación web funcional (Streamlit) con interfaz de predicción.
- Documentación completa del proyecto en formato TDSP en repositorio GitHub.

### 3.2 Excluido del alcance

- Verificación de firmas digitales o electrónicas certificadas.
- Reconocimiento de escritura manuscrita general (OCR).
- Verificación de sellos, timbres o rúbricas institucionales.
- Soporte para grafologías no latinas (árabe, mandarín, etc.).
- Decisiones legales autónomas — el sistema es de **apoyo**, no reemplaza al perito.
- Verificación en tiempo real con integración a sistemas bancarios de producción.

### 3.3 Uso del producto por parte del cliente

El sistema clasifica cada verificación en tres niveles de acción:

| Score del modelo | Resultado | Acción recomendada |
|---|---|---|
| < 0.40 | Genuina | Aprobación automática |
| 0.40 – 0.60 | Incierto | Derivar a revisión humana |
| > 0.60 | Falsificada | Rechazo y alerta de fraude |

---

## ## 4. Plan de Trabajo — Metodología CRISP-DM

| Fase CRISP-DM | Actividad principal | Entregable | Notebook |
|---|---|---|---|
| **1. Entendimiento del negocio** | Marco de proyecto, stakeholders, objetivos, plan | `project_charter.md` + `data_definition.md` | `scripts/1_business_data_load.ipynb` |
| **2. Entendimiento de los datos** | Carga del dataset CEDAR, verificación de integridad | Dataset cargado y verificado | `scripts/1_business_data_load.ipynb` |
| **3. Preparación de los datos** | EDA, análisis descriptivo, visualizaciones, preprocesamiento | `data_report.md` + datos limpios | `scripts/2_exploratory_data_analysis.ipynb` |
| **4. Modelado** | Diseño Red Siamesa, partición, data augmentation, entrenamiento | `model_report.md` + modelo `.keras` | `scripts/3_experimental_set_up.ipynb` + `scripts/4_modeling.ipynb` |
| **5. Evaluación** | Métricas, comparación modelos, interpretación resultados | Sección evaluación en `model_report.md` | `scripts/4_modeling.ipynb` |
| **6. Despliegue** | App Streamlit, documentación infraestructura | `deployment.md` + app desplegada | `scripts/5_deployment.ipynb` |

---


## 5. Cronograma

| Tarea | Fecha esperada de finalización | Descripción/Detalles |
| ----------- | -------------------- |----------- |
|Entendimiento del negocio y carga de datos |	27/03/2026	|Definición de los objetivos del negocio y del proyecto. Planeación. Carga y obtención de datos.  transformación de los datos y partición del conjunto de datos.|
|Entendimiento y preparación de los datos|	09/04/2026|	Procesamiento y análisis preliminar de los datos mediante un EDA. Visualización de los datos.Preprocesamiento.|
|Diseño e implementación	|16/04/2026|Diseño y selección del modelo de Deep Learning.  Implementación del modelo.|
|Entrenamiento y validación	|23/04/2026	|Entrenamiento del modelo. Diseño de experimentación, construcción y entrenamiento de modelos, y definición de los criterios o métricas de evaluación.|
|Entrega de reporte final	|27/04/2026	|Evaluación del modelo con métricas de desempeño. Interpretación de los resultados. Finalización y entrega del reporte final.|


---

## 6. Riesgos del Proyecto

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| AUC-ROC por debajo del objetivo (≥ 0.95) | Alta | Alto | Ajustar EarlyStopping, explorar Contrastive Loss, aumentar dataset |
| Overfitting en la Red Siamesa | Alta | Medio | Dropout(0.3) en CNN + data augmentation + EarlyStopping patience=2 |
| EfficientNet no convergencia (TL dominio diferente) | Alta | Bajo | Hipótesis esperada; sirve como comparador negativo para el paper |
| Recursos Colab T4 insuficientes | Baja | Alto | Reducir batch_size a 16, acotar epochs a 20, usar `mixed_precision` |
| Dataset CEDAR pequeño (686 pares por clase) | Media | Medio | Data augmentation (flip, rotación, zoom) integrado en arquitectura |

---

## 7. Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Deep Learning | TensorFlow 2.x / Keras 3 |
| Datos | kagglehub, pandas, NumPy, OpenCV |
| Visualización | Matplotlib, Seaborn |
| ML utilities | scikit-learn (métricas, partición) |
| Despliegue | Streamlit |
| Versionamiento | Git + GitHub |
| Gestión del proyecto | TDSP template (mindlab-unal) |
| Entorno de ejecución | Google Colab (GPU T4) |

---

## 8. Equipo del Proyecto

- Ivonne Guevara: lider y cientifica de datos
- TBD: Ingeniero de ML

## 9. Presupuesto del Proyecto

El presupuesto se estima considerando los recursos humanos requeridos para cada
fase CRISP-DM, los costos de infraestructura computacional para el entrenamiento
del modelo de Red Siamesa, y los gastos misceláneos asociados al proyecto.

| Categoría de Gasto | Descripción | Costo Estimado (USD) | Notas / Consideraciones |
| :----------------- | :---------- | :------------------- | :---------------------- |
| **I. Personal** | | | |
| Científico de Datos | Diseño de la arquitectura Siamesa, preprocesamiento de imágenes, optimización de hiperparámetros, evaluación del modelo. | $6,000 | 2 meses a tiempo parcial |
| Ingeniero de ML | Implementación del pipeline de datos, experimentos de entrenamiento, EDA, desarrollo de la app Streamlit. | $4,000 | 2 meses a tiempo parcial |
| **II. Infraestructura y Software** | | | |
| Plataforma Cloud (GPU) | Google Colab Pro / AWS para entrenamiento de la Red Siamesa (instancias con GPU T4/A100). | $150 | ~50 horas de GPU; el entrenamiento con el dataset CEDAR es ligero en comparación con proyectos de visión a gran escala |
| Almacenamiento Cloud | Google Drive / S3 para dataset CEDAR (~35 MB), checkpoints del modelo (`.keras`) y artefactos del proyecto. | $0 | El tamaño reducido del dataset cabe en los planes gratuitos de Google Drive y Kaggle |
| Licencias de Software | Herramientas de desarrollo e IDEs. | $0 | Se utilizarán exclusivamente herramientas open-source: Python, TensorFlow/Keras, OpenCV, scikit-learn, Streamlit, Git |
| Despliegue de la aplicación | Hosting de la app Streamlit para demostración del modelo. | $0 | Streamlit Cloud ofrece despliegue gratuito para proyectos públicos |
| **III. Misceláneos** | | | |
| Investigación y Desarrollo | Tiempo para revisión de literatura sobre redes Siamesas, falsificación de firmas y técnicas de verificación biométrica. | $500 | Horas dedicadas a exploración de arquitecturas alternativas y resolución de problemas técnicos |
| Gestión de Proyecto | Coordinación del proyecto, documentación TDSP, reuniones de revisión de resultados. | $300 | Incluye redacción de los documentos `project_charter.md`, `data_report.md` y `model_report.md` |
| Contingencias | Fondo para imprevistos (aprox. 10% del total estimado). | $1,095 | Buffer para iteraciones adicionales de entrenamiento, ajuste de arquitectura o problemas de infraestructura no previstos |
| **TOTAL ESTIMADO DEL PROYECTO** | | **$12,045** | |

> **Notas generales:**
> - Los costos de personal corresponden a tarifas de referencia para perfiles
>   de ciencia de datos en el mercado latinoamericano (2025).
> - El costo de infraestructura es significativamente menor que el promedio de
>   proyectos de visión por computadora debido al tamaño reducido del dataset
>   CEDAR (~35 MB, 1,649 imágenes) y a la disponibilidad de herramientas
>   gratuitas para el despliegue.
> - Este presupuesto corresponde a un proyecto piloto/académico. Un despliegue
>   en producción bancaria requeriría una estimación separada que incluya
>   infraestructura cloud empresarial (AWS/GCP/Azure), costos de integración
>   con sistemas internos, y mantenimiento continuo del modelo.

## 10. Stakeholders (partes interesadas)

| Stakeholder | Rol | Interés en el proyecto |
|---|---|---|
| Entidades bancarias y aseguradoras | Cliente principal | Verificar cheques, contratos y pólizas a escala |
| Notarías y registros públicos | Beneficiario directo | Validar autenticidad de firmas en documentos legales |
| Tribunales y juzgados | Beneficiario indirecto | Usar el score como evidencia de apoyo pericial |
| Despachos de abogados | Beneficiario indirecto | Verificar contratos y escrituras antes de firmar |
| Entidades gubernamentales | Beneficiario directo | Proteger documentos oficiales (pasaportes, licencias) |
| Equipo de ciencia de datos | Ejecutor | Diseñar, entrenar y mantener el modelo |

## 11. Referencias

[1] Nasdaq Verafin. (2024). *Global Financial Crime Report*. https://verafin.com/nasdaq-verafin-global-financial-crime-report/

[2] PactVera. (2025). *2026 E-Signature Fraud Statistics*. https://www.pactvera.com/2026-e-signature-fraud-statistics/

[3] Elsayed AS, Mwaheb MA, Elsary AY, El Rashed K, Saleh AR. Awareness and perception of physicians about forgery and counterfeiting in the medical field in Egypt. *Sci Rep.* 2025 Apr 19;15(1):13549.

[4] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*, 2.

