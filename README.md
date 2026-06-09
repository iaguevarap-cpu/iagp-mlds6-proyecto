# Identificación de Firmas Adulteradas con Redes Siamesas

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![TDSP](https://img.shields.io/badge/Metodología-TDSP-navy)](https://github.com/mindlab-unal/tdsp_template)
[![License](https://img.shields.io/badge/Licencia-MIT-green)](LICENSE)
[![UNAL](https://img.shields.io/badge/UNAL-MLDS%20Módulo%206-yellow)](https://unal.edu.co/)

Proyecto aplicado del **Módulo 6 — Desarrollo de Aplicaciones con Machine Learning**  
Programa MLDS · Universidad Nacional de Colombia · Facultad de Ingeniería · 2025

**Autora:** Ivonne Alexandra Guevara Prieto · CC 1032376081

---

## Descripción

Sistema de verificación automática de firmas manuscritas basado en una **Red Siamesa con CNN**. Dado un par de imágenes de firmas (A, B), el modelo produce un score de similitud que determina si el par es genuino o falsificado, apoyando la detección de fraude documental en instituciones financieras y legales.

| Métrica | Resultado |
|---|---|
| AUC-ROC (test) | **0.8184** |
| Recall clase falsificada | **95%** |
| Arquitectura | Red Siamesa + CNN base |
| Dataset | CEDAR (1,649 imágenes, 55 individuos) |
| Despliegue | Streamlit Cloud (gratuito) |

---

## Estructura del repositorio

```
iagp-mlds6-proyecto/
│
├── docs/                                   ← Documentación TDSP
│   ├── project/
│   │   └── project_charter.md             ← Marco de proyecto, presupuesto, riesgos
│   ├── data/
│   │   ├── data_definition.md             ← Diccionario del dataset CEDAR
│   │   ├── data_summary.md                ← Resumen estadístico de los datos
│   │   └── data_report.md                 ← Reporte EDA completo
│   ├── modeling/
│   │   ├── baseline_models.md             ← Comparación CNN base vs EfficientNetB0
│   │   └── model_report.md                ← Modelo final, métricas e interpretación
│   └── deployment/
│       └── deployment.md                  ← Infraestructura, instrucciones, costos
│
├── scripts/                                ← Notebooks Jupyter (fases CRISP-DM)
│   ├── 1_business_data_load.ipynb         ← Fase 1: Negocio + carga del dataset
│   ├── 2_exploratory_data_analysis.ipynb  ← Fase 2: EDA + preprocesamiento
│   ├── 3_experimental_set_up.ipynb        ← Fase 3: Diseño + arquitectura Siamesa
│   ├── 4_modeling.ipynb                   ← Fase 4: Entrenamiento + evaluación
│   └── 5_deployment.ipynb                 ← Fase 5: Despliegue Streamlit
│
├── src/
│   └── firmas_adulteradas/                ← Módulos Python reutilizables
│       ├── __init__.py
│       ├── preprocessing.py               ← preprocesar_imagen(), construir_pares_desde_csv()
│       └── predict.py                     ← verificar_firma(), niveles de alerta
│
├── app.py                                  ← Aplicación Streamlit
├── requirements.txt                        ← Dependencias para Streamlit Cloud
├── modelo_siames_firmas_v2.keras          ← Modelo entrenado
├── pyproject.toml                          ← Configuración del proyecto Python
├── .gitignore
└── README.md
```

---

## Quickstart

### Requisitos

- Python 3.11
- Google Colab (para entrenamiento) o entorno local con GPU

### Instalación local

```bash
git clone https://github.com/iaguevarap-cpu/iagp-mlds6-proyecto.git
cd iagp-mlds6-proyecto
pip install -r requirements.txt
streamlit run app.py
```

### Reproducir el experimento

Ejecutar los notebooks en orden desde `scripts/`:

```
1_business_data_load.ipynb       → descarga y verifica el dataset
2_exploratory_data_analysis.ipynb → EDA y preprocesamiento
3_experimental_set_up.ipynb      → diseño de la arquitectura
4_modeling.ipynb                 → entrenamiento y evaluación
5_deployment.ipynb               → genera app.py y requirements.txt
```

---

## Sistema de verificación

El modelo clasifica cada par de firmas en tres niveles de alerta:

| Score | Resultado | Acción recomendada |
|---|---|---|
| < 0.40 | ✅ GENUINO | Aprobación automática |
| 0.40 – 0.60 | ⚠️ INCIERTO | Derivar a revisión humana |
| > 0.60 | 🚨 FALSIFICADO | Rechazo y alerta de fraude |

---

## Dataset

**CEDAR Signature Verification Dataset**  
Fuente: [Kaggle — Robin Reni](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)  
Licencia: CC0 Public Domain

```python
import kagglehub
path = kagglehub.dataset_download("robinreni/signature-verification-dataset")
```

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Deep Learning | TensorFlow 2.15 / Keras |
| Datos | kagglehub, pandas, NumPy, OpenCV |
| Despliegue | Streamlit Cloud |
| Versionamiento | Git + GitHub |
| Metodología | TDSP (mindlab-unal) |

---

## Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iagp-mlds6-proyecto-firmas-adulteradas.streamlit.app/)


## Referencia principal

Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese neural networks for one-shot image recognition*. ICML Deep Learning Workshop, 2.

---

**Universidad Nacional de Colombia · Facultad de Ingeniería · MLDS · 2025**
