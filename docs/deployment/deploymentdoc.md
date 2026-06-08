# Despliegue de modelos

**Proyecto:** Identificación de Firmas Adulteradas con Redes Siamesas
**Autor:** Ivonne Alexandra Guevara Prieto — CC 1032376081
**Fecha:** Junio 2025
**Notebook de referencia:** `scripts/5_deployment.ipynb`

---

## 1. Infraestructura

- **Nombre del modelo:** `modelo_siames_firmas_final.keras` — Red Siamesa con CNN base para verificación de firmas manuscritas.

- **Plataforma de despliegue:** [Streamlit Cloud](https://streamlit.io/cloud) — plataforma de despliegue gratuita para aplicaciones de datos en Python. Permite desplegar la aplicación directamente desde el repositorio de GitHub sin configuración adicional de servidor.

- **Requisitos técnicos:**

  | Componente | Versión | Uso |
  |---|---|---|
  | Python | ≥ 3.10 | Lenguaje base |
  | TensorFlow / Keras | ≥ 2.13 | Carga y ejecución del modelo |
  | OpenCV (`opencv-python-headless`) | ≥ 4.8 | Preprocesamiento de imágenes |
  | NumPy | ≥ 1.24 | Operaciones matriciales |
  | Streamlit | ≥ 1.32 | Interfaz web de la aplicación |
  | Pillow | ≥ 10.0 | Lectura de imágenes subidas por el usuario |
  | RAM mínima | 1 GB | Carga del modelo en memoria |
  | Almacenamiento | ~50 MB | Modelo `.keras` + dependencias |
  | GPU | No requerida | Inferencia en CPU es suficiente (<2s por par) |

- **Requisitos de seguridad:**

  - **Autenticación:** No requerida para el prototipo académico. En un despliegue bancario de producción se requeriría autenticación con OAuth2 o JWT.
  - **Encriptación:** Las imágenes subidas a Streamlit Cloud se procesan en memoria y no se persisten en disco. No se almacena información personal del usuario.
  - **Modelo protegido:** El archivo `modelo_siames_firmas_final.keras` no debe exponerse públicamente. En producción debe almacenarse en un bucket privado (S3/GCS) con acceso restringido por IAM.
  - **HTTPS:** Streamlit Cloud provee HTTPS por defecto en el dominio `.streamlit.app`.
  - **Datos sensibles:** Las firmas son datos biométricos. En producción, el sistema debe cumplir con las regulaciones locales de protección de datos (Ley 1581 de 2012 en Colombia, GDPR en Europa).

- **Diagrama de arquitectura:**

  ```
  ┌─────────────────────────────────────────────────────────┐
  │                    USUARIO FINAL                        │
  │         (operador bancario / analista de fraude)        │
  └──────────────────────┬──────────────────────────────────┘
                         │ HTTPS — sube 2 imágenes PNG/JPG
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │              STREAMLIT CLOUD (Frontend + Backend)       │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  app.py (Streamlit)                             │   │
  │  │  1. Recibe imagen_referencia + imagen_sospechosa│   │
  │  │  2. Llama preprocesar_imagen() [preprocessing.py]│   │
  │  │  3. Llama verificar_firma()    [predict.py]     │   │
  │  │  4. Muestra score + nivel de alerta             │   │
  │  └──────────────────┬──────────────────────────────┘   │
  │                     │ carga en inicio                   │
  │  ┌──────────────────▼──────────────────────────────┐   │
  │  │  modelo_siames_firmas_final.keras               │   │
  │  │  (Red Siamesa CNN, ~50 MB, cargado en RAM)      │   │
  │  └─────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                 RESPUESTA AL USUARIO                    │
  │  score ∈ [0,1]  +  nivel de alerta:                    │
  │  ✅ GENUINO / ⚠️ INCIERTO / 🚨 FALSIFICADO             │
  └─────────────────────────────────────────────────────────┘
  ```

---

## 2. Código de despliegue

- **Archivo principal:** `app.py` — aplicación Streamlit que carga el modelo y expone la interfaz de verificación de firmas.

- **Rutas de acceso a los archivos:**

  ```
  iagp-mlds6-proyecto/
  ├── app.py                                ← Archivo principal de la app
  ├── modelo_siames_firmas_final.keras      ← Modelo entrenado
  ├── requirements.txt                      ← Dependencias para Streamlit Cloud
  └── src/
      └── firmas_adulteradas/
          ├── __init__.py
          ├── preprocessing.py              ← preprocesar_imagen()
          └── predict.py                    ← verificar_firma()
  ```

- **Variables de entorno:**

  | Variable | Descripción | Valor por defecto |
  |---|---|---|
  | `MODEL_PATH` | Ruta al modelo `.keras` | `modelo_siames_firmas_final.keras` |
  | `UMBRAL_GENUINO` | Umbral inferior de decisión | `0.40` |
  | `UMBRAL_FALSIFICADA` | Umbral superior de decisión | `0.60` |

  En Streamlit Cloud las variables de entorno se configuran en **Settings → Secrets** usando la sintaxis TOML:
  ```toml
  MODEL_PATH = "modelo_siames_firmas_final.keras"
  UMBRAL_GENUINO = 0.40
  UMBRAL_FALSIFICADA = 0.60
  ```

---

## 3. Documentación del despliegue

### 3.1 Instrucciones de instalación

**Opción A — Ejecución local:**

```bash
# 1. Clonar el repositorio
git clone https://github.com/iaguevarap-cpu/iagp-mlds6-proyecto.git
cd iagp-mlds6-proyecto

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

**Contenido de `requirements.txt`:**
```
tensorflow>=2.13.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
streamlit>=1.32.0
Pillow>=10.0.0
```

**Opción B — Despliegue en Streamlit Cloud (recomendado):**

1. Hacer push del repositorio a GitHub (asegurarse de que `app.py` y `modelo_siames_firmas_final.keras` están en la raíz).
2. Ir a [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Seleccionar el repositorio `iagp-mlds6-proyecto`, rama `main`, archivo `app.py`.
4. Hacer clic en **Deploy**. La app estará disponible en `https://<nombre>.streamlit.app` en ~2 minutos.
5. No se requiere configuración adicional de servidor ni de GPU.

### 3.2 Instrucciones de configuración

1. **Modelo:** Copiar el archivo `modelo_siames_firmas_final.keras` (generado en `scripts/4_modeling.ipynb`) a la raíz del repositorio antes del despliegue.
2. **Umbrales:** Los umbrales de decisión (`UMBRAL_GENUINO=0.40`, `UMBRAL_FALSIFICADA=0.60`) pueden ajustarse en `src/firmas_adulteradas/predict.py` o mediante variables de entorno según la política de riesgo del cliente.
3. **Tamaño de imagen:** El modelo espera imágenes de cualquier resolución — el preprocesamiento las redimensiona automáticamente a 105×105 px en escala de grises.
4. **Formato de entrada:** Se aceptan imágenes PNG, JPG y JPEG escaneadas o fotografiadas.

### 3.3 Instrucciones de uso

La aplicación expone una interfaz web simple de tres pasos:

**Paso 1 — Subir las dos imágenes de firma:**
- `Firma de referencia`: imagen de la firma conocida como genuina (extraída del registro del cliente).
- `Firma sospechosa`: imagen de la firma a verificar (extraída del documento en cuestión).

**Paso 2 — Ejecutar la verificación:**
- Hacer clic en el botón **"Verificar firma"**.
- El modelo procesa el par en menos de 2 segundos.

**Paso 3 — Interpretar el resultado:**

| Score | Resultado | Acción recomendada |
|---|---|---|
| < 0.40 | ✅ GENUINO | Aprobación automática |
| 0.40 – 0.60 | ⚠️ INCIERTO | Derivar a revisión humana |
| > 0.60 | 🚨 FALSIFICADO | Rechazo y alerta de fraude |

> **Nota:** El sistema es una herramienta de **apoyo a la decisión**, no reemplaza al perito calígrafo. Los casos en zona de incertidumbre (⚠️) deben ser revisados por un especialista antes de tomar una decisión definitiva.

**Uso programático (desde Python):**
```python
import keras
from src.firmas_adulteradas.predict import verificar_firma

# Cargar modelo
modelo = keras.models.load_model("modelo_siames_firmas_final.keras")

# Verificar un par de firmas
resultado = verificar_firma(
    ruta_referencia="firma_referencia.png",
    ruta_sospechosa="firma_sospechosa.png",
    modelo=modelo
)
# Retorna: {'resultado': 'GENUINO', 'score': 0.23, 'alerta': 'Aprobación automática', 'nivel': '✅'}
```

### 3.4 Instrucciones de mantenimiento

**Monitoreo del modelo:**
- Registrar en un log la distribución de scores para detectar **drift** en producción: si la media de scores cambia significativamente con el tiempo, el modelo puede requerir reentrenamiento.
- Monitorear la tasa de derivaciones a revisión humana (zona 0.40–0.60). Un aumento sostenido indica degradación del modelo.

**Reentrenamiento:**
- Reentrenar el modelo periódicamente incorporando los casos revisados por el perito como nuevos datos de entrenamiento (*active learning*).
- Ejecutar `scripts/4_modeling.ipynb` con el dataset actualizado y reemplazar `modelo_siames_firmas_final.keras` en el repositorio.
- Hacer nuevo deploy en Streamlit Cloud haciendo push del modelo actualizado a GitHub.

**Actualizaciones de dependencias:**
- Revisar compatibilidad de TensorFlow/Keras al actualizar versiones, ya que el formato `.keras` puede tener cambios entre versiones mayores.
- Ejecutar `pip install --upgrade -r requirements.txt` en un entorno de prueba antes de actualizar producción.

**Costos de infraestructura:**

| Componente | Plan | Costo mensual |
|---|---|---|
| Streamlit Cloud | Community (gratuito) | $0 |
| GitHub (repositorio) | Free | $0 |
| Almacenamiento modelo | Repositorio GitHub (<100 MB) | $0 |
| **Total operativo** | | **$0 / mes** |

> En un escenario de producción bancaria, la infraestructura migrarías a servicios cloud empresariales. Costos estimados de referencia:

| Componente | Servicio | Costo estimado |
|---|---|---|
| API de predicción | AWS Lambda o Cloud Run | ~$5–20/mes según volumen |
| Almacenamiento del modelo | S3 / GCS (~50 MB) | ~$0.01/mes |
| Monitoreo y logs | AWS CloudWatch / GCP Logging | ~$5–15/mes |
| **Total producción (estimado)** | | **~$10–35/mes** |

