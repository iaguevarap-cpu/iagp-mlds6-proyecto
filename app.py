import streamlit as st
import keras
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# ── Configuración de la página ────────────────────────────────────────────
st.set_page_config(
    page_title="Verificador de Firmas",
    page_icon="✍️",
    layout="centered"
)

# ── Capa personalizada L1 — DEBE definirse ANTES de load_model ───────────
# Sin esto, Keras no puede deserializar 'firmas>DistanciaL1' del archivo .keras
@keras.saving.register_keras_serializable(package="firmas")
class DistanciaL1(keras.layers.Layer):
    """Distancia L1 entre dos embeddings de la Red Siamesa."""
    def call(self, tensors):
        return tf.abs(tensors[0] - tensors[1])
    def compute_output_shape(self, input_shape):
        return input_shape[0]

# ── Constantes ────────────────────────────────────────────────────────────
MODEL_PATH         = os.getenv("MODEL_PATH",         "modelo_siames_firmas_v2.keras")
UMBRAL_GENUINO     = float(os.getenv("UMBRAL_GENUINO",     0.40))
UMBRAL_FALSIFICADA = float(os.getenv("UMBRAL_FALSIFICADA", 0.60))

# ── Carga del modelo (cacheado) ───────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Modelo no encontrado en '{MODEL_PATH}'. "
            "Asegúrese de que 'modelo_siames_firmas_v2.keras' "
            "está en la raíz del repositorio."
        )
        st.stop()
    return keras.models.load_model(MODEL_PATH)

# ── Preprocesamiento ──────────────────────────────────────────────────────
def preprocesar_desde_pil(imagen_pil, tam=(105, 105)):
    """Convierte PIL → array numpy (105,105,1) normalizado [0,1]."""
    img = np.array(imagen_pil.convert("L"))
    img = cv2.resize(img, tam)
    img = img / 255.0
    return img.reshape(tam[0], tam[1], 1).astype(np.float32)

# ── Inferencia ────────────────────────────────────────────────────────────
def verificar_firma(img_ref, img_sosp, modelo):
    score = modelo.predict(
        [img_ref.reshape(1, 105, 105, 1),
         img_sosp.reshape(1, 105, 105, 1)],
        verbose=0
    )[0][0]

    if score < UMBRAL_GENUINO:
        return "✅ GENUINO",     float(score), "Aprobación automática",       "green"
    elif score <= UMBRAL_FALSIFICADA:
        return "⚠️ INCIERTO",   float(score), "Derivar a revisión humana",   "orange"
    else:
        return "🚨 FALSIFICADO", float(score), "Rechazo y alerta de fraude",  "red"

# ── Interfaz ──────────────────────────────────────────────────────────────
st.title("✍️ Verificador de Firmas Adulteradas")
st.markdown(
    "Sistema de apoyo para la detección de firmas falsificadas basado en "
    "**Redes Siamesas con CNN** (AUC-ROC = 0.8184). "
    "Suba las dos imágenes de firma y obtenga el resultado de verificación."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Firma de referencia")
    st.caption("Firma conocida como genuina (registro del cliente)")
    archivo_ref = st.file_uploader(
        "Subir firma de referencia",
        type=["png", "jpg", "jpeg"],
        key="ref"
    )
    if archivo_ref:
        st.image(Image.open(archivo_ref), caption="Firma de referencia",
                 use_column_width=True)

with col2:
    st.subheader("Firma sospechosa")
    st.caption("Firma extraída del documento a verificar")
    archivo_sosp = st.file_uploader(
        "Subir firma sospechosa",
        type=["png", "jpg", "jpeg"],
        key="sosp"
    )
    if archivo_sosp:
        st.image(Image.open(archivo_sosp), caption="Firma sospechosa",
                 use_column_width=True)

st.divider()

if st.button("🔍 Verificar firma", type="primary", use_container_width=True):
    if archivo_ref is None or archivo_sosp is None:
        st.error("⚠️ Por favor suba ambas imágenes antes de verificar.")
    else:
        with st.spinner("Analizando el par de firmas..."):
            modelo_cargado = cargar_modelo()
            img_ref_arr    = preprocesar_desde_pil(Image.open(archivo_ref))
            img_sosp_arr   = preprocesar_desde_pil(Image.open(archivo_sosp))
            resultado, score, alerta, color = verificar_firma(
                img_ref_arr, img_sosp_arr, modelo_cargado
            )

        st.markdown(f"### Resultado: :{color}[{resultado}]")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Score de falsificación", f"{score:.4f}")
        with col_b:
            st.metric("Umbrales", f"{UMBRAL_GENUINO} / {UMBRAL_FALSIFICADA}")

        st.info(f"**Acción recomendada:** {alerta}")

        with st.expander("ℹ️ Cómo interpretar el score"):
            st.markdown(
                f"| Score | Resultado | Acción |\n"
                f"|---|---|---|\n"
                f"| < {UMBRAL_GENUINO} | ✅ GENUINO | Aprobación automática |\n"
                f"| {UMBRAL_GENUINO}–{UMBRAL_FALSIFICADA} | ⚠️ INCIERTO | Revisión humana |\n"
                f"| > {UMBRAL_FALSIFICADA} | 🚨 FALSIFICADO | Rechazo + alerta |"
            )

        st.divider()
        st.caption(
            "⚠️ Este sistema es una herramienta de **apoyo a la decisión**, "
            "no reemplaza al perito calígrafo."
        )
