import streamlit as st
import keras
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

# ── Constantes (ajustables según política del negocio) ────────────────────
MODEL_PATH         = os.getenv("MODEL_PATH",         "modelo_siames_firmas_final.keras")
UMBRAL_GENUINO     = float(os.getenv("UMBRAL_GENUINO",     0.40))
UMBRAL_FALSIFICADA = float(os.getenv("UMBRAL_FALSIFICADA", 0.60))

# ── Carga del modelo (cacheado para no recargar en cada interacción) ──────
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Modelo no encontrado en '{MODEL_PATH}'. "
            "Asegúrese de que 'modelo_siames_firmas_final.keras' "
            "está en la raíz del repositorio."
        )
        st.stop()
    keras.config.enable_unsafe_deserialization()
    return keras.models.load_model(MODEL_PATH)

# ── Preprocesamiento ──────────────────────────────────────────────────────
def preprocesar_desde_pil(imagen_pil, tam=(105, 105)):
    """
    Convierte una imagen PIL a array numpy preprocesado para la Red Siamesa.
    - Escala de grises, redimensión a 105×105, normalización [0,1], reshape (105,105,1).
    """
    img = np.array(imagen_pil.convert("L"))
    img = cv2.resize(img, tam)
    img = img / 255.0
    return img.reshape(tam[0], tam[1], 1).astype(np.float32)

# ── Inferencia ────────────────────────────────────────────────────────────
def verificar_firma(img_ref, img_sosp, modelo):
    """
    Compara dos imágenes preprocesadas y retorna resultado, score, alerta y color.
    """
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

# ── Interfaz de usuario ───────────────────────────────────────────────────
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
            modelo       = cargar_modelo()
            img_ref_arr  = preprocesar_desde_pil(Image.open(archivo_ref))
            img_sosp_arr = preprocesar_desde_pil(Image.open(archivo_sosp))
            resultado, score, alerta, color = verificar_firma(
                img_ref_arr, img_sosp_arr, modelo
            )

        st.markdown(f"### Resultado: :{color}[{resultado}]")

        col_score, col_umbral = st.columns(2)
        with col_score:
            st.metric(label="Score de falsificación", value=f"{score:.4f}")
        with col_umbral:
            st.metric(
                label="Umbrales configurados",
                value=f"{UMBRAL_GENUINO} / {UMBRAL_FALSIFICADA}"
            )

        st.info(f"**Acción recomendada:** {alerta}")

        with st.expander("ℹ️ Cómo interpretar el score"):
            st.markdown(
                f"""
                | Score | Resultado | Acción |
                |---|---|---|
                | < {UMBRAL_GENUINO} | ✅ GENUINO | Aprobación automática |
                | {UMBRAL_GENUINO} – {UMBRAL_FALSIFICADA} | ⚠️ INCIERTO | Derivar a revisión humana |
                | > {UMBRAL_FALSIFICADA} | 🚨 FALSIFICADO | Rechazo y alerta de fraude |
                """
            )

        st.divider()
        st.caption(
            "⚠️ Este sistema es una herramienta de **apoyo a la decisión**, "
            "no reemplaza al perito calígrafo. Los casos en zona de incertidumbre "
            "deben ser revisados por un especialista antes de tomar una decisión definitiva."
        )
