```python
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ================================
# 🔧 PFAD-KONFIGURATION
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "model", "labels.txt")


# ================================
# 🧠 FIX für DepthwiseConv2D BUG
# ================================
from tensorflow.keras.layers import DepthwiseConv2D

class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# ================================
# 📦 MODELL LADEN
# ================================
@st.cache_resource
def load_model_file():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modell nicht gefunden: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
        )
        return model

    except Exception as e:
        st.error("❌ Fehler beim Laden des Modells")
        st.exception(e)
        return None


# ================================
# 🏷️ LABELS LADEN
# ================================
def load_labels():
    if not os.path.exists(LABEL_PATH):
        st.error(f"❌ Labels nicht gefunden: {LABEL_PATH}")
        return []

    labels = []
    try:
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split(" ", 1)
                labels.append(parts[1] if len(parts) > 1 else parts[0])
    except Exception as e:
        st.error("❌ Fehler beim Lesen der labels.txt")
        st.exception(e)

    return labels


# ================================
# 🖼️ BILD PREPROCESSING
# ================================
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.asarray(img)

    normalized = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)

    return data


# ================================
# 🔮 VORHERSAGE
# ================================
def predict(model, image):
    data = preprocess_image(image)

    prediction = model.predict(data)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])

    return index, confidence


# ================================
# 🎨 UI SETUP
# ================================
st.set_page_config(page_title="Schul-Fundbüro KI", page_icon="🏫")

st.title("🏫 Schul-Fundbüro KI-App")
st.write("Lade ein Bild eines verlorenen Gegenstands hoch, um ihn automatisch zu kategorisieren.")


# Debug-Bereich
with st.expander("🔧 Debug-Infos"):
    st.write("MODEL_PATH:", MODEL_PATH)
    st.write("LABEL_PATH:", LABEL_PATH)
    st.write("Modell vorhanden:", os.path.exists(MODEL_PATH))
    st.write("Labels vorhanden:", os.path.exists(LABEL_PATH))
    st.write("TensorFlow Version:", tf.__version__)


# ================================
# 📥 LADEN
# ================================
model = load_model_file()
labels = load_labels()


# ================================
# 📷 FILE UPLOAD
# ================================
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if model is None:
        st.error("❌ Modell konnte nicht geladen werden.")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")

    # ✅ kompatibel mit alten + neuen Streamlit Versionen
    try:
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
    except TypeError:
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    with st.spinner("🔍 KI analysiert das Fundstück..."):
        index, confidence = predict(model, image)

    st.divider()
    st.subheader("🔎 Analyse-Ergebnis")

    if labels and index < len(labels):
        label_name = labels[index]

        col1, col2 = st.columns(2)
        col1.metric("Gegenstand", label_name)
        col2.metric("Sicherheit", f"{confidence * 100:.1f}%")

        st.progress(confidence)

        if confidence < 0.6:
            st.warning("⚠️ Die KI ist sich unsicher. Bitte manuell prüfen.")

    else:
        st.error("❌ Vorhersage konnte keinem Label zugeordnet werden.")


# ================================
# 📎 FOOTER
# ================================
st.sidebar.info("💡 Tipp: Gute Beleuchtung + neutraler Hintergrund verbessern die Erkennung.")
```
