import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Seite konfigurieren
st.set_page_config(page_title="KI Fundbüro", layout="centered")

def find_model_path(target_file):
    """Durchsucht das gesamte Projekt nach der Modelldatei."""
    for root, dirs, files in os.walk("."):
        if target_file in files:
            return os.path.join(root, target_file)
    return None

@st.cache_resource
def load_model():
    # Wir suchen nach 'model.h5' egal in welchem Ordner
    path = find_model_path("model.h5")
    
    if path:
        try:
            model = tf.keras.models.load_model(path)
            return model, path
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")
            return None, None
    else:
        # Fehlermeldung mit Debug-Info
        st.error("Modell 'model.h5' wurde im gesamten Projekt nicht gefunden!")
        st.write("Vorhandene Dateien im Hauptverzeichnis:", os.listdir("."))
        return None, None

def predict_image(model, image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return model.predict(img_array)

def main():
    st.title("🔍 KI-gestütztes Fundbüro")
    
    model, path = load_model()
    
    if path:
        st.success(f"Modell erfolgreich geladen aus: {path}")

    uploaded_file = st.file_uploader("Bild des Fundstücks hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
        
        if st.button("Gegenstand erkennen"):
            if model:
                prediction = predict_image(model, image)
                
                # BITTE HIER DEINE KLASSENNAMEN ANPASSEN:
                labels = ["Schlüssel", "Geldbeutel", "Handy", "Brille", "Regenschirm"]
                
                idx = np.argmax(prediction[0])
                if idx < len(labels):
                    st.metric("Ergebnis", labels[idx])
                else:
                    st.write("Unbekannte Klasse:", idx)
            else:
                st.error("Aktion nicht möglich, da Modell fehlt.")

if __name__ == "__main__":
    main()
