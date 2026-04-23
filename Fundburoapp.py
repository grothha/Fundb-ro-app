import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Seite konfigurieren
st.set_page_config(page_title="KI Fundbüro", layout="centered")

# HIER IST DIE ÄNDERUNG: Wir suchen nach deinem Dateinamen
MODEL_FILENAME = "keras_model.h5"

def find_model_path(target_file):
    """Durchsucht das gesamte Projekt nach der Modelldatei."""
    # Erst im Hauptverzeichnis schauen
    if os.path.exists(target_file):
        return target_file
    
    # Dann in allen Unterordnern suchen
    for root, dirs, files in os.walk("."):
        if target_file in files:
            return os.path.join(root, target_file)
    return None

@st.cache_resource
def load_model():
    path = find_model_path(MODEL_FILENAME)
    
    if path:
        try:
            # Modell laden
            model = tf.keras.models.load_model(path)
            return model, path
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei {path}: {e}")
            return None, None
    else:
        # Fehlermeldung, falls die Datei gar nicht existiert
        st.error(f"Die Datei '{MODEL_FILENAME}' wurde nicht gefunden!")
        st.write("Gefundene Dateien im Ordner:", os.listdir("."))
        return None, None

def predict_image(model, image):
    # Vorverarbeitung
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
        st.success(f"Modell geladen: {path}")

    uploaded_file = st.file_uploader("Bild des Fundstücks hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
        
        if st.button("Gegenstand erkennen"):
            if model:
                with st.spinner("KI denkt nach..."):
                    prediction = predict_image(model, image)
                    
                    # BITTE PASS DIESE LISTE AN DEINE KLASSEN AN:
                    labels = ["Schlüssel", "Geldbeutel", "Handy", "Brille", "Regenschirm"]
                    
                    idx = np.argmax(prediction[0])
                    if idx < len(labels):
                        st.metric("Ergebnis", labels[idx])
                        st.write(f"Sicherheit: {np.max(prediction[0])*100:.2f}%")
                    else:
                        st.write("Klasse erkannt, aber kein Name in der Liste hinterlegt.")
            else:
                st.error("Modell ist nicht bereit.")

if __name__ == "__main__":
    main()
