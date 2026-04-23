import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Konfiguration der Seite
st.set_page_config(page_title="KI Fundbüro", layout="centered")

@st.cache_resource
def load_model():
    # Lädt das vortrainierte Modell (Pfad muss ggf. angepasst werden)
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

def predict_image(model, image):
    # Bildvorverarbeitung für das Modell
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    return prediction

def main():
    st.title("🔍 KI-gestütztes Fundbüro")
    st.write("Laden Sie ein Foto des gefundenen Gegenstands hoch, um ihn zu klassifizieren.")

    model = load_model()

    uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Korrektur: use_column_width statt use_container_width für Streamlit 1.33.0
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
        
        st.write("Analysiere...")
        
        if model is not None:
            prediction = predict_image(model, image)
            # Hier müsste die Logik zur Anzeige der Klassen (Labels) folgen
            st.success("Analyse abgeschlossen!")
            st.write(f"Rohdaten der Vorhersage: {prediction[0]}")
        else:
            st.warning("Das Modell konnte nicht geladen werden. Bitte prüfen Sie, ob 'model.h5' vorhanden ist.")

if __name__ == "__main__":
    main()
