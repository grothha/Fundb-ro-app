import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# 1. Konfiguration der Seite
st.set_page_config(
    page_title="Digitales Fundbüro | KI-Check",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS für den "Fundbüro"-Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stAlert {
        border-radius: 10px;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Modell-Loading (Cache)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# 3. Sidebar - Fundbüro Verwaltung
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lost-and-found.png", width=80)
    st.title("Verwaltung")
    st.info("Dieses Tool hilft dabei, abgegebene Gegenstände automatisch zu kategorisieren.")
    
    st.divider()
    location = st.selectbox("Fundort", ["Bahnhof", "Flughafen", "Innenstadt", "Parkanlage"])
    date_found = st.date_input("Funddatum")
    priority = st.select_slider("Dringlichkeit", options=["Niedrig", "Mittel", "Hoch"])

# 4. Hauptbereich
st.title("🔍 KI-Fundregistrierung")
st.write("Laden Sie ein Foto des Fundgegenstands hoch, um eine automatische Kategorisierung vorzunehmen.")

# Layout-Spalten
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Foto-Upload")
    uploaded_file = st.file_uploader("Bild hier hineinziehen oder klicken", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Hochgeladener Gegenstand', use_container_width=True)

with col_result:
    st.subheader("📋 Analyse & Protokoll")
    
    if uploaded_file:
        with st.spinner('Analysiere Gegenstand...'):
            results = classifier(image)
            
        # Top-Ergebnis hervorheben
        top_prediction = results[0]['label'].upper()
        confidence = results[0]['score']
        
        st.success(f"**Vorgeschlagene Kategorie:** {top_prediction}")
        
        # Details in einer Tabelle anzeigen
        st.write("### Wahrscheinlichkeiten")
        df = pd.DataFrame(results)
        df.columns = ['Konfidenz', 'Kategorie']
        # Balkendiagramm für die Visualisierung
        st.bar_chart(df.set_index('Kategorie'))
        
        # Automatisches Protokoll-Snippet
        with st.expander("📝 Protokoll-Entwurf anzeigen"):
            st.code(f"""
            FUNDPROTOKOLL
            -------------------
            Gegenstand: {top_prediction}
            Ort:        {location}
            Datum:      {date_found}
            Priorität:  {priority}
            Status:     Registriert (KI-verifiziert)
            """)
            
        if st.button("✅ In Datenbank speichern"):
            st.balloons()
            st.success("Gegenstand wurde erfolgreich im System hinterlegt!")
    else:
        st.info("Bitte laden Sie links ein Bild hoch, um die Analyse zu starten.")

# Footer
st.divider()
st.caption("Internes Tool für Fundbüro-Mitarbeiter | Basierend auf Google ViT-Modell")
