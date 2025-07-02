import streamlit as st
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Importer Streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas

# URL de l’API FastAPI (ajuste si besoin)
API_URL = "http://backend:8000/api/v1/predict"

st.title("Reconnaissance de chiffres manuscrits MNIST")

# Canvas pour dessiner
canvas_result = st_canvas(
    fill_color="#000000",  # noir
    stroke_width=10,
    stroke_color="#FFFFFF",  # blanc (car fond noir)
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Afficher l’image dessinée (convertie en 28x28 pour le modèle)
    img = (
        Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        .convert("L")
        .resize((28, 28))
    )
    st.image(img, caption="Image envoyée au modèle", width=150)

    # Convertir en bytes pour envoyer
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    if st.button("Envoyer à l'API pour prédiction"):
        with st.spinner("Prédiction en cours..."):
            files = {"file": ("drawing.png", img_bytes, "image/png")}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Chiffre prédit : {result['predicted_digit']} avec confiance {result['confidence']:.2%}"
                )
            else:
                st.error(f"Erreur API : {response.status_code} {response.text}")
