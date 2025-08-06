import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load trained model
model = load_model("mobilenetv2_fish_model.keras")  # 🔁 Change to your best model

# Class labels (replace with your actual class names)
class_names = ['Salmon', 'Tuna', 'Trout', 'Hilsa', 'Rohu', 'Catfish', 'Pomfret', 'Seabass']

# Streamlit UI
st.title("🐟 Fish Species Classifier")
st.markdown("Upload an image of a fish and the model will predict its species.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(128, 128))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"🐠 Predicted: **{predicted_class}**")
    st.info(f"🧠 Confidence: {confidence*100:.2f}%")
