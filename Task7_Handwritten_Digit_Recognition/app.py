import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("digit_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    img = np.array(image)

    img = cv2.resize(img, (28,28))

    img = img / 255.0

    img = img.reshape(1,28,28)

    prediction = model.predict(img)

    digit = np.argmax(prediction)

    st.image(image)

    st.success(f"Predicted Digit: {digit}")