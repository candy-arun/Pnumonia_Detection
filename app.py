import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# -------------------------
# Load saved model
# -------------------------
model = tf.keras.models.load_model("Trained_Model.keras")

# Title
st.title("Pneumonia Detection App")
st.write("Upload a chest X-ray image to classify it as **NORMAL** or **PNEUMONIA**.")


# -------------------------
# Prediction Function
# -------------------------
def predict_image(model, img):

    # convert PIL → numpy
    img = np.array(img)

    # if image is grayscale, convert to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # resize
    img_resized = cv2.resize(img, (150,150))

    # normalize
    img_normalized = img_resized / 255.0

    # batch dimension
    img_input = np.expand_dims(img_normalized, axis=0)

    # prediction
    pred = model.predict(img_input)[0][0]

    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = float(pred)

    return label, confidence


# -------------------------
# File Upload Section
# -------------------------
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # predict button
    if st.button("Predict"):
        label, score = predict_image(model, img)

        st.subheader("Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Score:** {score:.4f}")

        if label == "PNEUMONIA":
            st.error("⚠️ Pneumonia Detected")
        else:
            st.success("✔️ Normal Chest X-ray")
