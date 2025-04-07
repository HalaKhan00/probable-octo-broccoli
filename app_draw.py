import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.image import resize
from tensorflow.keras.models import load_model
import cv2

# Load trained model
@st.cache_resource
def load_trained_model():
    return load_model('hybrid_cnn_rnn_model_final.keras')

model = load_trained_model()

# Dummy stroke converter (x, y, pen-down format)
def get_stroke_from_canvas(stroke_paths, max_len=100):
    strokes = []
    for path in stroke_paths:
        points = path.get("path", [])
        for i, point in enumerate(points):
            if isinstance(point, list) and len(point) >= 2:
                try:
                    x = float(point[0])
                    y = float(point[1])
                    pen = 1.0 if i < len(points) - 1 else 0.0
                    strokes.append([x, y, pen])
                except ValueError:
                    continue  # skip bad points like "M" or malformed coords
    strokes = strokes[:max_len]
    while len(strokes) < max_len:
        strokes.append([0.0, 0.0, 0.0])
    return np.array(strokes, dtype=np.float32)


# Image processor
def preprocess_canvas_image(canvas_img):
    if canvas_img is None:
        raise ValueError("Canvas image is None. Please draw something first.")

    image = np.array(canvas_img)

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    image = Image.fromarray(image).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = resize(image, (224, 224)).numpy()
    return image


# UI
st.title("🧠 Clock Drawing Alzheimer Predictor (Draw on Canvas)")

st.markdown("✍️ Draw a clock below:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=3,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None and canvas_result.json_data is not None:
    # Show drawing
    st.image(canvas_result.image_data, caption="Your Clock Drawing", width=250)

    if st.button("🧠 Submit for Prediction"):
        stroke_array = get_stroke_from_canvas(canvas_result.json_data["objects"])
        image_array = preprocess_canvas_image(canvas_result.image_data)

        pred = model.predict({
            "image_input": np.expand_dims(image_array, axis=0),
            "sequence_input": np.expand_dims(stroke_array, axis=0)
    })

        confidence = float(pred[0][0])
        if confidence < 0.5:
            st.subheader("Prediction Result:")
            st.success(f"✅ Healthy (Confidence: {1 - confidence:.2f})")
        else:
            st.subheader("Prediction Result:")
            st.error(f"🧠 Alzheimer Detected (Confidence: {confidence:.2f})")
        st.caption(f"🔍 Raw model score: {confidence:.4f}")
