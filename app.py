import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from src.model import create_model, get_anchors
from src.inference_utils import process_predictions, visualize_detection

# Page config
st.set_page_config(page_title="Face Mask Detection", layout="wide")

st.title("Face Mask Detection System (MobileNetV2)")
st.write("Upload an image to detect if people are wearing masks correctly.")

# Load Model
@st.cache_resource
def load_model():
    model = create_model(num_classes=4)
    # Try loading weights
    try:
        model.load_weights('saved_models/face_mask_model.h5')
        return model
    except:
        return None

model = load_model()

if model is None:
    st.error("Model weights not found. Please train the model first.")
else:
    st.success("Model loaded successfully!")


# Anchors
anchors = get_anchors()

# Input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # Fix deprecation warning
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Add controls
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.5, 0.05)
    
    if st.button("Detect Masks"):
        with st.spinner('Processing...'):
            # Preprocess
            img_resized = image.resize((224, 224))
            img_arr = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_arr.astype(np.float32), axis=0)
            
            # Predict
            cls_pred, reg_pred = model.predict(img_batch)
            
            # Debug: check raw max confidence
            # cls_pred shape: (1, 245, 4)
            # Ignore background (idx 3)
            # Get max prob for class 0,1,2
            valid_probs = cls_pred[0, :, :3]
            max_conf = np.max(valid_probs)
            st.write(f"Max detection confidence found: {max_conf:.4f}")

            # Process
            boxes, classes, scores = process_predictions(cls_pred, reg_pred, anchors, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            
            # Visualize
            st.write(f"Found {len(boxes)} faces.")
            if len(boxes) > 0:
                fig = visualize_detection(np.array(image), boxes, classes, scores)
                st.pyplot(fig)
            else:
                st.warning("No faces detected. Try lowering the confidence threshold.")

