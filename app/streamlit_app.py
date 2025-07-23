"""
Streamlit app for thyroid nodule classification and Grad-CAM visualization.
"""

import streamlit as st
import torch
import numpy as np
import os
import tempfile
from src.models.model import get_model
from src.models.inference import preprocess_image, grad_cam
from src.utils.visualization import plot_roc, plot_confusion_matrix, plot_class_distribution
from src.utils.logging import log_user_action, log_prediction

st.set_page_config(page_title="Thyroid Nodule Classifier", layout="wide")

# -- Security: only allow PNG, JPEG. Max 5MB.
ALLOWED_TYPES = ["png", "jpg", "jpeg"]
MAX_SIZE_MB = 5

def is_valid_file(uploaded_file):
    if uploaded_file is None:
        return False
    if not uploaded_file.name.split('.')[-1].lower() in ALLOWED_TYPES:
        return False
    if uploaded_file.size > MAX_SIZE_MB * 1024 * 1024:
        return False
    return True

@st.cache_resource
def load_model(ckpt_path):
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model

def main():
    st.title("🦋 Thyroid Nodule Classifier")
    st.write("Upload an ultrasound frame to predict malignancy with interpretability.")

    user = st.session_state.get('user', f"session_{st.session_state.get('run_id', 'anon')}")
    log_user_action(user, "dashboard_open")

    uploaded_file = st.file_uploader("Upload Ultrasound Frame (.png, .jpg, <5MB)", type=ALLOWED_TYPES)
    model_path = "checkpoints/best_model.pth"
    sha_path = model_path + ".sha256"

    if uploaded_file is not None:
        if not is_valid_file(uploaded_file):
            st.error("Invalid file type or size.")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        log_user_action(user, "file_upload", uploaded_file.name)

        # File session isolation
        img_tensor = preprocess_image(tmp_path)
        model = load_model(model_path)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            label = "Malignant" if pred_class else "Benign"
            confidence = probs[pred_class]

        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2%})")
        log_prediction(user, uploaded_file.name, label, confidence)

        # Display image and Grad-CAM
        st.image(tmp_path, caption="Uploaded Ultrasound Frame", width=224)
        cam = grad_cam(model, img_tensor, pred_class)
        import cv2
        orig = cv2.imread(tmp_path)
        orig = cv2.resize(orig, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
        st.image(overlay, caption="Grad-CAM Visualization", width=224)

    # Example analytics (stub, replace with real data)
    st.header("Analytics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**ROC Curve**")
        # plot_roc(y_true, y_probs)
        st.image("outputs/roc_curve.png")
    with col2:
        st.markdown("**Confusion Matrix**")
        # plot_confusion_matrix(y_true, y_pred)
        st.image("outputs/confusion_matrix.png")
    with col3:
        st.markdown("**Class Distribution**")
        # plot_class_distribution(y)
        st.image("outputs/class_distribution.png")

    st.info("All predictions and user actions are logged for monitoring and improvement.")

if __name__ == "__main__":
    main()
