import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json

st.set_page_config(
    page_title="Malaria Cell Detection",
    page_icon="icon.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    body { background-color: #f4f4f9; }
    .main-header { font-size: 2.8rem; color: #1a3c34; text-align: center; font-family: 'Georgia', serif; margin-bottom: 1.5rem; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1); }
    .subheader { font-size: 1.5rem; color: #2e5a50; font-family: 'Georgia', serif; margin-bottom: 1rem; }
    .prediction-box { padding: 1.2rem; border-radius: 10px; text-align: center; font-size: 1.2rem; font-weight: 600; margin: 1rem 0; border: 2px solid; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s ease-in-out; }
    .prediction-box:hover { transform: translateY(-5px); }
    .infected { background-color: #ffe6e6; color: #c0392b; border-color: #c0392b; }
    .uninfected { background-color: #e6f3e6; color: #27ae60; border-color: #27ae60; }
    .low-confidence { background-color: #fff8e1; color: #d4a017; border-color: #d4a017; }
    .stButton>button { background-color: #1a3c34; color: white; border-radius: 8px; padding: 0.5rem 1.5rem; font-family: 'Georgia', serif; font-size: 1rem; transition: background-color 0.3s ease; }
    .stButton>button:hover { background-color: #2e5a50; }
    .info-box { background-color: #ffffff; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-family: 'Arial', sans-serif; font-size: 0.9rem; color: #34495e; }
    .footer { text-align: center; font-size: 0.85rem; color: #7f8c8d; margin-top: 2rem; font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Malaria Cell Detection</h1>', unsafe_allow_html=True)

def load_threshold():
    threshold_file = "threshold.json"
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            data = json.load(f)
            return float(data.get("optimal_threshold", 0.5))
    return 0.5 

OPTIMAL_THRESHOLD = load_threshold()

@st.cache_resource
def load_model():
    model_path = "best_malaria_model.keras"
    
    # If model doesn't exist locally, download from Google Drive
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=1FlW5oVtmjm9NA51aak9hLZt_luK9egDW"
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = img_array.astype(np.float32) / 255.0
    mean = np.mean(img_array)
    stddev = np.std(img_array)
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(img_array.size))
    img_array = (img_array - mean) / adjusted_stddev
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.markdown("""
        <h2 class="subheader">Upload Blood Cell Image</h2>
        <div class="info-box">
         Upload a microscopic blood cell image for automated malaria screening. 
         Test with sample images from the 
         <a href="https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria" target="_blank">
         National Institutes of Health (NIH)</a> - an official United States government medical research resource.
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=['png','jpg','jpeg'], help="Upload a blood cell image for analysis")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        
        if st.button("Analyze Image"):
            try:
                with st.spinner("Loading model..."):
                    model = load_model()
                with st.spinner("Processing image..."):
                    processed_image = preprocess_image(image)
                with st.spinner("Analyzing..."):
                    prediction = float(model.predict(processed_image, verbose=0)[0][0])

                st.markdown('<h2 class="subheader">Analysis Results</h2>', unsafe_allow_html=True)
                
                threshold = OPTIMAL_THRESHOLD
                if prediction < threshold:
                    result = "PARASITIZED"
                    confidence = (threshold - prediction) / threshold
                    css_class = "infected"
                    message = "Malaria parasites detected. Immediate medical consultation recommended."
                else:
                    result = "UNINFECTED"
                    confidence = (prediction - threshold) / (1 - threshold)
                    css_class = "uninfected"
                    message = "No malaria parasites detected."

                confidence = float(confidence)
                
                if confidence < 0.70:
                    css_class = "low-confidence"
                    result_display = f"{result}"
                    message = "Low confidence prediction. Professional laboratory analysis recommended."
                else:
                    result_display = result
                
                st.markdown(f"""
                <div class="prediction-box {css_class}">
                    {result_display}<br>
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)

                if confidence < 0.70:
                    st.warning(message)
                elif result == "PARASITIZED":
                    st.error(message)
                else:
                    st.success(message)
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p> This AI tool assists in rapid malaria screening using blood cell images. For educational and research purposes only.</p>
        <p><strong>Developed by:</strong> Tasneem Bin Mahmood | 
        <a href="https://github.com/adittomahmood" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/adittomahmood/" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
