import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow import keras

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        r'C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\tb_model.h5'
    )
    return model

model = load_model()


st.set_page_config(
    page_title="TB Chest X-Ray Detector",
    layout="wide",
    page_icon="🫁"
)


pages = ["1. Introduction", "2. Image Detection", "3. Summary"]
selected_page = st.sidebar.radio("Go to", pages)


if selected_page == pages[0]:
    st.title("🫁 TB Chest X-Ray Detection with CNN")
    st.write("""
    This web app uses a **Convolutional Neural Network (CNN)** to detect **Tuberculosis (TB)** 
    in chest X-ray images. The model was trained on a dataset of labeled chest X-rays 
    (Normal and TB) and can classify new X-ray images in real time.
    
    ### Key Features
    - Deep Learning with CNN
    - Image upload and classification
    - Real-time results
    """)

elif selected_page == pages[1]:
    st.title("🔎 Upload Chest X-Ray Image")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)

   
        img_resized = img.resize((256, 256))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0 

       
        prediction = model.predict(img_array)[0][0]
        label = "🧫 TB Detected" if prediction > 0.5 else "✅ Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Confidence: **{confidence*100:.2f}%**")


elif selected_page == pages[2]:
    st.title("📊 Project Summary")
    
    st.markdown("""
    ### 📝 Summary

    This project demonstrates a **deep learning** model for binary image classification:

    - **Model Architecture:**  Conv2D + Pooling layers + Dropout layers + Flatten layer+ Fully connected Dense layers
    - **Dataset:** TB vs Normal Chest X-rays
    - **Validation Accuracy:** 90%
    """)
