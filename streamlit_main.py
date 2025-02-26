# ┌─────────────────────────────┐
# │ Required Library Imports    │
# └─────────────────────────────┘
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import streamlit as st
from PIL import Image
import time

# ┌─────────────────────────────┐
# │ Application Configuration   │
# └─────────────────────────────┘

st.set_page_config(
    page_title="Animal Image Classification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ┌─────────────────────────────┐
# │ Styling and UI Elements     │
# └─────────────────────────────┘

st.markdown("""
    <style>
        /* Professional Interface Styling */
        .main {
            padding: 2rem;
            background: #F8F9FA;
        }
        
        /* Standard Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 4px;
            height: 2.5em;
            background: #0066CC;
            color: white;
            transition: all 0.2s ease;
            border: none;
            font-weight: 500;
        }
        .stButton>button:hover {
            background: #0056b3;
        }
        
        /* Upload Section */
        .upload-section {
            text-align: center;
            padding: 1.5rem;
            background: #FFFFFF;
            border-radius: 6px;
            margin-top: 1.5rem;
            border: 1px solid #E9ECEF;
        }
        
        /* Title and Subtitle */
        .main-title {
            text-align: center;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            padding: 0.75rem;
            color: #212529;
        }
        
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #495057;
            font-weight: 400;
        }
        
        /* Prediction Results */
        .prediction-box {
            padding: 1.25rem;
            border-radius: 6px;
            margin: 1rem 0;
            background: #FFFFFF;
            border: 1px solid #E9ECEF;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background-color: #0066CC;
        }
        
        /* Confidence Display */
        .confidence-meter {
            margin: 1rem 0;
            padding: 1.25rem;
            background: #FFFFFF;
            border-radius: 4px;
            border: 1px solid #E9ECEF;
        }
        
        /* Status Messages */
        .stSuccess {
            background-color: rgba(25, 135, 84, 0.1);
            border: 1px solid #198754;
        }
        .stInfo {
            background-color: rgba(13, 110, 253, 0.1);
            border: 1px solid #0d6efd;
        }
        .stWarning {
            background-color: rgba(255, 193, 7, 0.1);
            border: 1px solid #ffc107;
        }
        
        /* Image Display */
        .stImage {
            border-radius: 4px;
            border: 1px solid #E9ECEF;
            padding: 2px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #6C757D;
            padding: 0.75rem;
            margin-top: 1.5rem;
            border-top: 1px solid #E9ECEF;
        }
    </style>
""", unsafe_allow_html=True)

# ┌─────────────────────────────┐
# │ Path Configuration          │
# └─────────────────────────────┘

model_path = ("models/animal_model.h5")
animals_dir = ("data/animals")

# ┌─────────────────────────────┐
# │ Core Functions              │
# └─────────────────────────────┘

def get_class_names(animals_dir):   
    if not os.path.exists(animals_dir):
        raise FileNotFoundError(f"Error: The animal classification directory was not found at the specified location: {animals_dir}.")
    return sorted([d for d in os.listdir(animals_dir) if os.path.isdir(os.path.join(animals_dir, d))])

# ┌─────────────────────────────┐
# │ Model Prediction Function   │
# └─────────────────────────────┘

def predict_image(model_path, img, class_names, img_size=(128, 128)):
    model = load_model(model_path)
    img = cv2.resize(img, img_size) / 255.0  
    img = np.expand_dims(img, axis=0)        
    prediction = model.predict(img)[0]       
    predicted_class = class_names[np.argmax(prediction)]  
    confidence = np.max(prediction)           
    return predicted_class, confidence

# ┌─────────────────────────────┐
# │ Main UI Components          │
# └─────────────────────────────┘

st.markdown("<h4 class='main-title'>Animal Image Classification System</h4>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Computer Vision for Animal Species Identification</p>", unsafe_allow_html=True)

# ┌─────────────────────────────┐
# │ Sidebar Configuration       │
# └─────────────────────────────┘
with st.sidebar:
    st.markdown("## System Information")
    st.info("""
        This application utilizes a neural network model to analyze and classify animal images.
        
        Please upload a high-quality image of an animal for accurate classification results.
    """)
    try:
        class_names = get_class_names(animals_dir)
    except FileNotFoundError as e:
        st.error(str(e))

# ┌─────────────────────────────┐
# │ Main Content Layout         │
# └─────────────────────────────┘

col1, col2 = st.columns([3, 1]) 
with col1:  
    with st.container():
        st.markdown("<div class=''>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Please select an animal image file for analysis", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)   
    if uploaded_file is not None:
        try:
            # ┌─────────────────────────┐
            # │ Image Preprocessing     │
            # └─────────────────────────┘

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            max_width = 400  
            height, width = img_rgb.shape[:2]            
            if width > max_width:
                aspect_ratio = height / width
                new_width = max_width
                new_height = int(max_width * aspect_ratio)
                display_img = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                display_img = img_rgb           
            st.image(display_img, caption="Submitted Image", use_column_width=False)
            
            # ┌─────────────────────────┐
            # │ Analysis Progress Bar   │
            # └─────────────────────────┘

            with st.spinner("Processing image analysis..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)                
                predicted_class, confidence = predict_image(model_path, img, class_names)        

                # ┌─────────────────────────┐
                # │ Results Display         │
                # └─────────────────────────┘

                st.markdown("### Analysis Results")                               
                col_pred, col_conf = st.columns(2)                
                with col_pred:
                    st.markdown(f"**Identified Species:** {predicted_class.title()}")                
                with col_conf:
                    confidence_percentage = float(confidence) * 100
                    st.markdown("**Confidence Level:** {:.1f}%".format(confidence_percentage))
                st.progress(float(confidence))
                if confidence > 0.9:
                    st.success("Classification completed with high confidence.")
                elif confidence > 0.7:
                    st.info("Classification completed with moderate confidence.")
                else:
                    st.warning("Classification completed with low confidence. Consider submitting a clearer image for more accurate results.")
                
        except Exception as e:
            st.error(f"An error occurred during image processing: {str(e)}")
    else:
        # ┌─────────────────────────┐
        # │ Instructions Display    │
        # └─────────────────────────┘

        st.markdown("""
        ### Usage Instructions
        1. Select an animal image using the file selection tool above
        2. The system will automatically process and analyze the image
        3. Review the classification results and confidence assessment
        
        **Note:** For optimal results, please provide clear, well-lit images with the animal as the primary subject
    """)

# ┌─────────────────────────────┐
# │ Footer Section              │
# └─────────────────────────────┘
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6C757D; padding: 0.75rem;'>
        Developed by Curious.PM
    </div>
""", unsafe_allow_html=True)