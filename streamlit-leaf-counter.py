import streamlit as st
import requests
import os
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# Streamlit page config
st.set_page_config(
    page_title="Zucchini Leaf Counter",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to download the file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"

    with requests.get(URL, stream=True) as r:
        if r.status_code == 200:
            with open(destination, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Model downloaded successfully to {destination}")
        else:
            st.error(f"Error downloading model: {r.status_code}")

# Function to load model
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return YOLO(model_path)
    st.error(f"Model file not found: {model_path}")
    return None

# Image preprocessing
def preprocess_image(image):
    img_array = np.array(image)
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# Function to run inference
def run_inference(model, image, conf_thresh, iou_thresh):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.write(f"💻 Using device: {device}")
        results = model.predict(
            image,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=640,
            device=device,
            verbose=False
        )
        return results[0] if results else None
    except Exception as e:
        st.error(f"❌ Error during inference: {str(e)}")
        return None

# Counting leaves
def count_leaves(result):
    if result and hasattr(result, 'boxes') and result.boxes is not None:
        confs = result.boxes.conf.cpu().numpy().tolist() if hasattr(result.boxes, 'conf') else []
        return len(result.boxes), confs
    return 0, []

# Function to create annotated image
def create_annotated_image(image, result, show_conf=True):
    img_array = np.array(image)
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    height, width = img_array.shape[:2]
    num_leaves, confidences = count_leaves(result)

    if result and hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.detach().cpu().numpy().copy()
        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (width, height))
            mask_color = (0, 255, 0)
            alpha = 0.3
            for c in range(3):
                img_array[:, :, c] = np.where(
                    mask_resized > 0,
                    (1 - alpha) * img_array[:, :, c] + alpha * mask_color[c],
                    img_array[:, :, c]
                ).astype(np.uint8)

    if hasattr(result.boxes, 'xyxy'):
        boxes = result.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if show_conf and i < len(confidences):
                label = f"leaf {int(confidences[i] * 100)}%"
                cv2.putText(img_array, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if num_leaves > 0:
        label = f"{num_leaves} leaves detected"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        x, y = (width - text_size[0]) // 2, 50
        cv2.rectangle(img_array, (x - 10, y - 35), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(img_array, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return img_array

def main():
    st.markdown('<h1 class="main-header">🌿 Zucchini Leaf Counter</h1>', unsafe_allow_html=True)
    
    # Add subtitle with description
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%); border-radius: 15px; border: 1px solid #e0f2e0;">
        <p style="font-size: 1.2rem; color: #666; margin: 0; font-weight: 500;">
            🔬 Advanced leaf detection and counting using YOLOv8/YOLOv11 instance segmentation
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model configuration section
        st.markdown("#### 🤖 Model Settings")
        model_file_id = st.text_input("Google Drive File ID", "1YcZxsm5g1Cw3zUxHziB9zOrwBKb7ZBW7")
        
        # Detection parameters section
        st.markdown("#### 🎯 Detection Parameters")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05, 
                                        help="Minimum confidence score for detections")
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.7, 0.05,
                                help="Intersection over Union threshold for non-maximum suppression")
        
        # Display options section
        st.markdown("#### 🎨 Display Options")
        show_confidence = st.checkbox("Show Confidence Scores", True, 
                                    help="Display confidence scores on detected leaves")
        
        # Add device info
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        device_color = '#32CD32' if device == 'CUDA' else '#FF6B6B'
        st.markdown(f"""
        <div style="margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%); border-radius: 10px; border: 1px solid #e0f2e0;">
            <p style="margin: 0; text-align: center;">
                <strong>Device:</strong> <span style="color: {device_color}; font-weight: bold;">{device}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Path to save the model
    model_file_path = "./best.pt"

    # If the model path is not specified or missing, download it from Google Drive
    if not os.path.exists(model_file_path):
        st.warning("Model not found locally. Downloading from Google Drive...")
        download_file_from_google_drive(model_file_id, model_file_path)

    model = load_model(model_file_path)
    if not model:
        return

    # Enhanced upload section
    st.markdown('<h3 style="color: black; font-weight: bold;">📤 Upload Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_file:
        # Create two columns for original and processed images
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 style="color: black; font-weight: bold;">📷 Original Image</h4>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing image..."):
            processed_image = preprocess_image(image)
            result = run_inference(model, processed_image, confidence_threshold, iou_threshold)

            if result is None:
                st.error("❌ Model did not return any results. Check your image or model configuration.")
                return

            num_leaves, confidences = count_leaves(result)
            annotated_img = create_annotated_image(image, result, show_confidence)

        with col2:
            st.markdown('<h4 style="color: black; font-weight: bold;">🎯 Detection Results</h4>', unsafe_allow_html=True)
            st.image(annotated_img, caption="Detection Results", use_container_width=True)

        # Results summary section
        st.markdown('<div class="results-header">📊 Detection Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'''
            <div class="metric-container">
                <div class="leaf-count">{num_leaves}</div>
                <div class="confidence-text">Leaves Detected</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            avg_conf = np.mean(confidences) if confidences else 0
            st.markdown(f'''
            <div class="metric-container">
                <div class="leaf-count">{avg_conf:.1%}</div>
                <div class="confidence-text">Avg Confidence</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            max_conf = max(confidences) if confidences else 0
            st.markdown(f'''
            <div class="metric-container">
                <div class="leaf-count">{max_conf:.1%}</div>
                <div class="confidence-text">Max Confidence</div>
            </div>
            ''', unsafe_allow_html=True)

        # Detailed results section
        if show_confidence and num_leaves > 0:
            st.markdown('<h3 style="color: black; font-weight: bold;">📋 Detailed Detection Results</h3>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Create enhanced dataframe
            df = pd.DataFrame({
                'Leaf #': list(range(1, num_leaves + 1)),
                'Confidence': [f"{c:.1%}" for c in confidences],
                'Confidence Score': [f"{c:.3f}" for c in confidences]
            })
            
            # Display dataframe with custom styling
            st.dataframe(
                df, 
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing before download section
            
            # Download and statistics section with improved layout
            download_col, stats_col, spacer_col = st.columns([3, 2, 1])
            
            with download_col:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV Report", 
                    data=csv, 
                    file_name=f"leaf_detection_results_{uploaded_file.name.split('.')[0]}.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
            
            with stats_col:
                # Add summary statistics with better visibility
                if confidences:
                    st.markdown(f"""
                    <div style="padding: 1.5rem; background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%); border-radius: 12px; border: 2px solid #2E8B57; box-shadow: 0 4px 15px rgba(46, 139, 87, 0.15);">
                        <p style="margin: 0 0 0.8rem 0; text-align: center; font-weight: 700; font-size: 1rem; color: #2E8B57;">
                            📊 Detection Quality
                        </p>
                        <div style="text-align: center; line-height: 1.6;">
                            <div style="margin-bottom: 0.4rem;">
                                <span style="color: #666; font-weight: 500; font-size: 0.9rem;">Min:</span> 
                                <span style="color: #2E8B57; font-weight: 700; font-size: 1rem;">{min(confidences):.1%}</span>
                            </div>
                            <div style="margin-bottom: 0.4rem;">
                                <span style="color: #666; font-weight: 500; font-size: 0.9rem;">Max:</span> 
                                <span style="color: #2E8B57; font-weight: 700; font-size: 1rem;">{max(confidences):.1%}</span>
                            </div>
                            <div>
                                <span style="color: #666; font-weight: 500; font-size: 0.9rem;">Std Dev:</span> 
                                <span style="color: #2E8B57; font-weight: 700; font-size: 1rem;">{np.std(confidences):.1%}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add bottom spacing
            st.markdown("<br><br>", unsafe_allow_html=True)
    else:
        # Enhanced empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem; margin: 2rem 0; border: 2px dashed #2E8B57; border-radius: 15px; background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);">
            <h3 style="color: #2E8B57; margin-bottom: 1rem;">🌱 Ready to Count Leaves!</h3>
            <p style="color: #666; font-size: 1.1rem; margin: 0;">
                Upload an image of zucchini leaves to get started with AI-powered detection and counting.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
