import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from PIL import Image
from ultralytics import YOLO
from skimage.measure import blur_effect

st.set_page_config(page_title="VisionGuard AI - Object Detection & Quality Engine", layout="wide")

st.title("🎯 VisionGuard AI – Detection, ROI Cropping & Video Analytics")
st.write("An interactive Computer Vision pipeline featuring YOLOv8 detection, crop extraction (ROI), quality filtering, and video processing.")

# Load YOLOv8 Model
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

# Helper: Generate Sample Synthetic Image
def generate_sample_scene():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 220
    cv2.rectangle(img, (0, 250), (600, 400), (80, 80, 80), -1)
    cv2.rectangle(img, (150, 200), (450, 300), (0, 0, 200), -1)
    cv2.circle(img, (220, 300), 30, (0, 0, 0), -1)
    cv2.circle(img, (380, 300), 30, (0, 0, 0), -1)
    cv2.putText(img, "SAMPLE VEHICLE SCENE", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img

# Sidebar Settings
st.sidebar.header("1. Input Mode")
media_type = st.sidebar.radio("Select Media Type:", ("Image Mode", "Video / Live Feed Mode"))

st.sidebar.header("2. Detection Parameters")
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# ==============================================================================
# MODE 1: IMAGE PROCESSING & ROI CROPPING
# ==============================================================================
if media_type == "Image Mode":
    input_option = st.sidebar.radio("Image Source:", ("Use Sample Scene", "Upload Image (JPG/PNG)", "Take Photo"))
    
    image = None
    if input_option == "Use Sample Scene":
        image = generate_sample_scene()
    elif input_option == "Upload Image (JPG/PNG)":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
    else:
        captured_file = st.camera_input("Take Photo")
        if captured_file:
            file_bytes = np.asarray(bytearray(captured_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR", use_container_width=True)

        # Image Quality Assessment
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = blur_effect(gray)
        avg_brightness = np.mean(gray)
        quality_status = "Pass (Clear)" if laplacian_var > 100 else "Fail (Blurry Image Detected)"

        with col2:
            st.subheader("Quality Assessment (OpenCV & scikit-image)")
            st.metric(label="Quality Assessment Status", value=quality_status)
            st.write(f"• **Blur Variance (Laplacian):** `{laplacian_var:.2f}` (Threshold: > 100)")
            st.write(f"• **Blur Score:** `{blur_score:.4f}`")
            st.write(f"• **Mean Brightness:** `{avg_brightness:.2f} / 255`")

        st.markdown("---")

        if st.button("Run YOLOv8 & Extract ROI Crops"):
            start_time = time.time()
            results = model.predict(image, conf=conf_threshold)
            inference_time = (time.time() - start_time) * 1000

            annotated_frame = results[0].plot()
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader(f"YOLOv8 Output (Inference: {inference_time:.1f}ms)")
                st.image(annotated_frame, channels="BGR", use_container_width=True)

            detections = []
            roi_crops = []

            # Extract Bounding Boxes and Crop Images
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Crop Region of Interest (ROI) from original image using OpenCV
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    roi_crops.append((f"{class_name} #{i+1} ({confidence*100:.1f}%)", crop))

                detections.append({
                    "Object #": i + 1,
                    "Class": class_name,
                    "Confidence": f"{confidence * 100:.2f}%",
                    "Bounding Box [x1, y1, x2, y2]": [x1, y1, x2, y2]
                })

            with res_col2:
                st.subheader("Detections Summary")
                if detections:
                    df = pd.DataFrame(detections)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No objects detected above threshold.")

            # Display Extracted ROI Crops in Grid
            if roi_crops:
                st.markdown("---")
                st.subheader("🔍 Extracted Regions of Interest (ROI Crops)")
                st.write("Cropped object bounds extracted directly from tensor coordinates using OpenCV matrix slicing:")
                
                crop_cols = st.columns(min(len(roi_crops), 4))
                for idx, (label, crop_img) in enumerate(roi_crops):
                    col_idx = idx % 4
                    with crop_cols[col_idx]:
                        st.caption(label)
                        st.image(crop_img, channels="BGR", use_container_width=True)

# ==============================================================================
# MODE 2: VIDEO & LIVE WEBCAM PROCESSING
# ==============================================================================
else:
    st.subheader("📹 Real-Time Video Object Tracking")
    video_source = st.radio("Select Video Input:", ("Upload MP4 Video", "Live Camera Feed"))

    video_bytes = None
    
    if video_source == "Upload MP4 Video":
        uploaded_video = st.file_uploader("Upload an MP4 video file", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            st.write("Processing video stream frame by frame...")
            st_frame = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO prediction per frame
                results = model.predict(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()
                
                # Render live frame
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            cap.release()

    else:
        run_webcam = st.checkbox("Start Live Camera Feed")
        if run_webcam:
            cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera.")
                    break
                
                results = model.predict(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            cap.release()
