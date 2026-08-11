# Vision_Guard
VisionGuard is an automated, real-time Computer Vision (CV) and Quality Assurance pipeline designed to solve a major real-world problem in AI systems: garbage in, garbage out.
In production AI systems (especially in fintech, insurance claim processing, or surveillance), passing blurry or degraded images into heavy deep learning models wastes compute and produces wrong results. 


VisionGuard acts as an intelligent dual-stage gatekeeper:
 1. Stage 1: Automated Image Quality Assessment (IQA)Before running expensive object detection models, VisionGuard evaluates the input image's physical quality:Blur Detection via OpenCV (Laplacian Variance): It calculates the variance of the Laplacian operator across the image. High variance indicates sharp, well-defined edges, while low variance indicates blur.Blur Measurement via scikit-image: It computes a normalized structural blur score across the intensity gradient.Brightness Scoring: It computes the mean pixel intensity across grayscale channels to check if an image is underexposed (too dark) or overexposed (washed out).Automated Decision Gate: If the blur variance falls below a set threshold (e.g., < 100), it flags the image as Fail (Blurry Image Detected), preventing poor data from corrupting downstream analytics.

 
 2. Stage 2: Deep Learning Object Detection (YOLOv8)Once an image passes or is evaluated, VisionGuard runs inference using YOLOv8 (You Only Look Once):Bounding Box Localization: Identifies precise spatial coordinates $(x_1, y_1, x_2, y_2)$ for detected objects.Multi-Class Classification: Labels objects (e.g., vehicles, pedestrians, electronics) with confidence scores.
Dynamic Thresholding: Allows the user to adjust the confidence threshold in real time to filter out low-probability detections.
Data Exporting: Converts bounding box vectors and detection classes into a clean, downloadable CSV dataset.


# 🎯 VisionGuard AI – Object Detection, ROI Cropping & Video Analytics

An interactive, production-grade Computer Vision dashboard built with **PyTorch**, **YOLOv8**, **OpenCV**, **scikit-image**, and **Streamlit**. 

VisionGuard AI serves as an automated dual-stage vision pipeline: evaluating image quality (blur and exposure) before performing real-time object detection, region-of-interest (ROI) tensor cropping, and frame-by-frame video processing.

🚀 **Live Web Application:** [Insert Your Streamlit App Link Here]

---

## 🌟 Key Features

- **Automated Image Quality Assessment (IQA):** Pre-evaluates input frames using OpenCV Laplacian variance and `scikit-image` blur metrics to filter out degraded or blurry images before inference.
- **Deep Learning Object Detection:** Powered by PyTorch-backed **YOLOv8** for real-time bounding box localization, multi-class labeling, and confidence scoring.
- **Dynamic ROI Cropping Engine:** Automatically extracts, crops, and renders individual detected objects in an interactive visual grid using OpenCV matrix slicing.
- **Real-Time Video & Webcam Analytics:** Supports frame-by-frame inference on uploaded MP4/AVI videos and live camera feeds.
- **Data Exporting:** Export detection coordinates, class tags, and confidence scores into structured CSV files with one click.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch, Ultralytics (YOLOv8)
- **Computer Vision & Processing:** OpenCV (`opencv-python-headless`), scikit-image, NumPy, PIL
- **Data Handling:** Pandas
- **Web Interface & Deployment:** Streamlit, Streamlit Community Cloud

---

## 📸 Usage & Modes

1. **Image Mode:**
   - Choose between synthetic sample scenes, custom image uploads (`.jpg`, `.png`), or camera snapshots.
   - Inspect physical quality metrics (Blur Variance, Brightness score).
   - Run YOLOv8 detection to generate bounding boxes and extract ROI crops.

2. **Video / Live Feed Mode:**
   - Upload an MP4 video or enable your webcam feed to perform frame-by-frame object detection and tracking.

---

