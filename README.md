# Vision_Guard
VisionGuard is an automated, real-time Computer Vision (CV) and Quality Assurance pipeline designed to solve a major real-world problem in AI systems: garbage in, garbage out.
In production AI systems (especially in fintech, insurance claim processing, or surveillance), passing blurry or degraded images into heavy deep learning models wastes compute and produces wrong results. 


VisionGuard acts as an intelligent dual-stage gatekeeper:
 1. Stage 1: Automated Image Quality Assessment (IQA)Before running expensive object detection models, VisionGuard evaluates the input image's physical quality:Blur Detection via OpenCV (Laplacian Variance): It calculates the variance of the Laplacian operator across the image. High variance indicates sharp, well-defined edges, while low variance indicates blur.Blur Measurement via scikit-image: It computes a normalized structural blur score across the intensity gradient.Brightness Scoring: It computes the mean pixel intensity across grayscale channels to check if an image is underexposed (too dark) or overexposed (washed out).Automated Decision Gate: If the blur variance falls below a set threshold (e.g., < 100), it flags the image as Fail (Blurry Image Detected), preventing poor data from corrupting downstream analytics.

 
 2. Stage 2: Deep Learning Object Detection (YOLOv8)Once an image passes or is evaluated, VisionGuard runs inference using YOLOv8 (You Only Look Once):Bounding Box Localization: Identifies precise spatial coordinates $(x_1, y_1, x_2, y_2)$ for detected objects.Multi-Class Classification: Labels objects (e.g., vehicles, pedestrians, electronics) with confidence scores.
Dynamic Thresholding: Allows the user to adjust the confidence threshold in real time to filter out low-probability detections.
Data Exporting: Converts bounding box vectors and detection classes into a clean, downloadable CSV dataset.
