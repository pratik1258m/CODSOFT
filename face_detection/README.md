# Task 5: Face Detection & Recognition

This directory contains my implementation of a computer vision face detection system using OpenCV's pre-trained frontal face Haar Cascade Classifier.

## 🚀 How to Run

### **1. Command Line Interface (CLI)**
Run the face detector on any portrait image using:
```bash
python face_detector.py <path_to_image> [output_file_name]
```
*(By default, the processed image highlighting faces with green boxes is saved as `detected_faces.jpg`.)*

### **2. Jupyter Notebook**
Explore the mathematical foundations of Viola-Jones, integral images, and cascade stage thresholds:
```bash
jupyter notebook face_detection_notebook.ipynb
```

---

## 🛠️ Implementation Details

### **Core Algorithm: Haar Cascade Classifier**
Haar Cascade is a machine learning object detection algorithm used to identify objects in images or video. It operates on the **Viola-Jones Framework**:
1. **Haar Features**: Extracts edge, line, and center-surround features based on pixel intensity differences.
2. **Integral Image**: Formulates cumulative sums to calculate features in constant $O(1)$ time, making it exceptionally fast.
3. **Grayscale Operation**: Converts images to single-channel grayscale to eliminate lighting/hue noise.
4. **Cascading Stages**: Rejects non-face areas immediately, focusing resources only on face-like candidates.

```text
 [Input Image] ──► [Grayscale Conversion] ──► [Cascade Stages (1..N)] ──► [Draw Green Rects]
                                                    │ (Rejected)
                                                    └──► [Skip Area]
```
