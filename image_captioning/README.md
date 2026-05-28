# Task 3: Image Captioning AI (ResNet + LSTM & BLIP)

This directory contains the implementations of the Image Captioning AI project, combining **Computer Vision (CNN)** and **Natural Language Processing (NLP)**.

## 🚀 How to Run

### **1. Transformer-Based Caption Generator (Modern BLIP)**
Run the state-of-the-art Hugging Face BLIP pre-trained transformer model locally to caption any image:
```bash
python caption_generator.py <path_to_your_image>
```

### **2. Jupyter Notebook**
Explore both the classic CNN-LSTM model implementation and the Hugging Face BLIP inference pipeline step-by-step:
```bash
jupyter notebook image_captioning_notebook.ipynb
```

---

## 🛠️ Implementation Details

We provide two separate implementations to showcase academic understanding and practical excellence:

### **1. Academic Architecture: ResNet-50 + LSTM Encoder-Decoder**
Defined in `model_architecture.py`, this showcases the classic CNN-RNN hybrid model designed for sequence-to-sequence translation:
- **Encoder (CNN)**: Uses a pre-trained **ResNet-50** network. The final fully connected classification layer is discarded, and spatial features are projected into a target embedding size (e.g., 256) through a Linear layer.
- **Decoder (RNN)**: A recurrent sequence generator based on a **Long Short-Term Memory (LSTM)** network. It accepts the extracted image feature vector as its initial state and recursively predicts the next word in the sequence.

```text
 [Image] ──► [ ResNet-50 CNN ] ──► [ Visual Embedding ] ──┐
                                                           ▼
 [Target Token] ──► [ Word Embedding ] ────────────────► [ LSTM ] ──► [ Probability Dist ]
```

### **2. Modern Production: BLIP Transformer**
Implemented in `caption_generator.py`, this uses **BLIP (Bootstrapped Language-Image Pre-training)**:
- **Vision-Language Pre-training (VLP)**: A unified model architecture that achieves state-of-the-art results by leveraging multi-modal attention layers.
- It is significantly more robust than traditional LSTM decoders, capturing fine-grained relationships, spatial objects, and actions with rich adjective descriptions.
