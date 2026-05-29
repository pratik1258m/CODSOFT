# CODSOFT Artificial Intelligence Internship Projects

Welcome to my **CODSOFT Artificial Intelligence Internship** repository! This repository contains the complete implementation of all **5 Tasks** in the AI curriculum. Each project is designed to showcase core AI concepts, from rule-based systems to game tree search, computer vision, sequence transformers, recommender math, and object detection.

---

## 🌟 Projects Overview

### **Task 1: Chatbot with Rule-Based Responses**
- **Description**: A conversational AI agent that processes user inputs using pattern-matching (Regular Expressions) to provide dynamic, pre-defined responses.
- **Implementation**: Standalone CLI script (`rule_based_chatbot.py`), Jupyter Notebook (`chatbot_notebook.ipynb`), and the Streamlit dashboard tab.

### **Task 2: Tic-Tac-Toe AI**
- **Description**: An unbeatable AI agent that plays the classic game of Tic-Tac-Toe against a human player using the Minimax Search Algorithm with Alpha-Beta Pruning.
- **Implementation**: Standalone CLI game (`tictactoe_ai.py`), Jupyter Notebook (`tictactoe_notebook.ipynb`), and interactive game grid on the Streamlit dashboard tab.

### **Task 3: Image Captioning AI**
- **Description**: An advanced AI that generates textual descriptions for uploaded images, utilizing a pre-trained **BLIP Transformer model** for inference, alongside a custom PyTorch **ResNet-50 + LSTM encoder-decoder architecture** from scratch.
- **Implementation**: Standalone BLIP generator (`caption_generator.py`), PyTorch source code (`model_architecture.py`), Jupyter Notebook (`image_captioning_notebook.ipynb`), and Streamlit tab.

### **Task 4: Movie Recommendation System**
- **Description**: A movie recommendation engine implementing **Content-Based Filtering** (TF-IDF vectorizer + Cosine Similarity) and user-based **Collaborative Filtering** (normalized user ratings Pearson correlations).
- **Implementation**: Standalone recommender CLI (`movie_recommender.py`), Jupyter Notebook (`recommendation_notebook.ipynb`), and Streamlit tab.

### **Task 5: Face Detection & Recognition**
- **Description**: A computer vision application that detects faces in digital images using OpenCV's Viola-Jones Haar Cascade Frontal Face Classifier.
- **Implementation**: Standalone detector (`face_detector.py`), Jupyter Notebook (`face_detection_notebook.ipynb`), and Streamlit tab.

---

## 📂 Directory Structure

```text
CODSOFT/
├── README.md                      # Master repository documentation
├── requirements.txt               # Unified project dependencies
├── app.py                         # Single-Page Streamlit Web App (Hosts all 5 tasks!)
│
├── chatbot/                       # TASK 1: Chatbot
│   ├── README.md                  # Chatbot setup & run guide
│   ├── rule_based_chatbot.py      # Standalone CLI chat application
│   └── chatbot_notebook.ipynb     # Step-by-step Jupyter Notebook
│
├── tictactoe/                     # TASK 2: Tic-Tac-Toe AI
│   ├── README.md                  # Game launch guide
│   ├── tictactoe_ai.py            # Standalone CLI game with Minimax AI
│   └── tictactoe_notebook.ipynb   # Theoretical/Game Tree Jupyter Notebook
│
├── image_captioning/              # TASK 3: Image Captioning AI
│   ├── README.md                  # Model details and pipeline guide
│   ├── caption_generator.py       # Standalone BLIP transformer inference script
│   ├── model_architecture.py      # Custom ResNet-50 + LSTM encoder-decoder source code
│   └── image_captioning_notebook.ipynb # Deep learning training tutorial notebook
│
├── recommendation/                # TASK 4: Movie Recommender
│   ├── README.md                  # Math details and run guide
│   ├── movie_recommender.py       # Standalone movie recommender script
│   └── recommendation_notebook.ipynb # Linear algebra Jupyter Notebook
│
└── face_detection/                # TASK 5: Face Detector
    ├── README.md                  # Viola-Jones guide
    ├── face_detector.py           # Standalone face detector script
    └── face_detection_notebook.ipynb # Computer vision Jupyter Notebook
```

---

## 🛠️ Installation and Setup

### **Prerequisites**
Make sure you have **Python 3.8+** installed on your system.

### **1. Clone the Repository**
```bash
git clone https://github.com/pratik1258m/CODSOFT.git
cd CODSOFT
```

### **2. Set up a Virtual Environment (Optional)**
```bash
# On Mac/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Projects

### 🖥️ **Unified Web Dashboard (Recommended)**
Open our beautiful interactive portfolio dashboard in your browser to test all 5 tasks at once:
```bash
streamlit run app.py
```

### 💻 **Command-Line Interfaces (CLI)**

You can also run each project individually in the terminal:

#### **Task 1: Chatbot**
```bash
python chatbot/rule_based_chatbot.py
```

#### **Task 2: Tic-Tac-Toe AI**
```bash
python tictactoe/tictactoe_ai.py
```

#### **Task 3: Image Captioning**
```bash
python image_captioning/caption_generator.py
```

#### **Task 4: Movie Recommender**
```bash
python recommendation/movie_recommender.py
```

#### **Task 5: Face Detector**
```bash
python face_detection/face_detector.py
```

---

## 👨‍🎓 Internship Information
- **Intern Name**: Pratik
- **Internship Domain**: Artificial Intelligence
- **Duration**: May 25, 2026 – June 25, 2026
- **Organization**: CODSOFT
