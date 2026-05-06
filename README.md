# 🧠 Combined Non-Manual Feature Extraction and Textual Integration for ISL

## 📌 Overview
This project focuses on extracting and integrating **non-manual features** of Indian Sign Language (ISL), such as **facial expressions, head movements, and eye gaze**, and converting them into a **unified textual description** in real time.

Unlike many ISL systems that focus only on hand gestures, this project emphasizes the **non-manual components**, which are essential for conveying meaning, emotion, and context in sign language.

---

## 🎯 Objectives
- Detect facial expressions using geometric landmark analysis  
- Track head movements (nodding and shaking)  
- Estimate eye gaze direction (left, right, center)  
- Apply temporal smoothing for stable predictions  
- Integrate all features into a single descriptive sentence  

---

## 🛠️ Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- Classical Computer Vision Techniques:
  - Motion tracking  
  - Geometric transformations  
  - Distance-based feature extraction  

---

## 🧱 System Architecture
Video Input (Webcam)
↓
Face Detection (MediaPipe)
↓
Feature Extraction
├── Facial Expression
├── Head Movement
└── Eye Gaze
↓
Temporal Smoothing
↓
Text Integration Module
↓
Final Output Sentence

---

## ⚙️ Features Implemented

### 😊 Facial Expression Detection
- Happy  
- Surprised  
- Neutral  

### 🤕 Head Movement Detection
- Nodding (Yes)  
- Shaking (No)  
- Stable  

### 👀 Eye Gaze Detection
- Looking Left  
- Looking Right  
- Center  

---

## 🧾 Sample Output
Expression: Happy
Head: Nodding
Gaze: Looking Left

→ Person is nodding with a happy expression while looking left

---

## 📁 Project Structure
ISL_PROJECT/
│── main.py
│── config.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── modules/
│ ├── video_capture.py
│ ├── face_detection.py
│ ├── expression.py
│ ├── head_movement.py
│ ├── gaze.py
│ └── integration.py
│
├── utils/
│ ├── math_utils.py
│ └── smoothing.py

---

## 🎓 Conclusion
This project demonstrates a real-time system for extracting and integrating non-manual features of ISL. It highlights the importance of facial expressions, head movement, and gaze in understanding sign language and provides a foundation for building a complete ISL translation system.

---

## 👨‍💻 Author
**Nitin Chahar**

---

## 📜 License
This project is for academic purposes only.