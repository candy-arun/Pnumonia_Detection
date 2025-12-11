# 🫁 Pneumonia Detection from Chest X-Rays using MobileNetV2

An end-to-end deep learning project that detects **Pneumonia** from chest X-ray images using a fine-tuned **MobileNetV2** model.  
This repository includes:

- A **Streamlit Web UI (`app.py`)** for real-time predictions  
- A **single image prediction notebook (`single_prediction.ipynb`)**  
- A trained MobileNetV2 model (`Trained_Model.keras`)  

---

## 📌 Project Overview

Pneumonia is a critical respiratory infection that requires fast and reliable diagnosis. X-ray interpretation by radiologists is effective but time-consuming and may vary between experts.  
This project automates the classification of chest X-rays into **Normal** or **Pneumonia**, providing a fast, accurate, AI-powered diagnostic support tool.

---

## 🧠 Model Architecture (MobileNetV2)

This project uses **MobileNetV2**, a lightweight and efficient convolutional neural network, as the base model.

### Why MobileNetV2?
- High accuracy with fewer parameters  
- Very fast inference  
- Great performance for medical images  
- Perfect for real-time deployments like Streamlit apps  

The base MobileNetV2 network is fine-tuned with custom dense layers for binary classification (Normal vs Pneumonia).

---

## 📈 Model Performance

Based on training and evaluation:

### **Training Metrics**
- **Training Accuracy:** 95.37%  
- **Training Loss:** 0.1118  

### **Testing Metrics**
- **Test Accuracy:** 85.25%  
- **Test Loss:** 0.4249  

### **Validation Accuracy (per epoch)**
Ranged roughly from **75% to 88%** over 10 epochs.

> Note: Results may vary depending on training hardware, hyperparameters, and dataset distribution.

---

## 🔍 Dataset

Dataset used: **Chest X-Ray Pneumonia Dataset (Kaggle)**  
🔗 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Dataset contains:

- **NORMAL**
- **PNEUMONIA** (viral/bacterial)

Folder structure:

chest_xray/
├── train/
├── val/
└── test/


---

## 🛠️ Features

### ✔ Streamlit Web App (`app.py`)
- Drag-and-drop or file upload  
- Real-time pneumonia prediction  
- Clean and modern UI  
- Uses your trained MobileNetV2 model  

### ✔ Single Image Prediction (`single_prediction.ipynb`)
- Provide a file path to any X-ray image  
- Notebook predicts and displays the result  

### ✔ MobileNetV2 Transfer Learning
- Faster training  
- Better accuracy  
- Suitable for deployment on low-resource systems  

---

## 🚀 Technologies Used

- Python  
- TensorFlow / Keras  
- MobileNetV2  
- NumPy, Pandas  
- Matplotlib  
- Streamlit  
- Pillow  

---

## ▶️ How to Run the Project

### **1️⃣ Install Dependencies**

pip install tensorflow streamlit numpy pillow


---

## 🛠️ Features

### ✔ Streamlit Web App (`app.py`)
- Drag-and-drop or file upload  
- Real-time pneumonia prediction  
- Clean and modern UI  
- Uses your trained MobileNetV2 model  

### ✔ Single Image Prediction (`single_prediction.ipynb`)
- Provide a file path to any X-ray image  
- Notebook predicts and displays the result  

### ✔ MobileNetV2 Transfer Learning
- Faster training  
- Better accuracy  
- Suitable for deployment on low-resource systems  

---

## 🚀 Technologies Used

- Python  
- TensorFlow / Keras  
- MobileNetV2  
- NumPy, Pandas  
- Matplotlib  
- Streamlit  
- Pillow  

---

## ▶️ How to Run the Project

### **1️⃣ Install Dependencies**

pip install tensorflow streamlit numpy pillow

2️⃣ Run the Streamlit App

streamlit run app.py

3️⃣ Predict Single Image via Notebook

Open single_prediction.ipynb, set your image path, and run all cells.

📦 Repository Structure

📁 Pneumonia-Detection
│── app.py                    
│── single_prediction.ipynb     
│── Trained_Model.keras        
│── README.md                   
│── .gitignore                
│
└── (ignored - not uploaded)
    ├── chest_xray/             
    └── model.ipynb   


🧪 Dataset Download Instructions

Download dataset from Kaggle:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Place it inside:

/chest_xray/


❤️ Made for Better Healthcare

This project is developed to support early pneumonia detection, reduce workload on radiologists, and demonstrate the power of AI in medical imaging.
Lightweight, accurate, and easy to deploy — perfectly suited for real-world applications.
