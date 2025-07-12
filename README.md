# 🚦 GTSRB Traffic Sign Classification using PyTorch

This project focuses on **multi-class image classification** of the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. Two deep learning approaches were used:

1. ✅ A custom-built CNN model  
2. ✅ Transfer learning with fine-tuned **ReXNet v1.5** using the `timm` library

---

## 🗂️ Dataset Overview

- **Name**: [GTSRB - German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
- **Classes**: 43 unique traffic signs  
- **Total Images**: 39,270  
- **Split**:
  - Training: 80%
  - Validation: 20%
  - Test: Predefined split

---

## 📌 Project Highlights

- 📦 **Automatic dataset loading** via `torchvision.datasets.GTSRB`
- 🖼️ Image preprocessing: resizing to `224x224`, ImageNet normalization
- 📊 Class distribution analysis (bar chart and pie chart)
- 🎨 Grad-CAM++ for visual explainability
- 📈 Accuracy, loss, and F1-score curves
- 📥 Model saving based on best validation F1-score

---

## 🧠 Models

### 1. 🔧 Custom CNN Model

A 4-block convolutional network with batch normalization and max-pooling:

Conv2D --> ReLU --> BatchNorm --> MaxPool (x4 blocks)     
Flatten --> FC(128) --> ReLU --> FC(43)


### 2. ⚙️ ReXNet v1.5 (Pretrained)

Fine-tuned using `timm.create_model()` with:
- Pretrained weights on ImageNet
- Final classifier modified to `num_classes=43`

---

## 🧪 Training Strategy

| Component         | Value/Technique                  |
|------------------|----------------------------------|
| Loss Function     | CrossEntropyLoss                 |
| Optimizer         | Adam (lr = 3e-4)                 |
| Scheduler         | ReduceLROnPlateau (val loss)     |
| Metric            | Accuracy, F1-score (`torchmetrics`) |
| Early Stopping    | Patience = 3                     |
| Device            | GPU (`cuda`) or CPU fallback     |

---

## 📸 Visualizations

### ▶️ Random Sample Visualization
Shows 15 random images and their true class labels.

### 📊 Class Distribution (Train/Val/Test)

Bar charts and pie charts to highlight **dataset imbalance**.

![Train Class Distribution](https://github.com/user-attachments/assets/41015714-4373-4e1e-8bf5-35c08a95a9a3)
![Validation Class Distribution](https://github.com/user-attachments/assets/5a9d526d-93c7-4319-9c73-4173bd0e8bfa)
![Test Class Distribution](https://github.com/user-attachments/assets/bf01f4fc-1316-4185-bfa1-16a2ca8fbb17)

---

## 📈 Learning Curves

### ✅ Loss
![Train vs Val Loss](https://github.com/user-attachments/assets/216c16f8-d13f-49a1-8920-ea8477d479a6)

### ✅ Accuracy
![Train vs Val Accuracy](https://github.com/user-attachments/assets/09d63b1f-5866-4fa2-92a6-b45e8ece211f)

### ✅ F1 Score
![Train vs Val F1](https://github.com/user-attachments/assets/4a198839-7e25-478d-9347-93b9ecc1fa20)

---

## 🔍 Grad-CAM++ & Confusion Matrix

Visualizes model attention on correct and incorrect predictions.

- 🔥 Grad-CAM++ overlay on test images  
- ✅ Prediction probability bar chart  
- 📉 Confusion matrix for all 43 classes

![Grad-CAM Example](https://github.com/user-attachments/assets/ed801bec-05a6-4065-9a0e-3b78e4d2f4d4)

---

## ✅ Final Test Accuracy

> **📊 95.3% Accuracy** on the official test set using `ReXNet_150`.

---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/Matrasulov/GTRSB-dataset.git
cd GTRSB-dataset

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py  # Or use the notebook provided
