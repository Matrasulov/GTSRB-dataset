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
