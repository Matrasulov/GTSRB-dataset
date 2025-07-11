# GTSRB Traffic Sign Classification

This project implements image classification on the German Traffic Sign Recognition Benchmark (GTSRB) dataset using two approaches:
1. A custom-built Convolutional Neural Network (CNN)
2. Fine-tuning a pre-trained ResNet-18 model

## 🧠 Project Highlights

- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Framework**: PyTorch, Torchvision
- **Approaches**:
  - Custom CNN model trained from scratch
  - Transfer learning with ResNet-18 using fine-tuning
- **Data Handling**: Automatic download with `torchvision.datasets.GTSRB`
- **Transforms**: Image resizing and normalization (ImageNet stats)

## 🔧 Features

- Modular training and evaluation pipeline
- Image classification with accuracy tracking
- Transfer learning with frozen and unfrozen layers
- Side-by-side performance comparison of both models


## 📊 Training Curves

### 🔹 Custom CNN Accuracy

![CNN Accuracy](images/cnn_accuracy.png)

---

### 🔹 Fine-Tuned ResNet18 Accuracy

![ResNet Accuracy](images/resnet_accuracy.png)
