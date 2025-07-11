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

## Plotting random images
<img width="1589" height="996" alt="image" src="https://github.com/user-attachments/assets/0d7cb572-6172-4f51-9ef1-33839344a490" />

## Class imbalance analysis
<img width="1389" height="390" alt="image" src="https://github.com/user-attachments/assets/41015714-4373-4e1e-8bf5-35c08a95a9a3" />
<img width="1389" height="390" alt="image" src="https://github.com/user-attachments/assets/5a9d526d-93c7-4319-9c73-4173bd0e8bfa" />
<img width="1389" height="390" alt="image" src="https://github.com/user-attachments/assets/bf01f4fc-1316-4185-bfa1-16a2ca8fbb17" />

## 📊 Training Curves

### 🔹 Custom CNN Accuracy

![CNN Accuracy](images/cnn_accuracy.png)

---

### 🔹 Fine-Tuned ResNet18 Accuracy

![ResNet Accuracy](images/resnet_accuracy.png)
