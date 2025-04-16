# 🤟 Indian Sign Language (ISL) Image Classification using PyTorch Lightning

This repository contains an image classification project that uses a custom Convolutional Neural Network (CNN) to classify Indian Sign Language (ISL) alphabets. The model is built with PyTorch Lightning for clean, modular, and scalable deep learning.

---

## 📂 Dataset

- **Source**: [ISL Alphabet Dataset](https://www.kaggle.com/datasets/vaishalahunt/isl-alphabet-dataset)
- **Classes**: 23 (A-Z excluding H, J, Y)
- **Format**: Folder-structured dataset where each folder corresponds to a class

---

## 🧠 Model Architecture

The custom CNN includes:
- 2 convolutional layers
- 3 fully connected (dense) layers
- ReLU activations
- Max pooling
- Log Softmax output

Model Summary:
```
conv1: 3 → 6
conv2: 6 → 16
fc1: 46656 → 120
fc2: 120 → 84
fc3: 84 → 20
fc4: 20 → 23 (num_classes)
```

---

## 🚀 Training & Evaluation

- **Framework**: PyTorch Lightning
- **Optimizer**: Adam
- **Loss Function**: Cross Entropy Loss
- **Epochs**: 100
- **Batch Size**: 32

**Transforms**:
- Random Rotation
- Horizontal Flip
- Resize to 224x224
- Center Crop
- Normalization

---

## 📈 Results

After 100 epochs of training:

| Metric       | Value       |
|--------------|-------------|
| Test Accuracy| **86.52%**  |
| Test Loss    | **1.0159**  |
| F1-Score     | Avg ~81.9%  |

📊 **Classification Report Snippet**:
```
Class  | Precision | Recall | F1-score
-------|-----------|--------|---------
  C    | 0.8750    | 1.0000 | 0.9333
  D    | 0.8000    | 1.0000 | 0.8889
  P    | 1.0000    | 0.8571 | 0.9231
  T    | 1.0000    | 1.0000 | 1.0000
  U    | 1.0000    | 1.0000 | 1.0000
```

## 📦 Requirements

```
torch
torchvision
pytorch-lightning
numpy
pandas
scikit-learn
tqdm
matplotlib
Pillow
```

---

## 📌 Notes

- If using GPU, ensure CUDA is properly configured.
- You can increase `num_workers` in `DataLoader` to speed up data loading.
- Model checkpointing and logging are managed by PyTorch Lightning.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Kaggle ISL Dataset](https://www.kaggle.com/datasets/vaishalahunt/isl-alphabet-dataset)
