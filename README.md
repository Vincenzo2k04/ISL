
# 🧠 Indian Sign Language Recognition using Deep Learning

This project focuses on the classification of Indian Sign Language (ISL) gestures using Convolutional Neural Networks (CNNs) implemented in TensorFlow. It aims to bridge communication gaps by interpreting hand signs into textual representations.

---

## 📂 Dataset

- **Directory**: `E:\Projects\ISL`
- **Structure**: Each folder inside the dataset directory represents a class corresponding to a particular ISL sign.
- **Image Format**: JPG/PNG
- **Image Size**: Resized to 128x128 pixels.

---

## ⚙️ Model Architecture

The CNN model used in this project consists of:
- **Data Augmentation**: `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`
- **Conv Layers**: Multiple `Conv2D` and `MaxPooling2D` layers
- **Flattening**: `Flatten` layer to convert 2D feature maps to 1D
- **Dense Layers**: Fully connected layers with dropout for regularization

---

## 📦 Libraries Used

```python
tensorflow
numpy
matplotlib
os
```

Install requirements using:

```bash
pip install tensorflow numpy matplotlib
```

---

## 🧪 Training Configuration

- **Image Size**: 128 x 128
- **Batch Size**: 32
- **Validation Split**: 20%
- **Shuffle**: Enabled for training set
- **Label Mode**: Categorical (One-hot encoded)

---

## 🖼️ Image Loading

Images are loaded using `tf.keras.utils.image_dataset_from_directory` with automatic label inference.

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=123,
    shuffle=True
)
```

---

## 🚀 Results

- Model performance can be evaluated using metrics like accuracy and loss plotted over epochs.
- Accuracy varies depending on class distribution and dataset quality.

---

## 📈 Improvements & Future Work

- Increase dataset size and diversity
- Apply more complex CNN architectures or transfer learning (e.g., ResNet, MobileNet)
- Deploy the model with a real-time webcam interface for gesture detection

---

## 👨‍💻 Author

**Tanishq Patil**  
Third Year AIML Student  
Manipal University Jaipur
