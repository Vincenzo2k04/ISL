```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = 'E:\Projects\ISL'
```


```python
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
```


```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=123
)
```

    Found 3780 files belonging to 35 classes.
    Using 3024 files for training.
    Found 3780 files belonging to 35 classes.
    Using 756 files for validation.



```python
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Detected {NUM_CLASSES} classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2),
], name="data_augmentation")

```

    Detected 35 classes: ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



```python
model = Sequential([
    data_augmentation,
    Rescaling(1./255), 
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 50

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('ISL.keras')
print("Model saved as 'ISL.keras'")
```

    C:\Users\tanis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\keras\src\layers\convolutional\base_conv.py:99: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ data_augmentation (<span style="color: #0087ff; text-decoration-color: #0087ff">Sequential</span>)  │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ rescaling (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)           │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ ?                      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    Epoch 1/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 447ms/step - accuracy: 0.0705 - loss: 3.4272 - val_accuracy: 0.3320 - val_loss: 2.3013
    Epoch 2/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 443ms/step - accuracy: 0.3952 - loss: 1.9629 - val_accuracy: 0.9788 - val_loss: 0.2206
    Epoch 3/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 443ms/step - accuracy: 0.6837 - loss: 0.9740 - val_accuracy: 0.9246 - val_loss: 0.2146
    Epoch 4/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 409ms/step - accuracy: 0.7895 - loss: 0.6333 - val_accuracy: 0.9921 - val_loss: 0.0944
    Epoch 5/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 398ms/step - accuracy: 0.8711 - loss: 0.3880 - val_accuracy: 0.9947 - val_loss: 0.0628
    Epoch 6/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 382ms/step - accuracy: 0.8960 - loss: 0.3320 - val_accuracy: 0.9563 - val_loss: 0.0591
    Epoch 7/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 359ms/step - accuracy: 0.9344 - loss: 0.1932 - val_accuracy: 1.0000 - val_loss: 0.0026
    Epoch 8/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 396ms/step - accuracy: 0.9461 - loss: 0.1647 - val_accuracy: 1.0000 - val_loss: 0.0062
    Epoch 9/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 371ms/step - accuracy: 0.9501 - loss: 0.1469 - val_accuracy: 1.0000 - val_loss: 0.0030
    Epoch 10/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 401ms/step - accuracy: 0.9549 - loss: 0.1345 - val_accuracy: 0.9934 - val_loss: 0.0106
    Epoch 11/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 378ms/step - accuracy: 0.9519 - loss: 0.1440 - val_accuracy: 1.0000 - val_loss: 0.0023
    Epoch 12/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 375ms/step - accuracy: 0.9560 - loss: 0.1270 - val_accuracy: 1.0000 - val_loss: 7.2252e-04
    Epoch 13/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m45s[0m 478ms/step - accuracy: 0.9724 - loss: 0.0878 - val_accuracy: 1.0000 - val_loss: 2.9392e-04
    Epoch 14/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 465ms/step - accuracy: 0.9648 - loss: 0.0974 - val_accuracy: 0.9683 - val_loss: 0.0505
    Epoch 15/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 395ms/step - accuracy: 0.9628 - loss: 0.1302 - val_accuracy: 1.0000 - val_loss: 0.0022
    Epoch 16/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 397ms/step - accuracy: 0.9721 - loss: 0.0736 - val_accuracy: 1.0000 - val_loss: 0.0011
    Epoch 17/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 409ms/step - accuracy: 0.9751 - loss: 0.0864 - val_accuracy: 1.0000 - val_loss: 0.0069
    Epoch 18/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 383ms/step - accuracy: 0.9788 - loss: 0.0631 - val_accuracy: 1.0000 - val_loss: 7.3071e-04
    Epoch 19/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 433ms/step - accuracy: 0.9830 - loss: 0.0452 - val_accuracy: 1.0000 - val_loss: 2.8805e-04
    Epoch 20/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 422ms/step - accuracy: 0.9834 - loss: 0.0492 - val_accuracy: 1.0000 - val_loss: 6.6288e-04
    Epoch 21/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 405ms/step - accuracy: 0.9799 - loss: 0.0657 - val_accuracy: 1.0000 - val_loss: 2.8228e-05
    Epoch 22/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 436ms/step - accuracy: 0.9736 - loss: 0.0800 - val_accuracy: 1.0000 - val_loss: 1.4674e-05
    Epoch 23/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 400ms/step - accuracy: 0.9765 - loss: 0.0661 - val_accuracy: 1.0000 - val_loss: 2.6926e-04
    Epoch 24/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 372ms/step - accuracy: 0.9803 - loss: 0.0608 - val_accuracy: 1.0000 - val_loss: 1.0093e-04
    Epoch 25/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 370ms/step - accuracy: 0.9846 - loss: 0.0516 - val_accuracy: 1.0000 - val_loss: 0.0121
    Epoch 26/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 363ms/step - accuracy: 0.9851 - loss: 0.0507 - val_accuracy: 1.0000 - val_loss: 5.1101e-04
    Epoch 27/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 361ms/step - accuracy: 0.9839 - loss: 0.0497 - val_accuracy: 1.0000 - val_loss: 6.4173e-05
    Epoch 28/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 395ms/step - accuracy: 0.9885 - loss: 0.0346 - val_accuracy: 1.0000 - val_loss: 2.6035e-06
    Epoch 29/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 381ms/step - accuracy: 0.9781 - loss: 0.0863 - val_accuracy: 1.0000 - val_loss: 0.0043
    Epoch 30/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 392ms/step - accuracy: 0.9706 - loss: 0.0883 - val_accuracy: 1.0000 - val_loss: 3.5024e-04
    Epoch 31/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 378ms/step - accuracy: 0.9863 - loss: 0.0376 - val_accuracy: 1.0000 - val_loss: 6.8217e-05
    Epoch 32/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 360ms/step - accuracy: 0.9833 - loss: 0.0436 - val_accuracy: 1.0000 - val_loss: 4.0152e-05
    Epoch 33/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 368ms/step - accuracy: 0.9766 - loss: 0.0589 - val_accuracy: 1.0000 - val_loss: 2.3819e-06
    Epoch 34/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 362ms/step - accuracy: 0.9907 - loss: 0.0290 - val_accuracy: 1.0000 - val_loss: 1.2471e-06
    Epoch 35/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 360ms/step - accuracy: 0.9858 - loss: 0.0419 - val_accuracy: 1.0000 - val_loss: 7.6468e-04
    Epoch 36/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 364ms/step - accuracy: 0.9795 - loss: 0.0732 - val_accuracy: 1.0000 - val_loss: 1.7250e-05
    Epoch 37/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 361ms/step - accuracy: 0.9867 - loss: 0.0417 - val_accuracy: 1.0000 - val_loss: 3.4317e-05
    Epoch 38/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 361ms/step - accuracy: 0.9940 - loss: 0.0214 - val_accuracy: 1.0000 - val_loss: 3.9660e-06
    Epoch 39/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 364ms/step - accuracy: 0.9871 - loss: 0.0415 - val_accuracy: 1.0000 - val_loss: 2.1923e-04
    Epoch 40/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 382ms/step - accuracy: 0.9896 - loss: 0.0336 - val_accuracy: 1.0000 - val_loss: 2.6068e-05
    Epoch 41/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 365ms/step - accuracy: 0.9878 - loss: 0.0396 - val_accuracy: 1.0000 - val_loss: 1.5099e-04
    Epoch 42/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 367ms/step - accuracy: 0.9902 - loss: 0.0345 - val_accuracy: 1.0000 - val_loss: 4.2969e-04
    Epoch 43/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 360ms/step - accuracy: 0.9891 - loss: 0.0414 - val_accuracy: 0.9947 - val_loss: 0.0115
    Epoch 44/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 361ms/step - accuracy: 0.9860 - loss: 0.0415 - val_accuracy: 1.0000 - val_loss: 1.3300e-04
    Epoch 45/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 364ms/step - accuracy: 0.9894 - loss: 0.0312 - val_accuracy: 1.0000 - val_loss: 3.8818e-05
    Epoch 46/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 364ms/step - accuracy: 0.9905 - loss: 0.0344 - val_accuracy: 1.0000 - val_loss: 0.0010
    Epoch 47/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 360ms/step - accuracy: 0.9773 - loss: 0.0628 - val_accuracy: 1.0000 - val_loss: 6.7013e-06
    Epoch 48/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 358ms/step - accuracy: 0.9902 - loss: 0.0251 - val_accuracy: 1.0000 - val_loss: 2.0982e-05
    Epoch 49/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 365ms/step - accuracy: 0.9831 - loss: 0.0497 - val_accuracy: 1.0000 - val_loss: 5.3101e-04
    Epoch 50/50
    [1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m34s[0m 357ms/step - accuracy: 0.9833 - loss: 0.0550 - val_accuracy: 1.0000 - val_loss: 2.5792e-04



    
![png](output_4_7.png)
    


    Model saved as 'ISL.keras'



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\6_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 144ms/step
    
    Predicted class: 6
    Confidence: 100.00%



    
![png](output_5_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\7_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 132ms/step
    
    Predicted class: 7
    Confidence: 100.00%



    
![png](output_6_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\8_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 134ms/step
    
    Predicted class: 8
    Confidence: 100.00%



    
![png](output_7_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\9_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 134ms/step
    
    Predicted class: 9
    Confidence: 100.00%



    
![png](output_8_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\T_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001FCD970B560> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 137ms/step
    
    Predicted class: T
    Confidence: 99.84%



    
![png](output_9_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\A_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    WARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001FCD957C720> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 138ms/step
    
    Predicted class: A
    Confidence: 100.00%



    
![png](output_10_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\N_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 133ms/step
    
    Predicted class: N
    Confidence: 100.00%



    
![png](output_11_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\I_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 134ms/step
    
    Predicted class: I
    Confidence: 100.00%



    
![png](output_12_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\S_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 135ms/step
    
    Predicted class: S
    Confidence: 100.00%



    
![png](output_13_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\H_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 124ms/step
    
    Predicted class: H
    Confidence: 100.00%



    
![png](output_14_1.png)
    



```python
#Tester 

tf.config.set_visible_devices([], 'GPU')
print("Configured to use CPU only.")

MODEL_PATH = 'ISL.keras' 
IMAGE_PATH = "E:\\Projects\\ISL_Test\\Q_Test.jpg"

IMAGE_SIZE = (128, 128) 

CLASS_NAMES = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

test_image_array = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

predictions = model.predict(test_image_array)

predicted_probabilities = predictions[0] 
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = predicted_probabilities[predicted_class_index] * 100

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(6, 6))
display_img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
```

    Configured to use CPU only.
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 135ms/step
    
    Predicted class: Q
    Confidence: 100.00%



    
![png](output_15_1.png)
    

