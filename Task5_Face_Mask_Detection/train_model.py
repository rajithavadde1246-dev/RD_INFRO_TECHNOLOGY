import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Dataset paths
with_mask_path = "dataset/with_mask"
without_mask_path = "dataset/without_mask"

data = []
labels = []

# Load with mask images
for image in os.listdir(with_mask_path):
    img_path = os.path.join(with_mask_path, image)

    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)

    data.append(img)
    labels.append(1)

# Load without mask images
for image in os.listdir(without_mask_path):
    img_path = os.path.join(without_mask_path, image)

    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)

    data.append(img)
    labels.append(0)

# Convert to NumPy
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print("Total Images:", len(data))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# MobileNetV2 Model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training Started...")

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=4,
    validation_data=(X_test, y_test)
)

model.save("mask_detector_model.h5")

print("Model Saved Successfully!")