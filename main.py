# ==============================
# Face Mask Detection (FINAL)
# ==============================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# STEP 1: LOAD DATA
# ==============================

print("Loading dataset...")

data = []
labels = []

categories = ["with_mask", "without_mask"]

for category in categories:
    base_path = r"D:\Vishwakarma University\Sem 6\Neural network and reinforcement learning\project\dataset"
    path = os.path.join(base_path, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (128,128))
            data.append(image)
            labels.append(label)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

print("Dataset Loaded:", data.shape)

# ==============================
# STEP 2: SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ==============================
# STEP 3: SMALL DATA FOR ML
# ==============================

X_train_small = X_train[:2000]
y_train_small = y_train[:2000]

X_test_small = X_test[:500]
y_test_small = y_test[:500]

# Flatten
X_train_flat = X_train_small.reshape(len(X_train_small), -1)
X_test_flat = X_test_small.reshape(len(X_test_small), -1)

# Normalize
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# ==============================
# STEP 4: BASELINE MODELS
# ==============================

print("\nTraining Baseline Models...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_flat, y_train_small)
lr_pred = lr.predict(X_test_flat)
lr_acc = accuracy_score(y_test_small, lr_pred)

# SVM
svm = SVC(kernel='linear')
svm.fit(X_train_flat, y_train_small)
svm_pred = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test_small, svm_pred)

print("Logistic Regression Accuracy:", lr_acc)
print("SVM Accuracy:", svm_acc)

# ==============================
# STEP 5: HOG FEATURES
# ==============================

print("\nExtracting HOG features...")

def extract_hog(images):
    features = []
    for img in images:
        hog_feature = hog(img,
                          pixels_per_cell=(8,8),
                          cells_per_block=(2,2),
                          channel_axis=-1)
        features.append(hog_feature)
    return np.array(features)

X_train_hog = extract_hog(X_train_small)
X_test_hog = extract_hog(X_test_small)

svm_hog = SVC(kernel='linear')
svm_hog.fit(X_train_hog, y_train_small)
hog_pred = svm_hog.predict(X_test_hog)
hog_acc = accuracy_score(y_test_small, hog_pred)

print("SVM + HOG Accuracy:", hog_acc)

# ==============================
# STEP 6: CNN MODEL
# ==============================

print("\nBuilding CNN model...")

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==============================
# STEP 7: TRAIN CNN
# ==============================

print("\nTraining CNN...")

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32
)

model.save("mask_model.h5")

# ==============================
# STEP 8: EVALUATE CNN
# ==============================

loss, cnn_acc = model.evaluate(X_test, y_test)

print("\nCNN Accuracy:", cnn_acc)

# ==============================
# STEP 9: PLOT ACCURACY
# ==============================

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# ==============================
# STEP 10: PLOT LOSS
# ==============================

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ==============================
# STEP 11: CONFUSION MATRIX
# ==============================

print("\nGenerating Confusion Matrix...")

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=["Mask", "No Mask"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ==============================
# FINAL RESULTS
# ==============================

print("\n===== FINAL RESULTS =====")
print("Logistic Regression:", lr_acc)
print("SVM:", svm_acc)
print("SVM + HOG:", hog_acc)
print("CNN:", cnn_acc)

# ==============================
# SAMPLE OUTPUT CHECK
# ==============================

print("\nSample Predictions:", y_pred[:10])
print("Actual Labels:", y_test[:10])