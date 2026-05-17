# Face Mask Detection System 😷

A real-time Face Mask Detection System built using **Python, OpenCV, TensorFlow/Keras, and CNN**.
This project detects whether a person is wearing a mask or not through a webcam feed and provides live predictions.

---

## 📌 Features

* Real-time face detection using webcam
* Detects:

  * ✅ Mask
  * ❌ No Mask
* Voice alert for no-mask detection
* CNN-based deep learning model
* Fast and lightweight implementation
* Easy to train with custom dataset

---

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Haar Cascade Classifier
* pyttsx3 (Voice Alert)

---

## 📂 Project Structure

```bash
Face-Mask-Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── mask_model.h5
├── train_model.py
├── realtime.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python tensorflow numpy pyttsx3
```

---

## ▶️ Run the Project

### Train the Model

```bash
python train_model.py
```

### Run Real-Time Detection

```bash
python realtime.py
```

---

## 🧠 Model Information

The project uses a **Convolutional Neural Network (CNN)** for image classification.

### CNN Layers Used

* Convolution Layer
* Max Pooling Layer
* Flatten Layer
* Dense Layer
* Output Layer

---

## 📸 Working

1. Webcam captures live video
2. OpenCV detects faces
3. Face image is passed to CNN model
4. Model predicts:

   * Mask
   * No Mask
5. Result displayed on screen with confidence score
6. Voice alert plays if no mask is detected

---

## 📊 Future Enhancements

* Mobile application integration
* Multiple face tracking
* Better accuracy using advanced models
* IoT integration for smart monitoring
* Cloud deployment

---

## 🎯 Applications

* Hospitals
* Airports
* Schools & Colleges
* Offices
* Public Transport
* Shopping Malls

---

## 👨‍💻 Authors

* Tanish Jain
* Team Members

---

## 📖 Conclusion

This project demonstrates how Deep Learning and Computer Vision can be used together to build a real-time safety monitoring system. The Face Mask Detection System helps automate mask compliance monitoring efficiently and accurately.

---

## 📜 License

This project is for educational purposes only.

---

## ⭐ GitHub

If you like this project, give it a ⭐ on GitHub!
