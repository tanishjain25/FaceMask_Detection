import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import time
import pyttsx3

# ==============================
# LOAD MODEL
# ==============================
model = load_model(
    r"D:\Vishwakarma University\Sem 6\Neural network and reinforcement learning\project\mask_model.h5",
    compile=False
)

# ==============================
# LOAD FACE DETECTOR
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# RELIABLE VOICE ALERT (WINDOWS)
# ==============================
import threading
import time
import os

alert_active = False

def speak():
    os.system('powershell -Command "Add-Type –AssemblyName System.Speech; '
              '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'Please wear a mask\');"')

def alert_loop():
    global alert_active
    while alert_active:
        speak()
        time.sleep(2)

def start_alert():
    global alert_active
    if not alert_active:
        alert_active = True
        threading.Thread(target=alert_loop, daemon=True).start()

def stop_alert():
    global alert_active
    alert_active = False
    
# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit")

prev_label = None
stable_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Ignore small detections
        if w < 80 or h < 80:
            continue

        face = frame[y:y+h, x:x+w]

        try:
            # Preprocess
            face = cv2.resize(face, (128, 128))
            face = cv2.GaussianBlur(face, (5, 5), 0)
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            # Confidence filtering
            if prediction < 0.3:
                label = "With Mask"
                color = (0, 255, 0)
                confidence = (1 - prediction) * 100
                stop_alert()

            elif prediction > 0.7:
                label = "No Mask"
                color = (0, 0, 255)
                confidence = prediction * 100
                start_alert()

            else:
                continue

            # Stability filter
            if label == prev_label:
                stable_count += 1
            else:
                stable_count = 0
                prev_label = label

            if stable_count < 3:
                continue

            # Warning text
            if label == "No Mask":
                cv2.putText(frame,
                            "WARNING: NO MASK!",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

            # Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Label
            cv2.putText(frame,
                        f"{label} ({confidence:.1f}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        except:
            continue

    cv2.imshow("Mask Detection (Final with Alert)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()