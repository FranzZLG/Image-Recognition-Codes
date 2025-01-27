import cv2
import os
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a folder to store collected face images
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Initialize face ID and a counter for image collection
face_id = input('Enter your ID: ')
count = 0

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Capture face image
        count += 1
        face = gray[y:y + h, x:x + w]
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", face)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow("Face Collecting", frame)

    # Break if 500 images are collected
    if count >= 500:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()