import cv2
import os
import numpy as np
from PIL import Image

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare the training data
faces = []
labels = []

# Path to dataset folder
dataset_path = 'dataset'

# Loop through each file in the dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg'):
            # Extract label (ID) from filename
            label = int(file.split('.')[1])
            image_path = os.path.join(root, file)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            face_np = np.array(image, 'uint8')

            # Detect face in the image
            faces_detected = face_cascade.detectMultiScale(face_np)

            for (x, y, w, h) in faces_detected:
                faces.append(face_np[y:y + h, x:x + w])
                labels.append(label)

# Train the model using LBPH (Local Binary Pattern Histogram)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save('trainer.yml')

print("Training Complete")