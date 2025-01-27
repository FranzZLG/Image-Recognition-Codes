import cv2
import numpy as np
import os

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

names = {1: "Franz"}

video_capture = cv2.VideoCapture(0)

save_directory = r"C:\Users\franz\Downloads\Image Recognition Codes\Saved Pictures"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

counter = 0

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Recognize the face
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If the confidence is higher than a certain threshold, label it as "Unknown"
        if confidence < 100:
            name = names.get(id, "Unknown")
        else:
            name = "Unknown"  # If face is not recognized, label it as "Unknown"

        # Draw a rectangle around the face and display the name (Color: White)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("Face Recognition", frame)

    # Check for 'Enter' key to save an image
    if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter key
        # Create the filename based on the current counter
        base_filename = "Gloriani_FranzLouise_"
        image_path = os.path.join(save_directory, f"{base_filename}{counter}.jpg")

        # Check if the file already exists, if so, increment the counter
        while os.path.exists(image_path):
            counter += 1
            image_path = os.path.join(save_directory, f"{base_filename}{counter}.jpg")

        # Save the current frame as an image
        cv2.imwrite(image_path, frame)
        print(f"Image saved as: {image_path}")

        # Increment counter for the next image
        counter += 1

    # Check for ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        print("Exiting...")
        break

video_capture.release()
cv2.destroyAllWindows()
