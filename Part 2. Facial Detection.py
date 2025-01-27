import cv2
import os

# Get the directory of the file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to Haar Cascade file 
cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# Folder to save captured images
facial_detection_folder = os.path.join(script_dir, "Facial Detection")

# Ensure the output folder exists
os.makedirs(facial_detection_folder, exist_ok=True)

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Failed to open the camera.")
    exit()

# Counter for unique filenames
image_counter = 1

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces (Color: White)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the frame with detected faces
    cv2.imshow("Facial Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # "Enter" key to save the frame
        if len(faces) > 0:  # Only save the frame if faces are detected
            # Generate a unique filename
            file_name = f"face_detected{image_counter}.jpg"
            file_path = os.path.join(facial_detection_folder, file_name)

            # Save the frame
            cv2.imwrite(file_path, frame)
            print(f"Frame saved to {file_path}")

            # Increment the counter
            image_counter += 1

    elif key == 27:  # "Esc" key to quit
        break

camera.release()
cv2.destroyAllWindows()