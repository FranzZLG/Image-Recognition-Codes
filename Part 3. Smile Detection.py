import cv2
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to Smile Haar Cascade file (in the same directory as the script)
smile_cascade_path = os.path.join(script_dir, "haarcascade_smile.xml")
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Folder to save captured images (in the same directory as the script)
smile_detection_folder = os.path.join(script_dir, "Smile Detection")

# Ensure the output folder exists
os.makedirs(smile_detection_folder, exist_ok=True)

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

    # Convert frame to grayscale for smile detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect smiles in the frame
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=20, minSize=(50, 50))   

    # Draw rectangles around detected smiles (Color: White)
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the frame with detected smiles
    cv2.imshow("Smile Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter key to save the frame
        if len(smiles) > 0:  # Only save the frame if smiles are detected
            # Generate a unique filename
            file_name = f"smile_detected{image_counter}.jpg"
            file_path = os.path.join(smile_detection_folder, file_name)

            # Save the frame
            cv2.imwrite(file_path, frame)
            print(f"Frame saved to {file_path}")

            # Increment the counter
            image_counter += 1

    elif key == 27:  # Esc key to quit
        break

camera.release()
cv2.destroyAllWindows()