import cv2
import os

# Define the generic folder path based on the user's home directory
capture_image_folder = os.path.join(os.path.expanduser("~"), "Downloads", "Image Recognition Codes", "Capture Image")

# Ensure the folder exists, create it if it doesn't
os.makedirs(capture_image_folder, exist_ok=True)

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Failed to open the camera.")
    exit()

# Counter for the filenames
image_counter = 1

while True:
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow("Capture Image", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # "Enter" key to capture image
        # Generate the filename
        file_name = f"captured_image_{image_counter}.jpg"
        file_path = os.path.join(capture_image_folder, file_name)

        # Save the image
        cv2.imwrite(file_path, frame)
        print(f"Image saved to {file_path}")

        # Increment the counter for the next image
        image_counter += 1
    elif key == 27:  # "Esc" key to quit
        break

camera.release()
cv2.destroyAllWindows()