import cv2
import os
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

# Get the directory of the file
script_directory = os.path.dirname(os.path.abspath(__file__))

# Paths to Haar Cascade files and Design files
face_cascade_path = os.path.join(script_directory, "haarcascade_frontalface_default.xml")
smile_cascade_path = os.path.join(script_directory, "haarcascade_smile.xml")
saved_pictures_folder = os.path.join(script_directory, "Saved Pictures")
background_image_path = os.path.join(script_directory, "GUI (Design)", "background_image.png")
background_detection_image_path = os.path.join(script_directory, "GUI (Design)", "background_detection.png")
last_background_image_path = os.path.join(script_directory, "GUI (Design)", "Motivation.png")

# Check if save directory exists
if not os.path.exists(saved_pictures_folder):
    os.makedirs(saved_pictures_folder)

# Check if Haar cascade files exist
if not os.path.exists(face_cascade_path) or not os.path.exists(smile_cascade_path):
    print("Error: Haar cascade files not found.")
    quit()

# Check if background image exists
if not os.path.exists(background_image_path):
    print("Error: Background image not found.")
    quit()

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Check if Haar cascades are loaded correctly
if face_cascade.empty():
    print("Error loading face cascade")
else:
    print("Face cascade loaded successfully")

if smile_cascade.empty():
    print("Error loading smile cascade")
else:
    print("Smile cascade loaded successfully")

# Initialize global variables
camera = cv2.VideoCapture(0)
current_frame = None
image_count = 0

# Load LBPH Face Recognizer and trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(script_directory, 'trainer.yml'))

# Names dictionary for recognized faces
names = {1: "Franz"}

# Define the zoom factor
zoom_factor = 0.7

# Function to update the camera feed
def update_feed():
    global current_frame
    ret, frame = camera.read()
    if ret:
        # Resize the frame to simulate zoom out
        frame_resized = cv2.resize(frame, (0, 0), fx=zoom_factor, fy=zoom_factor)

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame_resized[y:y + h, x:x + w]

            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))

            if len(smiles) > 0:
                cv2.putText(frame_resized, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                cv2.putText(frame_resized, "Say 'Cheese!'", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Recognize the face using LBPH face recognizer
            id, confidence = recognizer.predict(roi_gray)
            if confidence < 90:
                name = names.get(id, "Unknown")
            else:
                name = "Unknown"
            cv2.putText(frame_resized, name, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        current_frame = frame_resized.copy()
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, update_feed)  # Slight delay to ensure smooth feed

# Function to quit the application
def quit_app():
    global camera
    camera.release()
    root.quit()
    root.destroy()

# Function to switch to detection screen
def start_detection():
    detection_frame.pack(fill="both", expand=True)
    first_frame.pack_forget()
    start_button.place_forget()
    quit_button.place_forget()

# Function to return to main menu
def back_to_main_menu():
    detection_frame.pack_forget()
    first_frame.pack(fill="both", expand=True)
    start_button.place(x=250, y=215)
    quit_button.place(x=250, y=275)

# Function to capture and save the current frame
def capture_image():
    global current_frame, image_count
    if current_frame is not None:
        image_count += 1
        file_name = os.path.join(saved_pictures_folder, f"Gloriani, FranzLouise_{image_count}.jpg")
        cv2.imwrite(file_name, current_frame)
        print(f"Captured image saved as: {file_name}")

# Function to switch to last frame
def last_frame():
    detection_frame.pack_forget()
    last_frame.pack(fill="both", expand=True)
    back_button.place(x=250, y=600)

# Function to return to detection screen
def last_back():
    detection_frame.pack(fill="both", expand=True)
    last_frame.pack_forget()
    back_button.place_forget()

# Function to change the background color on hover
def on_enter(event, button):
    button.config(bg="dark grey")
    button.config(fg="white")

# Function to reset the button style when the mouse leaves
def on_leave(event, button):
    button.config(bg="light grey")
    button.config(fg="black")

# Create Tkinter window
root = tk.Tk()
root.title("Python-based Facial Detection and Recognition System")
root.geometry("700x700")
root.resizable(False, False)

# Background image setup for the main menu
background_image = Image.open(background_image_path)
background_photo = ImageTk.PhotoImage(background_image)
first_frame = tk.Canvas(root, width=700, height=700)
first_frame.pack(fill="both", expand=True)
first_frame.create_image(0, 0, anchor=tk.NW, image=background_photo)

# Main Menu Frame
main_menu_frame = tk.Frame(root, bg="white")
main_menu_frame.place(relx=0.5, rely=0.5, anchor="center")

start_button = Button(root, text="S T A R T", command=start_detection, padx=10, pady=5, font=("Helvetica", 14), width=15, bg="light grey", fg="black", bd=5)
start_button.place(x=250, y=215)

# Bind hover effects to the start button
start_button.bind("<Enter>", lambda event, button=start_button: on_enter(event, button))
start_button.bind("<Leave>", lambda event, button=start_button: on_leave(event, button))

quit_button = Button(root, text="Q U I T", command=quit_app, padx=10, pady=5, font=("Helvetica", 14), width=15, bg="light grey", fg="black", bd=5)
quit_button.place(x=250, y=275)

# Bind hover effects to the quit button
quit_button.bind("<Enter>", lambda event, button=quit_button: on_enter(event, button))
quit_button.bind("<Leave>", lambda event, button=quit_button: on_leave(event, button))

# Detection System Frame with background image
background_detection_image = Image.open(background_detection_image_path)
background_detection_photo = ImageTk.PhotoImage(background_detection_image)

# Create a Canvas for background in detection frame
detection_frame = tk.Frame(root)
detection_canvas = tk.Canvas(detection_frame, width=700, height=700)
detection_canvas.pack(fill="both", expand=True)

# Create background image on the canvas
detection_canvas.create_image(0, 0, anchor=tk.NW, image=background_detection_photo)

# Pack the video feed and buttons on top of the background
video_label = Label(detection_frame)
video_label.place(x=350, y=232, anchor="center", width=615, height=400)

# Load the image for the capture button
capture_button_image = Image.open(os.path.join(script_directory, "GUI (Design)", "Camera.png"))
capture_button_image = capture_button_image.resize((185, 165), Image.Resampling.LANCZOS)
capture_button_image = ImageTk.PhotoImage(capture_button_image)

# Create the button with the image
capture_button = Button(detection_frame, image=capture_button_image, command=capture_image, bd=0, highlightthickness=0, activebackground="black")
capture_button.image = capture_button_image
capture_button.place(x=257, y=478)

# Load the image for the back button
left_button_image = Image.open(os.path.join(script_directory, "GUI (Design)", "LeftArrow.png"))
left_button_image = left_button_image.resize((160, 160), Image.Resampling.LANCZOS)
left_button_image = ImageTk.PhotoImage(left_button_image)

# Create the button with the image
left_button = Button(detection_frame, image=left_button_image, command=back_to_main_menu, bd=0, highlightthickness=0, activebackground="black")
left_button.image = left_button_image
left_button.place(x=70, y=478)

# Load the image for the next button
right_button_image = Image.open(os.path.join(script_directory, "GUI (Design)", "RightArrow.png"))
right_button_image = right_button_image.resize((160, 160), Image.Resampling.LANCZOS)
right_button_image = ImageTk.PhotoImage(right_button_image)

# Create the "Next" button with the image
right_button = Button(detection_frame, image=right_button_image, command=last_frame, bd=0, highlightthickness=0, activebackground="black")
right_button.image = right_button_image
right_button.place(x=470, y=478)

# Background image setup for the last frame
last_background_image = Image.open(last_background_image_path)
last_background_photo = ImageTk.PhotoImage(last_background_image)

# Create Canvas for last frame
last_frame = tk.Canvas(root, width=700, height=700)
last_frame.create_image(0, 0, anchor=tk.NW, image=last_background_photo)

back_button = Button(root, text="B A C K", command=last_back, padx=10, pady=5, font=("Helvetica", 14), width=15, bg="light grey", fg="black", bd=5)

# Bind hover effects to the back button
back_button.bind("<Enter>", lambda event, button=back_button: on_enter(event, button))
back_button.bind("<Leave>", lambda event, button=back_button: on_leave(event, button))

update_feed()
root.mainloop()
