import cv2
import os
import numpy as np

# Path where the dataset is stored
dataset_path = "dataset/"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_dict = {}

# Assign a label (number) to each person
person_folders = ["sam", "taylor", "me"]  # Explicitly define the folder names
for label, person_name in enumerate(person_folders):
    label_dict[label] = person_name
    person_path = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(label)

# Convert to NumPy array
faces = np.array(faces, dtype="object")
labels = np.array(labels)

# Train the model
recognizer.train(faces, labels)
recognizer.save("face_trained.yml")

# Save label dictionary
np.save("labels.npy", label_dict)

print("Training complete. Model saved as 'face_trained.yml'")

# Load trained model and label dictionary
recognizer.read("face_trained.yml")
label_dict = np.load("labels.npy", allow_pickle=True).item()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        try:
            label, confidence = recognizer.predict(face_roi)
            
            if confidence < 100:  # Lower confidence = better match
                name = label_dict[label]
                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces
            
            # Draw rectangle and display name
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except:
            cv2.putText(frame, "Error in recognition", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit if 'ESC' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
