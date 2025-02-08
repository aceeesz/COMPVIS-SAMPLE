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
person_folders = ["wick", "taylor", "me"]  # Explicitly define the folder names
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
