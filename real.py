import cv2
import os
import numpy as np

def train_model(dataset_path="dataset/"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_dict = {}
    
    person_folders = ["sam", "taylor", "me"]  # Explicitly define folder names
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
    
    faces = np.array(faces, dtype="object")
    labels = np.array(labels)
    
    recognizer.train(faces, labels)
    recognizer.save("face_trained.yml")
    np.save("labels.npy", label_dict)
    
    print("Training complete. Model saved as 'face_trained.yml'")

def recognize_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_trained.yml")
    label_dict = np.load("labels.npy", allow_pickle=True).item()
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
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
                name = label_dict[label] if confidence < 100 else "Unknown"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except:
                cv2.putText(frame, "Error in recognition", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("camera", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit if 'ESC' is pressed
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_model()
    recognize_faces()