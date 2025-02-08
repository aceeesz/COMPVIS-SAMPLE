import numpy as np
import os
import sys
import cv2


def read_images(path, sz=None):
    c = 0
    X, y = [], []

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == "directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    # Resize image
                    if sz is not None:
                        im = cv2.resize(im, (200, 200))

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)

                except IOError as e:
                    print("I/O Error")
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c += 1  # Increment label count outside the loop
    return [X, y]


def face_rec():
    names = ['wick', 'taylor', 'me']

    pathing = "C:/Users/chris/Downloads/PROJ/dataset"

    [X, y] = read_images(pathing)
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y)

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                params = model.predict(roi)
                label = names[params[0]]
                if label == "wick":
                    cv2.putText(img, label + " : " + str(params[1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif label == "taylor":
                    cv2.putText(img, "Not him " + str(params[1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            except:
                continue

        cv2.imshow("camera", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Check if 'q' or 'Esc' is pressed
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_rec()
