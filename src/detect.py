import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('resources/haarcascade_profileface.xml')


"""
:frame: np.ndarray
"""
def detect_faces(frame, flip=False):
    if flip:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = profile_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in faces:  # Draw rectangles around faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    faces2 = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in faces2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame, faces

    
"""
:frame: np.ndarray
"""
def detect_face(frame, flip=False):
    if flip:
        frame = cv2.flip(frame, 1)

    # Preprocessing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Finding face
    faces = profile_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    face = None

    if len(faces) != 0:
        # Select the biggest face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3])[-1]
        face = x, y, w, h

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return frame, face


def preprocess_face_img(face, frame):
    x, y, w, h = face

    # Minimum image sizes
    if w < 100:
        w = 100
    if h < 100:
        h = 100

    # 1. Crop face from image
    # 2. Resize for fixed size
    return cv2.resize(frame[y:y+h, x:x+w], (250, 250,))


def show_face(face, frame):
    face_img = preprocess_face_img(face, frame)
    cv2.imshow("face", face_img)


def save_face(face, frame, name='face'):
    face_img = preprocess_face_img(face, frame)

    with open('faces/{}.npy'.format(name), 'wb') as f:
        np.save(f, face_img)

    return face_img
