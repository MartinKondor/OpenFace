import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity


face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('resources/haarcascade_profileface.xml')


"""
:returns: preprocessed image
"""
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


"""
Checks if the two objects represent the same face or not

:face, saved_face: np.ndarray
:returns: bool
"""
def is_the_same_face(face1, face2):    
    score, diff = structural_similarity(face1, face2, full=True)
    diff = (diff * 255).astype("uint8")
    # print(diff)
    return False


"""
Checks for an existing face and if found returns the face's name

:x, y, w, h: x, y face coordinates, w - width, h - height
:returns: str, the name of the recognized face
"""
def check_face(face, frame):
    saved_faces = [(np.load(os.path.join('faces', file)), file) for file in os.listdir('faces') if file.split('.')[-1] == 'npy']
    face = preprocess_face_img(face, frame)
    
    for saved_face, saved_face_name in saved_faces:
        if is_the_same_face(face, saved_face):
            return saved_face_name

    return 'Unknown'


"""
:frame: np.ndarray
"""
def detect_faces(frame, flip=False):
    if flip:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = profile_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in faces:  # Draw rectangles around faces
        
        # Check for existing face
        name = check_face((x,y,w,h), frame)

        cv2.putText(frame, name, (x, y - 5), cv2.cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0,))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    """
    faces2 = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in faces2:
        cv2.putText(frame, 'Unrecognized', (x, y - 5), cv2.cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255,))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    """

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


"""
:face, frame: np.ndarray
"""
def show_face(face, frame):
    face_img = preprocess_face_img(face, frame)
    cv2.imshow("face", face_img)


"""
:returns: preprocessed image
"""
def save_face(face, frame, name='face'):
    face_img = preprocess_face_img(face, frame)

    with open('faces/{}.npy'.format(name), 'wb') as f:
        np.save(f, face_img)

    return face_img
