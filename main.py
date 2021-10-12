import cv2
import time
import numpy as np

from src.detect import detect_face, save_face, show_face
from src.learn import learn_face



def print_hello():
    start_msg = '| OpenFace - Open source face recognizer |'
    print('-'*len(start_msg))
    print('|' + ' '*(len(start_msg) - 3) + ' |')
    
    print(start_msg)
    
    print('|' + ' '*(len(start_msg) - 3) + ' |')
    print('-'*len(start_msg))
    
    print('Press "Ctrl+C" to exit.')
    print()


vc = cv2.VideoCapture(0)
if vc.isOpened():
    rval, frame = vc.read()
    print_hello()
else:
    print('Cannot open the camera')
    rval = False


while rval:
    rval, frame = vc.read()

    # Detect face
    frame, face = detect_face(frame)

    if face is not None:
        show_face(face, frame)
        cv2.waitKey()

        ans = input('[?] Is your face clearly visible on the shown image? (y or n)\n')
        if ans != 'y':
            continue
        
        name = input('[?] How should I call this face?\n')
        face = save_face(face, frame, name)
        
        # Learn the saved face
        print('Leanring the {}\'s face ...'.format(name))
        learn_face(face)
        break

    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
