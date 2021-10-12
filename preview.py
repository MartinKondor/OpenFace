import cv2
import numpy as np
import time

from src.detect import detect_face


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)


 # Get the first frame
if vc.isOpened():
    rval, frame = vc.read()
    time.sleep(0.5)
else:
    print('Cannot open the camera')
    rval = False
    

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    # Detect face
    frame, face = detect_face(frame, flip=True)

    # Exit on ESC
    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
cv2.destroyWindow("preview")
