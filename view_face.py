import os
import cv2
import numpy as np
from argparse import ArgumentParser


args = ArgumentParser()
args.add_argument('image', metavar='i', type=str, help='path to the image to show')

if __name__ == '__main__':
    arg = args.parse_args()
    if arg.image:
        face = np.load(os.path.join('faces', arg.image))
        cv2.imshow("face", face)
        cv2.waitKey()
