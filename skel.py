import cv2
import numpy as np
from skimage.morphology import skeletonize


def skel(bw_image, iterations):
    skel_image = bw_image
    for i in range(iterations):
        skel_image = skeletonize(skel_image)

    return skel_image
