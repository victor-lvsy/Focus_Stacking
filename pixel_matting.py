import cv2
import numpy as np

from medfilt2 import medfilt2


def pixel_matting(size, map, x, y):
    left = x - int((size - 1) / 2)
    right = x + int((size - 1) / 2) + 1
    bottom = y + int((size - 1) / 2) + 1
    top = y - int((size - 1) / 2)
    if left < 0:
        left = 0
    if right > len(map) + 1:
        right = len(map) + 1
    if top < 0:
        top = 0
    if bottom > len(map[0]) + 1:
        bottom = len(map[0]) + 1

    if np.sum(map[left: right, top: bottom, 0]) >= np.sum(map[left: right, top: bottom, 1]):
        return 1

    return 0


def neighbor_matting(triMap):
    matting_map = np.zeros(triMap[:, :, 0].shape)
    for x in range(len(triMap)):
        for y in range(len(triMap[0])):
            if triMap[x, y, 2] == 255:
                matting_map[x, y] = pixel_matting(3, triMap, x, y)

    cv2.imshow("Matting Map", matting_map * 255)

    triMap[:, :, 0] += matting_map.astype(np.uint8) * 255
    triMap[:, :, 1] += np.bitwise_not(matting_map.astype(bool)).astype(np.uint8) * 255 - (
            (triMap[:, :, 1] / 255).astype(np.uint8) & np.bitwise_not(matting_map.astype(bool)).astype(np.uint8))
    triMap[:, :, 2] = np.zeros(triMap[:, :, 2].shape)

    cv2.imshow("Bi Map", triMap)
    biMap = medfilt2((triMap[:, :, 0] / 255).astype(np.uint8), 7)
