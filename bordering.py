import cv2
import numpy as np
from medfilt2 import medfilt2


def bordering(gray, image):
    # apply binary thresholding
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    KERNEL_SIZE = 3
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # visualize the binary image
    # cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)

    MIN_CONTOUR_SIZE = 20
    contours2 = []
    for i in range(len(contours)):
        if contours[i].shape[0] > MIN_CONTOUR_SIZE:
            contours2.append(contours[i])

    image_copy2 = np.zeros(image.shape)
    cv2.drawContours(image=image_copy2, contours=contours2[1:-1], contourIdx=-1, color=(255, 255, 255), thickness=3,
                     lineType=cv2.LINE_AA)
    # image_copy2 = cv2.erode(image_copy2, kernel, iterations=1)
    # image_copy2 = cv2.dilate(image_copy2, kernel, iterations=1)
    # cv2.imshow('REDUCED CONTOURS', image_copy2)
    # cv2.waitKey(0)

    PIXEL_COL_THRESHOLD = 8
    PIXEL_ROW_THRESHOLD = 10
    mat = np.ones(image[:, :, 0].shape)
    for i in range(len(image_copy2[0, :, 0])):
        if sum(image_copy2[:, i, 0] / 255) > PIXEL_COL_THRESHOLD:
            mat[:, i] = mat[:, i] * 2
        if i > 5:
            if mat[0, i - 5] == 2:
                if sum(mat[0, i - 5: i]) < PIXEL_ROW_THRESHOLD:
                    mat[:, i - 5] = mat[:, i - 5] - 1

    for i in range(len(image_copy2[:, 0, 0])):
        if sum(image_copy2[i, :, 0] / 255) < PIXEL_COL_THRESHOLD:
            mat[i, :] = np.ones(mat[i, :].shape)

    # Find the coordinates of the 1s in the matrix
    rows, cols = np.where(mat == 2)

    # Find the minimum and maximum row and column indices
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    # Define the rectangle corners
    top_left = (min_col, min_row)
    bottom_right = (max_col, max_row)

    mat = np.zeros(mat.shape)
    cv2.rectangle(mat, top_left, bottom_right, 1, thickness=cv2.FILLED)
    mat = mat * (image_copy2[:, :, 0] / 255) + mat

    s_mat = mat[min_row: max_row + 1, min_col: max_col + 1]
    for i in range(len(s_mat)):
        j = 0
        while s_mat[i, j] == 1:
            s_mat[i, j] = 0
            j += 1
            if j == len(s_mat[0]):
                break
        j = len(s_mat[0]) - 1
        while s_mat[i, j] == 1:
            s_mat[i, j] = 0
            j -= 1

    for j in range(len(s_mat[0])):
        i = 0
        k = True
        while s_mat[i, j] == 0:
            i += 1
            if i == len(s_mat):
                k = False
                break
        while (np.sum(s_mat[i: len(s_mat), j].astype(np.uint8)) > 0) & k:
            s_mat[i, j] = 1
            i += 1
            if i == len(s_mat):
                break

    mat[min_row: max_row + 1, min_col: max_col + 1] = s_mat

    # cv2.imshow('Test MATRIX', mat.astype(np.uint8) * 255)
    # cv2.waitKey(0)

    DILATATION_PERCENTAGE = 0.025
    surface = np.sqrt((max_row - min_row + 1) * (max_col - min_col + 1))
    mat = cv2.morphologyEx(mat, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                (np.uint8(DILATATION_PERCENTAGE * surface), np.uint8(DILATATION_PERCENTAGE * surface))), iterations=3)
    # cv2.imshow('Test MATRIX DILATED', mat.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    return mat
