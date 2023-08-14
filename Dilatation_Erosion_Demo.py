import cv2
import cv2.gapi
import numpy as np


# Demonstration of the use of cv2 Morphological functions, these function are based on FOR loops,
# browsing the 1.000.000+ pixels of the images, therefore are too slow for computational uses. In the next part
# of the projects cv2 equivalents are used. Function based on these articles:
# https://en.wikipedia.org/wiki/Erosion_(morphology)
# https://en.wikipedia.org/wiki/Dilation_(morphology)
# https://en.wikipedia.org/wiki/Closing_(morphology)
# https://en.wikipedia.org/wiki/Opening_(morphology)
# https://en.wikipedia.org/wiki/Top-hat_transform

def loc_min(size, img, x, y):
    img = np.concatenate((img, np.ones((len(img), size), dtype=np.uint8) * 255), axis=1)
    img = np.concatenate((np.ones((len(img), size), dtype=np.uint8) * 255, img), axis=1)
    img = np.concatenate((img, np.ones((size, len(img[0])), dtype=np.uint8) * 255), axis=0)
    img = np.concatenate((np.ones((size, len(img[0])), dtype=np.uint8) * 255, img), axis=0)
    left = (x + size) - int((size - 1) / 2)
    right = (x + size) + int((size - 1) / 2) + 1
    bottom = (y + size) + int((size - 1) / 2) + 1
    top = (y + size) - int((size - 1) / 2)
    tempL = []
    tempM = img[left: right, top: bottom]
    for x in range(int((size - 1) / 2) + 1):
        for y in range(x + 1):
            tempL.append(tempM[(x, int((size - 1) / 2) + y)])
            tempL.append(tempM[(size - x - 1, int((size - 1) / 2) + y)])
            tempL.append(tempM[(x, int((size - 1) / 2) - y)])
            tempL.append(tempM[(size - x - 1, int((size - 1) / 2) - y)])

    return np.min(tempL)


def loc_max(size, img, x, y):
    img = np.concatenate((img, np.ones((len(img), size), dtype=np.uint8) * 0), axis=1)
    img = np.concatenate((np.ones((len(img), size), dtype=np.uint8) * 0, img), axis=1)
    img = np.concatenate((img, np.ones((size, len(img[0])), dtype=np.uint8) * 0), axis=0)
    img = np.concatenate((np.ones((size, len(img[0])), dtype=np.uint8) * 0, img), axis=0)
    left = (x + size) - int((size - 1) / 2)
    right = (x + size) + int((size - 1) / 2) + 1
    bottom = (y + size) + int((size - 1) / 2) + 1
    top = (y + size) - int((size - 1) / 2)
    tempL = []
    tempM = img[left: right, top: bottom]
    for x in range(int((size - 1) / 2) + 1):
        for y in range(x + 1):
            tempL.append(tempM[(x, int((size - 1) / 2) + y)])
            tempL.append(tempM[(size - x - 1, int((size - 1) / 2) + y)])
            tempL.append(tempM[(x, int((size - 1) / 2) - y)])
            tempL.append(tempM[(size - x - 1, int((size - 1) / 2) - y)])

    return np.max(tempL)


def dilatation(img_input, img_output, size):
    i = 0
    print("--------------------------")
    print("--  BEGIN DILATATION    --")
    print("--------------------------")
    for x in range(len(img_output)):
        for y in range(len(img_output[0])):
            img_output[x, y] = loc_min(size, img_input, x, y)
            if i % 2000 == 0:
                print(i, "th pixel")
                print(x, "th line")
            i = i + 1
    print("--------------------------")
    print("--   END DILATATION     --")
    print("--------------------------")
    return img_output


def erosion(img_input, img_output, size):
    i = 0
    print("--------------------------")
    print("--    BEGIN EROSION     --")
    print("--------------------------")
    for x in range(len(img_output)):
        for y in range(len(img_output[0])):
            img_output[x, y] = loc_max(size, img_input, x, y)
            if i % 2000 == 0:
                print(i, "th pixel")
                print(x, "th line")
            i = i + 1
    print("--------------------------")
    print("--     END EROSION      --")
    print("--------------------------")
    return img_output


def computation(inputM, size):
    grayTemp1 = cv2.cvtColor(inputM, cv2.COLOR_BGR2GRAY)
    grayTemp2 = cv2.cvtColor(inputM, cv2.COLOR_BGR2GRAY)
    grayOut = cv2.cvtColor(inputM, cv2.COLOR_BGR2GRAY)
    # opening operation and top-hat transform
    openImg = inputM - dilatation(erosion(inputM, grayTemp1, size), grayTemp2, size)
    # closing operation and black-hat transform
    closeImg = erosion(dilatation(inputM, grayTemp1, size), grayTemp2, size) - inputM
    for x in range(len(grayOut)):
        for y in range(len(grayOut[0])):
            grayOut[x, y] = max(openImg[x, y], closeImg[x, y])
    return grayOut
