import numpy as np


def delete_row_and_column(img, num):
    for i in range(num):
        img = np.delete(img, obj=0, axis=0)
        img = np.delete(img, obj=0, axis=1)

    return img


def trimap_conversion(trimap, n):
    if n == 0:
        out = trimap[:, :, 0] / 255.0 + trimap[:, :, 2] / (2 * 255.0)
    if n == 1:
        out = trimap[:, :, 1] / 255.0 + trimap[:, :, 2] / (2 * 255.0)
    return out
