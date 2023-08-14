import argparse
import time
import cv2.gapi
import numpy as np

from image_matting import image_matting
from skel import skel
from medfilt2 import medfilt2
import util
from bordering import bordering

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input image")
ap.add_argument("-j", "--image2", type=str, required=True,
                help="path to input image")
args = vars(ap.parse_args())

image1 = cv2.imread(args["image"])
image2 = cv2.imread(args["image2"])
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
output1 = np.zeros(gray1.shape, dtype=np.uint8)
output2 = np.zeros(gray2.shape, dtype=np.uint8)

cv2.imshow("Image 1", image1)
cv2.waitKey(0)
cv2.imshow("Image 2", image2)
cv2.waitKey(0)


# Compute gradients using Sobel filters
gradient_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)

# Compute the magnitude of the gradient
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Display the gradient magnitude
cv2.imshow("Grad", gradient_magnitude.astype(np.uint8))
cv2.waitKey(0)


start_time = time.time()
NUMBER_OF_ITERATION = 100
for i in range(1, NUMBER_OF_ITERATION):
    scale = 3 * i + 1
    filterSize = (scale, scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    g1 = gray1 - cv2.morphologyEx(gray1, cv2.MORPH_OPEN, kernel)
    g2 = cv2.morphologyEx(gray1, cv2.MORPH_CLOSE, kernel) - gray1
    g3 = gray2 - cv2.morphologyEx(gray2, cv2.MORPH_OPEN, kernel)
    g4 = cv2.morphologyEx(gray2, cv2.MORPH_CLOSE, kernel) - gray2
    g_1 = np.maximum(g1, g2)
    g_2 = np.maximum(g3, g4)
    output1 = output1 + g_1 / scale
    output2 = output2 + g_2 / scale

output1 = np.round(output1, 0).astype(np.uint8)
output2 = np.round(output2, 0).astype(np.uint8)

cv2.imshow("Output", output1)
cv2.waitKey(0)
cv2.imshow("Output 2", output2)
cv2.waitKey(0)

contour1 = bordering(output1, image1).astype(np.uint8)
contour2 = bordering(output2, image2).astype(np.uint8)

test = np.zeros(image1.shape).astype(np.uint8)
test[:, :, 0] = contour1 * 127
test[:, :, 1] = contour2 * 127

cv2.imshow("Border superposed", test)
cv2.waitKey(0)

FILTER_SIZE = 7
r1 = np.uint8(output1 >= output2)
r1 = medfilt2(r1, FILTER_SIZE)
r2 = np.uint8(output2 >= output1)
r2 = medfilt2(r2, FILTER_SIZE)

cv2.imshow("R1", r1 * 255)
cv2.waitKey(0)
cv2.imshow("R2", r2 * 255)
cv2.waitKey(0)

rms1 = skel(r1, 5).astype(np.uint8) * 255
rms2 = skel(r2, 5).astype(np.uint8) * 255

FILTER_SIZE = 3
rms1 = medfilt2(rms1, FILTER_SIZE)
rms2 = medfilt2(rms2, FILTER_SIZE)


cv2.imshow("Skeleton 1", rms1)
cv2.waitKey(0)
cv2.imshow("Skeleton 2", rms2)
cv2.waitKey(0)

THRESHOLD = 70
pd1 = (output1 - output2 > THRESHOLD).astype(np.uint8)
pd2 = (output2 - output1 > THRESHOLD).astype(np.uint8)

cv2.imshow("pd1", pd1 * 255)
cv2.waitKey(0)
cv2.imshow("pd2", pd2 * 255)
cv2.waitKey(0)


rd2 = np.logical_or(rms1, pd1).astype(np.uint8)
rd1 = np.logical_or(rms2, pd2).astype(np.uint8)

triMap = np.zeros(r1.shape + (3,)).astype(np.uint8)
common = rd1 & rd2
common_contour = contour1 & contour2

cv2.imshow("common contour", common_contour * (r1 - common) * 255)
cv2.waitKey(0)

print("Common Pixels:", np.sum(common), ", R1 focused pixels:", np.sum(rd1), ", R2 focused pixels:", np.sum(rd2))
print("Summ R1 + R2 pixels:", np.sum(rd1) + np.sum(rd2), ", Total Pixels:", len(rd1[0]) * len(rd1))

triMap[:, :, 0] = ((rd1 - common) - common_contour - (((rd1 - common) - common_contour) & contour2)) * 255  # BLUE
triMap[:, :, 1] = ((rd2 - common) - common_contour - (((rd2 - common) - common_contour) & contour1)) * 255  # GREEN
triMap[:, :, 2] = (common_contour + (common & np.bitwise_not(common_contour)) + (((rd2 - common) - common_contour)
                                & contour1) + (((rd1 - common) - common_contour) & contour2)) * 255  # RED

# triMap[:, :, 0] = (rd1 - common) * 255  # BLUE
# triMap[:, :, 1] = (rd2 - common) * 255  # GREEN
# triMap[:, :, 2] = common * 255  # RED

cv2.imshow("Trimap", triMap)
cv2.waitKey(0)


alpha1 = image_matting(image1, util.trimap_conversion(triMap, 0))
alpha2 = image_matting(image2, util.trimap_conversion(triMap, 1))
cv2.imshow("B&W Trimap", np.uint8(util.trimap_conversion(triMap, 0) * 255))
cv2.waitKey(0)
cv2.imshow("Alpha 1", np.uint8(alpha1 * 255))
cv2.waitKey(0)
cv2.imshow("Alpha 2", np.uint8(alpha2 * 255))
cv2.waitKey(0)

threshold = 0.4
alpha_decision = np.uint8((alpha1 >= threshold) & (alpha2 <= threshold))
cv2.imshow("Alpha Decision", np.uint8(alpha_decision) * 255)
cv2.waitKey(0)

img_out_color = np.zeros(image1.shape, dtype=np.uint8)
for i in range(3):
    img_out_color[:, :, i] = image1[:, :, i] * alpha_decision + image2[:, :, i] * \
                             (np.bitwise_not(alpha_decision.astype(bool)).astype(np.uint8))

# img_out_color_filtered = np.zeros(((medfilt2(img_out_color[:, :, i], 5)).shape + (3,)), dtype=np.uint8)
# for i in range(3):
#     img_out_color_filtered[:, :, i] = medfilt2(img_out_color[:, :, i], 5)

print("Elapsed time:", time.time() - start_time)

cv2.imshow("Output Colored", img_out_color)
cv2.imwrite('results/focus_stacking_2.JPG', img_out_color)
cv2.waitKey(0)
# cv2.imshow("Output Colored Filtered", img_out_color_filtered)
# cv2.imwrite('with bitwise_not/output_colored_range5_filtered.JPG', img_out_color_filtered)
# cv2.waitKey(0)
