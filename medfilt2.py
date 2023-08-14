import cv2


def medfilt2(image, kernel_size):
    # Add padding to the image
    padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                      kernel_size // 2, cv2.BORDER_REFLECT)
    # Apply median filtering
    filtered_image = cv2.medianBlur(padded_image, kernel_size)

    # Get the dimensions of the original image
    h, w = image.shape[:2]

    # Remove padding from the filtered image to get the result with the original size
    filtered_image = filtered_image[kernel_size // 2:h + kernel_size // 2, kernel_size // 2:w + kernel_size // 2]

    return filtered_image
