from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf


def image_matting(image_path, trimap):
    image = image_path[:, :, ::-1]  # Convert BGR to RGB

    if image.shape[:2] != trimap.shape[:2]:
        raise ValueError("Input image and trimap must have same size")

    # Perform robust image matting using PyMatting's alpha estimation function
    alpha = estimate_alpha_cf(image / 255.0, trimap)

    return alpha
