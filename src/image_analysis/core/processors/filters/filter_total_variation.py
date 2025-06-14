import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle

def apply_total_variation(image, weight):
    denoised = denoise_tv_chambolle(image, weight=weight)
    denoised_uint8 = (denoised * 255).astype(np.uint8)
    return denoised_uint8

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        denoised = apply_total_variation(test_img, 0.2)
        cv2.imwrite("tv_denoised.png", denoised)