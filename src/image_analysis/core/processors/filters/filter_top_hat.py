import cv2
import numpy as np

def apply_top_hat(image, kernel_size):
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be odd and >= 3")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        result = apply_top_hat(test_img, 5)
        cv2.imwrite("tophat_result.png", result)