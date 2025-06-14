import cv2
import numpy as np

def apply_gabor(image, frequency, theta, sigma_x, sigma_y):
    kernel = cv2.getGaborKernel(
        (0, 0), sigma_x, theta, frequency, sigma_y, 0, ktype=cv2.CV_32F
    )
    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return (filtered, kernel)

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        filtered, kernel = apply_gabor(test_img, 0.1, np.pi/4, 5, 5)
        cv2.imwrite("gabor_filtered.png", filtered)
        cv2.imwrite("gabor_kernel.png", kernel * 255)