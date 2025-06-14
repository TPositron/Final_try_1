import cv2
import numpy as np

def apply_dog(image, sigma1, sigma2):
    blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    dog = blur1 - blur2
    
    dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dog_normalized

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        dog_img = apply_dog(test_img, 1.5, 3.0)
        cv2.imwrite("dog_result.png", dog_img)