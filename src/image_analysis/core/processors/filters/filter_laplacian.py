import cv2
import numpy as np

def apply_laplacian(image, ksize):
    laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=ksize)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    return abs_laplacian

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        edge_img = apply_laplacian(test_img, 3)
        cv2.imwrite("laplacian_edges.png", edge_img)