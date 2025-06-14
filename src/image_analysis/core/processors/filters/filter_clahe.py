import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image.
    Args:
        image: Input grayscale image (numpy array)
        clip_limit: Threshold for contrast limiting (float)
        tile_grid_size: Size of grid for histogram equalization (int or tuple)
    Returns:
        CLAHE enhanced image (numpy array)
    """
    if isinstance(tile_grid_size, int):
        tile_grid_size = (tile_grid_size, tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        clahe_img = apply_clahe(test_img, 2.0, 8)
        cv2.imwrite("clahe_result.png", clahe_img)
