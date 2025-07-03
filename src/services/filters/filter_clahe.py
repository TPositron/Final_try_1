import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image.

    :param image: Input grayscale image.
    :param clip_limit: Threshold for contrast limiting.
    :param tile_grid_size: Size of grid for histogram equalization.
    :return: Image after applying CLAHE.
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced_image = clahe.apply(image)
    
    return enhanced_image

def filter_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Filter an image using CLAHE and save the result to a file.

    :param image_path: Path to the input image file.
    :param output_path: Path to save the filtered image.
    :param clip_limit: Threshold for contrast limiting.
    :param tile_grid_size: Size of grid for histogram equalization.
    """
    # Read the input image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE to the image
    enhanced_image = apply_clahe(image, clip_limit, tile_grid_size)

    # Save the filtered image to the output path
    cv2.imwrite(output_path, enhanced_image)