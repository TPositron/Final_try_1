# Moved from src/image_analysis/core/processors/filters/filter_clahe.py

import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    :param image: Input image (grayscale).
    :param clip_limit: Threshold for contrast limiting.
    :param tile_grid_size: Size of grid for histogram equalization.
    :return: Image after applying CLAHE.
    """
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel back with A and B channels
    merged_lab = cv2.merge((cl, a_channel, b_channel))

    # Convert the LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)

    return enhanced_image

def filter_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Filter an image using CLAHE and save the result to a file.

    :param image_path: Path to the input image file.
    :param output_path: Path to save the filtered image.
    :param clip_limit: Threshold for contrast limiting.
    :param tile_grid_size: Size of grid for histogram equalization.
    """
    # Read the input image
    image = cv2.imread(image_path)

    # Apply CLAHE to the image
    enhanced_image = apply_clahe(image, clip_limit, tile_grid_size)

    # Save the filtered image to the output path
    cv2.imwrite(output_path, enhanced_image)