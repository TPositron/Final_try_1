"""
Generate Histogram - Image Histogram Generation Utility

This module provides histogram generation functionality for image analysis,
calculating pixel intensity distributions for visualization and analysis.

Main Function:
- generate_histogram(): Generates histogram of pixel intensities

Dependencies:
- Used by: Image processing and analysis components
- Used by: Filter services for histogram-based operations

Features:
- Configurable bin count for histogram resolution
- Pixel intensity distribution calculation
- Support for grayscale image analysis
"""

def generate_histogram(image, bins=256):
    """
    Generate a histogram of the pixel intensities in the image.

    Parameters:
    - image: The input image for which the histogram will be calculated.
    - bins: The number of bins to use for the histogram. Default is 256.

    Returns:
    - histogram: A list of length `bins` containing the histogram values.
    """

import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_histogram(image, show_histogram=False):
    if not show_histogram:
        return
    
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    plt.figure()
    plt.title("Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.grid(True)
    
    stats = {
        'min': np.min(image),
        'max': np.max(image),
        'mean': np.mean(image),
        'std': np.std(image)
    }
    
    plt.show(block=False)
    return stats

if __name__ == "__main__":
    test_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    stats = generate_histogram(test_img, show_histogram=True)
    print("Image Stats:", stats)