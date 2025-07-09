

import cv2
import numpy as np

def filter_canny(image, low_threshold, high_threshold):
    """
    Applies the Canny filter to the given image.

    :param image: Input image.
    :param low_threshold: Low threshold for the hysteresis procedure.
    :param high_threshold: High threshold for the hysteresis procedure.
    :return: Image with Canny filter applied.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny filter
    edges = cv2.Canny(image, low_threshold, high_threshold)

    return edges