import cv2
import numpy as np

class FilterNonLocalMeans:
    def __init__(self, h=10, h_color=10, template_window_size=7, search_window_size=21):
        """
        Initialize Non-Local Means Denoising filter with configurable parameters.
        
        :param h: Filter strength for luminance (larger = more denoising but may lose details)
        :param h_color: Filter strength for color components
        :param template_window_size: Size of patch used for comparison (odd number)
        :param search_window_size: Size of area to search for similar patches (odd number)
        """
        self.h = h
        self.h_color = h_color
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def __call__(self, image):
        """
        Apply Non-Local Means denoising to the input image.
        
        :param image: Input image (grayscale or color)
        :return: Denoised image
        """
        if len(image.shape) == 2:
            # Grayscale image
            return cv2.fastNlMeansDenoising(
                image,
                h=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
        else:
            # Color image
            return cv2.fastNlMeansDenoisingColored(
                image,
                h=self.h,
                hColor=self.h_color,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )