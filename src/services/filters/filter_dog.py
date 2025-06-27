# Moved from src/image_analysis/core/processors/filters/filter_dog.py

class DoG:
    """
    Difference of Gaussians (DoG) filter class.
    """

    def __init__(self, sigma1: float, sigma2: float):
        """
        Initialize the DoG filter with two sigma values.

        :param sigma1: Standard deviation of the first Gaussian.
        :param sigma2: Standard deviation of the second Gaussian.
        """
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply(self, image):
        """
        Apply the Difference of Gaussians filter to the input image.

        :param image: Input image to be filtered.
        :return: Filtered image.
        """
        from scipy.ndimage import gaussian_filter

        # Apply Gaussian filters with the two sigma values
        gaussian1 = gaussian_filter(image, sigma=self.sigma1)
        gaussian2 = gaussian_filter(image, sigma=self.sigma2)

        # Return the difference of the two Gaussian filters
        return gaussian1 - gaussian2