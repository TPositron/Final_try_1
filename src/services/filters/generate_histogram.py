# Moved from src/image_analysis/core/processors/filters/generate_histogram.py

def generate_histogram(image, bins=256):
    """
    Generate a histogram of the pixel intensities in the image.

    Parameters:
    - image: The input image for which the histogram will be calculated.
    - bins: The number of bins to use for the histogram. Default is 256.

    Returns:
    - histogram: A list of length `bins` containing the histogram values.
    """

    # ...existing code...