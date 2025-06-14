import cv2

def apply_canny(image, low_threshold, high_threshold, aperture_size=3):
    """
    Apply Canny edge detection to a grayscale image.
    
    Args:
        image: Input grayscale image (numpy array)
        low_threshold: Lower hysteresis threshold (int)
        high_threshold: Upper hysteresis threshold (int)
        aperture_size: Size of Sobel kernel (odd int, default=3)
        
    Returns:
        Binary edge image (numpy array)
    """
    edges = cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)
    return edges

if __name__ == "__main__":
    # Test case
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        edges = apply_canny(test_img, 50, 150)
        cv2.imwrite("edges.png", edges)