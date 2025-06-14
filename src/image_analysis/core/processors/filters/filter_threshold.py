import cv2

def apply_threshold(image, threshold_value, method='binary'):
    methods = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV,
        'tozero': cv2.THRESH_TOZERO,
        'tozero_inv': cv2.THRESH_TOZERO_INV,
        'trunc': cv2.THRESH_TRUNC
    }
    
    if method not in methods:
        raise ValueError(f"Unsupported method: {method}")
    
    _, thresh = cv2.threshold(image, threshold_value, 255, methods[method])
    return thresh

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        binary = apply_threshold(test_img, 127, 'binary')
        cv2.imwrite("threshold_result.png", binary)