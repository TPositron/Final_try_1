import numpy as np
import cv2

def apply_fft_highpass(image, radius):
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    
    # FFT and shift
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # Create mask
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)
    
    # Apply mask and inverse FFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return np.uint8(img_back)

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        filtered = apply_fft_highpass(test_img, 30)
        cv2.imwrite("highpass.png", filtered)