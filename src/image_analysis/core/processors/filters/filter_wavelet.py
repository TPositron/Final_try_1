import numpy as np
import cv2
import pywt

def apply_wavelet_edges(image, wavelet='db2', level=1):
    try:
        import pywt
        # Ensure image is float32 for wavelet processing
        img_float = image.astype(np.float32) / 255.0
        
        coeffs = pywt.wavedec2(img_float, wavelet, level=level)
        edges = np.zeros_like(img_float, dtype=np.float32)
        
        for i in range(1, min(level+1, len(coeffs))):
            if len(coeffs[i]) == 3:  # Ensure we have LH, HL, HH
                LH, HL, HH = coeffs[i]
                # Resize detail coefficients to match original image size
                LH_resized = cv2.resize(LH, (image.shape[1], image.shape[0]))
                HL_resized = cv2.resize(HL, (image.shape[1], image.shape[0]))
                HH_resized = cv2.resize(HH, (image.shape[1], image.shape[0]))
                
                edges += np.abs(LH_resized) + np.abs(HL_resized) + np.abs(HH_resized)
        
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        return edges.astype(np.uint8)
    except Exception as e:
        print(f"Wavelet filter error: {e}")
        # Fallback to Laplacian if wavelet fails
        return cv2.Laplacian(image, cv2.CV_8U, ksize=3)

if __name__ == "__main__":
    test_img = cv2.imread("test_gray.png", cv2.IMREAD_GRAYSCALE)
    if test_img is not None:
        edge_img = apply_wavelet_edges(test_img)
        cv2.imwrite("wavelet_edges.png", edge_img)