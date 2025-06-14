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