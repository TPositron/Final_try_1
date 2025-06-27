from PIL import Image

# Moved from src/image_analysis/core/processors/transforms/transform_rotate.py

class ImageRotator:
    def __init__(self, angle):
        self.angle = angle

    def rotate(self, image):
        """Rotate the image by the given angle."""
        rotated_image = image.rotate(self.angle)
        return rotated_image