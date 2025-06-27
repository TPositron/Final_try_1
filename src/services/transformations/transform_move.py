# Moved from src/image_analysis/core/processors/transforms/transform_move.py

class TransformMove:
    def __init__(self, x_offset, y_offset):
        self.x_offset = x_offset
        self.y_offset = y_offset

    def apply(self, image):
        # Logic to move the image
        pass