from PIL import Image

# Moved from src/image_analysis/core/processors/filters/filter_laplacian.py

def laplacian_filter(image):
    """
    Apply Laplacian filter to the given image.

    :param image: Input image
    :return: Image after applying Laplacian filter
    """
    # Kernel for Laplacian filter
    kernel = [[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]]

    # Get the dimensions of the image
    width, height = image.size

    # Create a new image to store the result
    new_image = Image.new("L", (width, height))

    # Apply the Laplacian filter
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            # Get the 3x3 region of the image around the current pixel
            region = image.crop((x - 1, y - 1, x + 2, y + 2))

            # Apply the kernel to the region
            new_pixel_value = sum(
                region.getpixel((i, j)) * kernel[j][i]
                for i in range(3) for j in range(3)
            )

            # Set the new pixel value in the new image
            new_image.putpixel((x, y), int(new_pixel_value))

    return new_image