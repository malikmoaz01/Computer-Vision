from PIL import Image
import numpy as np
import os

try:
    image_path = r"C:\Users\Malik Moaz\Desktop\Computer Vision\img.jpg"
    image = Image.open(image_path)

    image_width, image_height = image.size
    image_color_channels = len(image.getbands())
    
    image_size = os.path.getsize(image_path)

    image_array = np.array(image)

    if image_color_channels == 1:
        intensity = np.sum(image_array)
        pixels = image_array.size
        image_pixel_intensity = intensity / pixels
    else:
        intensity = np.sum(image_array, axis=(0, 1))
        pixels = image_array.shape[0] * image_array.shape[1]
        image_pixel_intensity = intensity / pixels

    image_type = image.mode

    print("The dimensions of image are:", image_width, "X", image_height)
    print("The colour channels of image are:", image_color_channels, "(R, G, B)")
    print("The size of image is:", image_size, "Bytes")
    print("The Pixel intensity of Image is:", image_pixel_intensity)
    print("The image type is:", image_type)

except FileNotFoundError:
    print("File not found error: Check the path to the image")
