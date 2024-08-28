import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(original, transformed, title1="Original", title2="Transformed"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.show()

def image_negative(image):
    negative = 255 - image
    display_images(image, negative, "Original", "Negative")
    return negative

def log_transformation(image, base=2):
    c = 255 / (np.log(1 + np.max(image)) / np.log(base))
    log_transformed = c * (np.log(1 + image) / np.log(base))
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    display_images(image, log_transformed, "Original", "Log Transformation")
    return log_transformed

def gamma_correction(image, gamma=1.0):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    display_images(image, gamma_corrected, "Original", f"Gamma Correction (γ={gamma})")
    return gamma_corrected

def contrast_stretching(image, min_val, max_val):
    stretched = np.clip(((image - np.min(image)) * ((max_val - min_val) / (np.max(image) - np.min(image))) + min_val), min_val, max_val)
    stretched = np.array(stretched, dtype=np.uint8)
    display_images(image, stretched, "Original", "Contrast Stretching")
    return stretched

def gray_level_slicing(image, min_val, max_val):
    sliced = np.zeros(image.shape, dtype=np.uint8)
    sliced[(image >= min_val) & (image <= max_val)] = 255
    display_images(image, sliced, "Original", "Gray-Level Slicing")
    return sliced

# 5. Bit-Plane Slicing
def bit_plane_slicing(image, bit_plane):
    sliced = np.bitwise_and(image, 1 << bit_plane)
    sliced = np.where(sliced > 0, 255, 0)
    display_images(image, sliced, "Original", f"Bit-Plane Slicing (Plane {bit_plane})")
    return sliced

def main():
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)
    
    image_negative(image)
    
    log_transformation(image, base=2)
    
    gamma_correction(image, gamma=2.0)
    
    contrast_stretching(image, min_val=50, max_val=200)
    
    gray_level_slicing(image, min_val=100, max_val=150)
    
    bit_plane_slicing(image, bit_plane=4)

if __name__ == "__main__":
    main()
