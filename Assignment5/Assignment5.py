import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(original, filtered, title1="Original", title2="Filtered"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.show()

def averaging_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(image, -1, kernel)
    display_images(image, filtered_image, "Original", f"Averaging Filter ({kernel_size}x{kernel_size})")
    return filtered_image

def weighted_average_filter(image, kernel):
    kernel_sum = np.sum(kernel)
    kernel = kernel / kernel_sum
    filtered_image = cv2.filter2D(image, -1, kernel)
    display_images(image, filtered_image, "Original", "Weighted Average Filter")
    return filtered_image

def median_filter(image, kernel_size):
    filtered_image = cv2.medianBlur(image, kernel_size)
    display_images(image, filtered_image, "Original", f"Median Filter ({kernel_size}x{kernel_size})")
    return filtered_image

def weighted_median_filter(image, kernel):
    pad_size = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            region = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            region_flat = region.flatten()
            weights_flat = kernel.flatten()
            weighted_values = np.multiply(region_flat, weights_flat)
            median_value = np.median(weighted_values)
            filtered_image[i - pad_size, j - pad_size] = median_value

    display_images(image, filtered_image, "Original", "Weighted Median Filter")
    return filtered_image

def max_filter(image, kernel_size):
    filtered_image = cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))
    display_images(image, filtered_image, "Original", f"Max Filter ({kernel_size}x{kernel_size})")
    return filtered_image

def min_filter(image, kernel_size):
    filtered_image = cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))
    display_images(image, filtered_image, "Original", f"Min Filter ({kernel_size}x{kernel_size})")
    return filtered_image

if __name__ == "__main__":
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)

    averaging_filter(image, kernel_size=3)

    weight_matrix = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    weighted_average_filter(image, weight_matrix)

    median_filter(image, kernel_size=3)

    weighted_median_filter(image, weight_matrix)

    max_filter(image, kernel_size=3)

    min_filter(image, kernel_size=3)
