import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(image, reference_image):
    hist_image = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_ref = cv2.calcHist([reference_image], [0], None, [256], [0, 256])
    hist_image = hist_image.ravel()
    hist_ref = hist_ref.ravel()

    cdf_image = hist_image.cumsum()
    cdf_ref = hist_ref.cumsum()
    cdf_image_normalized = cdf_image / cdf_image[-1]
    cdf_ref_normalized = cdf_ref / cdf_ref[-1]

    lookup_table = np.interp(cdf_image_normalized, cdf_ref_normalized, np.arange(256))
    lookup_table = np.floor(lookup_table).astype(np.uint8)

    matched_image = lookup_table[image]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Original Image')

    plt.subplot(1, 2, 2)
    plt.hist(matched_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Matched Image')

    plt.show()

    return matched_image

if __name__ == "__main__":
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread('sample_image1.jpeg', cv2.IMREAD_GRAYSCALE)
    
    matched_image = histogram_matching(image, reference_image)
