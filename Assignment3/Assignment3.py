import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel()

    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    cdf_mapped = np.floor(cdf_normalized * 255).astype(np.uint8)

    equalized_image = cdf_mapped[image]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Original Image')

    plt.subplot(1, 2, 2)
    plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Equalized Image')

    plt.show()

    return equalized_image

if __name__ == "__main__":
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)
    equalized_image = histogram_equalization(image)
