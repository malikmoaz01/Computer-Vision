import cv2
import numpy as np

def canny_edge_detection(image, low_threshold, high_threshold):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = cv2.filter2D(blurred_image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(blurred_image, cv2.CV_64F, sobel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    direction = np.abs(direction) % 180
          
    suppressed_image = np.zeros_like(magnitude, dtype=np.uint8)
    
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            angle = direction[i, j]
            q, r = 255, 255
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = magnitude[i + 1, j + 1]
                r = magnitude[i - 1, j - 1]
            elif 67.5 <= angle < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]
            
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed_image[i, j] = magnitude[i, j]
            else:
                suppressed_image[i, j] = 0
    
    strong_edges = (suppressed_image > high_threshold)
    weak_edges = ((suppressed_image >= low_threshold) & (suppressed_image <= high_threshold))
    
    final_edges = np.zeros_like(suppressed_image, dtype=np.uint8)
    
    strong_row, strong_col = np.nonzero(strong_edges)
    for r, c in zip(strong_row, strong_col):
        if final_edges[r, c] == 0:
            final_edges[r, c] = 255
            for i in range(r-1, r+2):
                for j in range(c-1, c+2):
                    if weak_edges[i, j]:
                        final_edges[i, j] = 255
    
    return final_edges

if __name__ == "__main__":
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)
    edges = canny_edge_detection(image, 50, 150)
    cv2.imwrite('sample_image.webp', edges)
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
