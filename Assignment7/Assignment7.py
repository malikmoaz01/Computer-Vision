import numpy as np
import cv2

def edge_detection(image, method):
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")
    
    if method == 'sobel':
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
        edge_image = np.sqrt(grad_x**2 + grad_y**2)

    elif method == 'prewitt':
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        grad_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
        edge_image = np.sqrt(grad_x**2 + grad_y**2)

    elif method == 'roberts':
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        grad_x = cv2.filter2D(image, cv2.CV_64F, roberts_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, roberts_y)
        edge_image = np.sqrt(grad_x**2 + grad_y**2)

    else:
        raise ValueError("Invalid method. Choose 'sobel', 'prewitt', or 'roberts'.")

    edge_image = (edge_image / np.max(edge_image)) * 255
    edge_image = edge_image.astype(np.uint8)
    
    return edge_image

if __name__ == "__main__":
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)
    result = edge_detection(image, 'sobel')
    cv2.imwrite('edge_detected_image.png', result)
    cv2.imshow('Edge Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
