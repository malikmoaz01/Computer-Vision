import cv2
import numpy as np

def image_derivatives(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_x = np.absolute(grad_x)
    grad_x = np.uint8(255 * grad_x / np.max(grad_x))
    cv2.imshow('First Derivative in X-direction', grad_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_y = np.absolute(grad_y)
    grad_y = np.uint8(255 * grad_y / np.max(grad_y))
    cv2.imshow('First Derivative in Y-direction', grad_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_xy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    grad_xy = np.absolute(grad_xy)
    grad_xy = np.uint8(255 * grad_xy / np.max(grad_xy))
    cv2.imshow('First Derivative in Both Directions', grad_xy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
    grad_xx = np.absolute(grad_xx)
    grad_xx = np.uint8(255 * grad_xx / np.max(grad_xx))
    cv2.imshow('Second Derivative in X-direction', grad_xx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
    grad_yy = np.absolute(grad_yy)
    grad_yy = np.uint8(255 * grad_yy / np.max(grad_yy))
    cv2.imshow('Second Derivative in Y-direction', grad_yy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(255 * laplacian / np.max(laplacian))
    cv2.imshow('Laplacian', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sharpen_image(image):
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow('Blurred', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(255 * laplacian / np.max(laplacian))
    sharpened_image = cv2.addWeighted(blurred_image, 1, laplacian, 0.5, 0)
    cv2.imshow('Sharpened', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def unsharp_masking(image):
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow('Blurred', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)
    cv2.imshow('Unsharp Masked', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def high_boost_filtering(image):
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow('Blurred', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    high_boost_image = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)
    cv2.imshow('High-Boost Filtered', high_boost_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image = cv2.imread('sample_image.webp', cv2.IMREAD_GRAYSCALE)
    image_derivatives(image)
    sharpen_image(image)
    unsharp_masking(image)
    high_boost_filtering(image)



if __name__ == "__main__":
    main()