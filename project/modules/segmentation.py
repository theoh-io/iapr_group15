import cv2
import numpy as np
import skimage.morphology as morph


def segment(image):
    """
    Segments the puzzle pieces from the background of an input image.

    Args:
        image: A numpy array representing the input image.

    Returns:
        A binary image where the puzzle pieces are represented by 1 and the background is represented by 0.
    """
    kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    thresholded_bis = cv2.adaptiveThreshold(
        sharpened_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=3,
        C=-4
    )

    thresholded_bis = morph.remove_small_holes(thresholded_bis.astype(bool), area_threshold=1000, connectivity=3).astype(np.uint8)
    thresholded_bis[thresholded_bis > 0] = 255
    
    kernel = np.ones((9,9),np.uint8)
    thresholded_bis_closed = cv2.morphologyEx(thresholded_bis, cv2.MORPH_CLOSE, kernel)
    
    return thresholded_bis_closed.astype(np.uint8)