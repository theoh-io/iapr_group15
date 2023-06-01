import cv2
import numpy as np
from .piece import Piece


def get_rectangles(img, segmented, height=128, width=128):
    # Find contours
    contours = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    target_area = height*width
    max_err = 0.15     # + ou - 10%

    # Compute rotated rectangle (minimum area)
    rectangles = []
    mask = np.zeros_like(img)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box)
        if (area > (1-max_err)*target_area) and (area < (1+max_err)*target_area): 
            rectangles.append(rect)
            cv2.drawContours(mask, [box.astype(int)], 0, (255, 255, 255), thickness=cv2.FILLED)
        
    return rectangles, mask


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))
    
    padded = np.zeros((diag, diag, 3), dtype=image.dtype)
    
    yoff = round((diag-h)/2)
    xoff = round((diag-w)/2)
    padded[yoff:yoff+h, xoff:xoff+w, :] = image.copy()
    
    center = (diag / 2, diag / 2)
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    rotated_image = cv2.warpAffine(padded, M, (diag,diag))
    return rotated_image


def crop_image(image, rect):
    box = np.int0(cv2.boxPoints(rect))
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    x1 = max(0, x_min)
    y1 = max(0, y_min)
    x2 = min(image.shape[1], x_max)
    y2 = min(image.shape[0], y_max)
    cropped = image[y1:y2, x1:x2]
    return cropped


def get_piece(image, rect, height=128, width=128):
    # Crop image 
    data = crop_image(image, rect)
    
    # Rotate piece
    angle = rect[2]
    data = rotate_image(data, angle)
    
    # Crop rotated piece
    center = (int(data.shape[0]/2), int(data.shape[1]/2))
    x1 = center[1] - int(width/2)
    y1 = center[0] - int(height/2)
    x2 = center[1] + int(width/2)
    y2 = center[0] + int(height/2)
    data = data[y1:y2, x1:x2]
    
    piece = Piece(data=data)

    return piece


def check_has_nan(array):
    # Check if the array contains NaN values
    has_nan = np.isnan(array).any()
    if has_nan:
        return True
    else:
        return False