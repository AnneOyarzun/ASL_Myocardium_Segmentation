import cv2 as cv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt



def orientation_detection(img): 
    # Apply a Gaussian blur to the image to remove noise
    gray = cv.GaussianBlur((img*255).astype(np.uint8), (5, 5), 0)
    # Apply Canny edge detection to detect edges in the image
    edges = cv.Canny((gray*255).astype(np.uint8), 50, 100)

    # Find contours in the image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)

    # Iterate through each contour and determine whether it corresponds to a right or left mask
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)
        
        # Determine the center of the bounding box
        cx = x + w // 2
        cy = y + h // 2
        
        # If the center of the bounding box is on the left side of the image, it is a left mask; otherwise, it is a right mask
        # Right
        if cx < img.shape[1] // 2:
            mask[y:y+h, x:x+w] = 1
        # Left
        else:
            mask[y:y+h, x:x+w] = 2
    return mask



def label_right_left(img): 
    '''
    It receives a sitk format image and returns an array with relabeled mask. Right mask = label 1. Left mask = label 2. 
    '''
    mask_right_total = np.zeros_like(img)
    mask_left_total = np.zeros_like(img)

    if len(img.shape) == 2: # 2d image
        img_unique = img
        mask = orientation_detection(img_unique)

        # Relabel original mask
        mask_right = np.logical_and(img_unique > 0, mask == 1)
        mask_left = np.logical_and(img_unique > 0, mask == 2)

        mask_right_total = mask_right
        mask_left_total = mask_left


    else:
        for i in range(0, img.shape[0]): 
            img_unique= img[i,:,:]
            mask = orientation_detection(img_unique)

            # Relabel original mask
            mask_right = np.logical_and(img_unique > 0, mask == 1)
            mask_left = np.logical_and(img_unique > 0, mask == 2)

            # mask_total[i, :, :] = mask_dual
            mask_right_total[i:] = mask_right
            mask_left_total[i:] = mask_left

    return mask_right_total.astype(np.uint8), mask_left_total.astype(np.uint8)
