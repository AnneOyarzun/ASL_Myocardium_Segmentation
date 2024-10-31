import numpy as np
import os 
import cv2 as cv
from skimage.exposure import rescale_intensity

def specific_intensity_window_1(image, window_percent=0.1):
    image = image.astype('int64') 
    arr = np.asarray_chkfinite(image)
    min_val = arr.min()
    number_of_bins = arr.max() - min_val + 1
    hist = np.bincount((arr-min_val).ravel(), minlength=number_of_bins)
    hist_new = hist[1:]
    total = np.sum(hist_new)
    window_low = window_percent * total
    window_high = (1 - window_percent) * total
    cdf = np.cumsum(hist_new)
    low_intense = np.where(cdf >= window_low) + min_val
    high_intense = np.where(cdf >= window_high) + min_val
    res = rescale_intensity(image, in_range=(low_intense[0][0], high_intense[0][0]),out_range=(arr.min(), arr.max()))
    return res

def dice(im1, im2):
   
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def contour_size(contour):
    return contour.shape[0]

def apply_mask(image, mask):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if mask[x,y]==0:
                image[x,y]=0
    return image
