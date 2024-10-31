import numpy as np
import SimpleITK as sitk
import numpy as np
import os 
import cv2 as cv
from skimage.exposure import rescale_intensity


def image_padding(img, new_shape):
    
    # Calcula la cantidad de píxeles de relleno en cada lado del eje x
    fill_x = max(0, new_shape[1] - img.shape[2]) // 2

    # Aplica el relleno de ceros utilizando numpy
    padded_img = np.pad(img, ((0, 0), (0, 0), (fill_x, fill_x)), mode='constant', constant_values=0)

    return padded_img


def specific_intensity_window(image, window_percent=0.1):
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
