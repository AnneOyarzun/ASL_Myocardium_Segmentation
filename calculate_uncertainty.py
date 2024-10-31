import numpy as np


gt_mask = 
uncert_mask = 


# Calculate the sum of values of all pixels in the uncertainty map normalized by the volume of predicted masks (pixels of GT)
uncert_pixesl = np.count_nonzero(uncert_mask)
gt_pixels = np.count_nonzero(gt_mask)

mcd_uncertainty = (uncert_pixesl / gt_pixels) * 100








