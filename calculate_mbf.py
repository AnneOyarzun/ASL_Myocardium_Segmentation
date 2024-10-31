import SimpleITK as sitk
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

def compute_tsnr(mean_values): 
    tsnr_value = np.nanmean(mean_values)/np.nanstd(mean_values)
    
    return tsnr_value

def compute_mean(img, mask):
    # Threshold the mask (convert values to 0 or 255)
    _, mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    masked_image = cv.bitwise_and(img, img, mask = mask)
    # Convert masked_image to float type to handle NaNs
    masked_image = masked_image.astype(np.float32)
    # Set masked pixels with value 0 to NaN
    masked_image[masked_image == 0] = np.nan

    return np.nanmean(masked_image.reshape(-1))

def calculate_pwi_serie(img_serie): 
    if (img_serie.shape[0] % 2) == 0: # es par
        control_idx = range(1, img_serie.shape[0], 2)
        label_idx = range(0, img_serie.shape[0], 2)
    else:
        control_idx = range(2, img_serie.shape[0], 2)
        label_idx = range(1, img_serie.shape[0], 2)
    
    pwi_serie = sitk.Image([img_serie.shape[2], img_serie.shape[1], len(control_idx)], sitk.sitkFloat64)
    pwi_serie_arr = sitk.GetArrayFromImage(pwi_serie)
    for pairs in range(0, len(control_idx)):
        pwi_serie_arr[pairs,:,:] = img_serie[control_idx[pairs],:,:] - img_serie[label_idx[pairs],:,:]

    return pwi_serie_arr


def calculate_tsnr(images, myomask_series): 
    control_idx = range(1, images.shape[0], 2)
    label_idx = range(0, images.shape[0], 2)
    
    tsnr_cortex_pre = []
    
    for pwi_imgs in range(0, len(control_idx)): 
        tsnr_cortex_pre.append(compute_mean(images[control_idx[pwi_imgs], :, :], myomask_series[control_idx[pwi_imgs], :, :]) - compute_mean(images[label_idx[pwi_imgs], :, :], myomask_series[label_idx[pwi_imgs], :, :]))

    # Post-process
    pos_threshold = np.nanmean(tsnr_cortex_pre) + (2 * np.nanstd(tsnr_cortex_pre))
    neg_threshold = np.nanmean(tsnr_cortex_pre) - (2 * np.nanstd(tsnr_cortex_pre))

    mean_filtered = tsnr_cortex_pre.copy()
    pwi_series = calculate_pwi_serie(images)
    filtered_image = pwi_series.copy()
    
    # List to store positions that meet the condition
    indices_to_delete = []
    for i in range(0, len(control_idx)): 
        if not neg_threshold < tsnr_cortex_pre[i] < pos_threshold:
            # mean_filtered[i] = np.nan
            # filtered_image[i,:,:] = np.nan
            indices_to_delete.append(i)

    
    # Delete elements based on the stored indices
    mean_filtered_del = np.delete(mean_filtered, indices_to_delete)
    filtered_image_del = np.delete(filtered_image, indices_to_delete, axis=0)

    mean_asl_value = np.nanmean(mean_filtered_del)
    tsnr = compute_tsnr(mean_filtered_del)
    filtered_pwi = np.nanmean(filtered_image_del, axis=0)

    # sitk.WriteImage(sitk.GetImageFromArray(filtered_pwi), pwi_path + 'filtered_pwi.nii')
    # sitk.WriteImage(sitk.GetImageFromArray(filtered_pwi_sd), pwi_path + 'filtered_pwiSD.nii')

    return abs(np.asarray(tsnr)), filtered_image_del, filtered_pwi, mean_asl_value

def calculate_mean_img(images): 
    arr = np.array(np.mean(images, axis=(0)))
    return arr

def calculate_mbfvalue(lambda_value, mean_asl, m0_signal, TI_value, T1_value): 
    mbf = ((lambda_value * mean_asl) / (2 * m0_signal * TI_value * np.exp((-TI_value/T1_value)))) * 60000
    return mbf

def erode_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv.erode(mask, kernel, iterations=iterations)
    return eroded_mask

def erode_ring_mask(mask, inner_radius=1, iterations=1):
    # Crear un kernel circular para la parte interna
    inner_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*inner_radius+1, 2*inner_radius+1))
    
    # Crear un kernel circular para la parte externa
    # outer_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*outer_radius+1, 2*outer_radius+1))
    
    # Erosionar la parte interna de la máscara
    eroded_mask = cv.erode(mask, inner_kernel, iterations=iterations)
    return eroded_mask

def load_series_to_numpy(folder, name, erode=False):
    series = []
    for root, _, files in os.walk(folder):
        for file in files:
            if name.lower() in file.lower():
                file_path = os.path.join(root, file)
                image = sitk.ReadImage(file_path)
                array = sitk.GetArrayFromImage(image)
                series.append(array)
    # Convert list of arrays into a single 3D array
    stacked_array = np.stack(series, axis=0)
    return stacked_array



###########################
if __name__ == '__main__': 
    
    erode = False
    # (1) Load ASL image series, masks (no hay que normalizar)
    images = 


    
    # (2) Load M0
    M0 = 
    gt_m0_mask = 

    # m0 signal
    m0_signal = compute_mean(M0, gt_m0_mask) 

    # Parameters for mbf
    lambda_value = 1 # ml/g
    TI_value = 1000 # ms 
    T1_value = 1434 # ms
    

    _, _, _, mean_asl = calculate_tsnr(images, gt_masks)
    
    mbf_value = calculate_mbfvalue(lambda_value, mean_asl, m0_signal, TI_value, T1_value)
    
    