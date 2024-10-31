import tensorflow as tf
import numpy as np
import os
from gettext import translation
import os
import sys
from tokenize import Imagnumber
from cv2 import rotate
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf
import shutil
import re
import matplotlib.pyplot as plt
import imgaug
import time
import pandas as pd
import nibabel as nib

# import cv2 



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Import Mask RCNN
from mrcnn.config import Config
from scipy.ndimage import binary_fill_holes
from mrcnn import utils
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn import visualize
from preprocesado import specific_intensity_window_1
from base import preprocessing
from base import eval_metrics

# Check if GPU is available
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("GPU is available")
else:
    print("GPU is not available")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def get_ax(rows=1, cols=1, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

############################################################
#  Configurations
############################################################

class MyoConfig(Config):
    """Configuration for training on the brain tumor dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'myo'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 # batch size
    NUM_CLASSES = 1 + 1  # background + myo
    DETECTION_MIN_CONFIDENCE = 0.85
    IMAGE_MAX_DIM = 128
    IMAGE_MIN_DIM = 128
    BACKBONE='resnet101'
    IMAGE_SHAPE = [128, 128, 3]
    STEPS_PER_EPOCH = 150
    DETECTION_MAX_INSTANCES = 1
    LEARNING_RATE = 0.001
    USE_MINI_MASK = False
    RPN_ANCHOR_RATIOS = [0.5,1,2]
    IMAGE_CHANNEL_COUNT = 3


class InferenceConfig(MyoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


############################################################
#  Dataset
############################################################
            
class MyoDataset(utils.Dataset):

    def load_scan(self, dataset_dir, subset):
        """Load a subset of the FarmCow dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("myo", 1, "myo")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset, 'Images')

        file_names = os.listdir(dataset_dir)
        for image_id in file_names:
            # print(image_id)
            image_path = os.path.join(dataset_dir, image_id)
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

            height, width = image.shape[1:]
            image = image.reshape((image.shape[1], image.shape[2], -1))

            self.add_image(
                "myo",
                image_id=image_id,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                subset = subset
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        mask = np.zeros([image_info["height"], image_info["width"],1])
        subset = image_info["subset"]
        path = os.path.join(DATASET_DIR, subset, 'Masks')
        files_names = os.listdir(path)[image_id]
        mask_path = os.path.join(path, files_names)
        mask_img = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask_transposed = np.transpose(mask_img, (1,2,0))


        return mask_transposed.astype(np.uint8), np.ones([mask.shape[-1]], dtype=np.int32)
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "kidney":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


            
def encode_mask(mask):
    """Encode binary mask using Run-Length Encoding (RLE)."""
    mask_flat = mask.flatten(order='F')  # Flatten the mask in Fortran order
    mask_diff = np.diff(mask_flat)  # Find differences between consecutive elements
    run_starts = np.where(mask_diff == 1)[0] + 2  # Add 2 to shift index by 1
    run_lengths = np.where(mask_diff == -1)[0] - run_starts + 1

    # Combine run-length data into a list of [start, length] pairs
    run_length_data = list(zip(run_starts.tolist(), run_lengths.tolist()))

    return run_length_data

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": encode_mask(mask)
            }
            results.append(result)
    return results


def evaluate(model, dataset, output_masks_folder):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = dataset.image_ids

    t_prediction = 0
    t_start = time.time()

    dice_scores = []
    precision_scores = []
    fpr_scores = []
    fdr_scores = []
    sensi_scores = []
    espec_scores = []

    results_df = pd.DataFrame(columns=['ImageName', 'DiceScore', 'BoundingBox', 'PredictionTime'])
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        image_rescaled = preprocessing.specific_intensity_window(image, window_percent=0.15)

        gt_mask = dataset.load_mask(image_id)[0]

        # Run detection
        t = time.time()
        r = model.detect([image_rescaled], verbose=0)[0]
        t_prediction += (time.time() - t)

        filename = os.path.splitext(dataset.image_info[image_id]['id'])[0]
        print(filename)
        
        # Calculate Dice score
        pred_mask = r["masks"][:,:,0].astype(bool) # it sometimes extract two instances
        dice = eval_metrics.calculate_dice(gt_mask[:,:,0], pred_mask)
        fpr = eval_metrics.false_positive_rate(gt_mask[:,:,0], pred_mask)
        # fdr = eval_metrics.false_discovery_rate(gt_mask[:,:,0], pred_mask)
        sensi = eval_metrics.calculate_sensitivity(gt_mask[:,:,0], pred_mask)
        espec = eval_metrics.calculate_specificity(gt_mask[:,:,0], pred_mask)

        dice_scores.append(dice)
        fpr_scores.append(fpr)
        # fdr_scores.append(fdr)
        sensi_scores.append(sensi)
        espec_scores.append(espec)

        # Save results to DataFrame
        results_df = results_df.append({
                                        'ImageName': filename,
                                        'BoundingBox': r['rois'][0],
                                        'PredictionTime': t_prediction,
                                        'DiceScore': dice,
                                        'FalsePositiveRate': fpr, 
                                        # 'FalseDiscoveryRate': fdr, 
                                        'Sensibility': sensi, 
                                        'Especificity': espec, 
                                        }, ignore_index=True)

        # Save predicted mask
        mask_filename = f"{filename}_pred_mask.nii"
        mask_filepath = os.path.join(output_masks_folder, mask_filename)
        sitk.WriteImage(sitk.GetImageFromArray(pred_mask.astype(np.uint8) * 255), mask_filepath)

    return results_df


def evaluate_uncertainty(model, dataset, output_masks_folder, num_mc_samples=10):
    """Runs official COCO evaluation.
    dataset: A Dataset object with validation data
    output_masks_folder: Folder to save output masks
    num_mc_samples: Number of Monte Carlo samples for uncertainty estimation
    """
    image_ids = dataset.image_ids

    t_start = time.time()

    results_df = pd.DataFrame(columns=['ImageName', 'DiceScore', 'DiceScoreStd',
                                       'FPR', 'FPRStd', 'Sensitivity', 'SensitivityStd',
                                       'Specificity', 'SpecificityStd', 'PredictionTime'])

    for i, image_id in enumerate(image_ids):
        print('Evaluating image nº:', i)
        image = dataset.load_image(image_id)
        image_rescaled = preprocessing.specific_intensity_window(image, window_percent=0.15)
        gt_mask = dataset.load_mask(image_id)[0]

        # Perform multiple forward passes with dropout enabled
        dice_scores = []
        fpr_scores = []
        sensi_scores = []
        espec_scores = []
        mc_pred_masks = []

        
        for step in range(num_mc_samples):
            # print('Step nº: ', step)
            t_prediction = time.time()
            r = model.detect([image_rescaled], verbose=0)[0] # training=True to enable dropout
            t_prediction = time.time() - t_prediction
            # print('Prediction took:', t_prediction)

            # Calculate Dice score
            pred_mask = r["masks"][:, :, 0].astype(bool)
            dice = eval_metrics.calculate_dice(gt_mask[:, :, 0], pred_mask)
            # print('Dice score: ', dice)
            dice_scores.append(dice)

            # Calculate False Positive Rate
            fpr = eval_metrics.false_positive_rate(gt_mask[:, :, 0], pred_mask)
            fpr_scores.append(fpr)

            # Calculate Sensitivity
            sensi = eval_metrics.calculate_sensitivity(gt_mask[:, :, 0], pred_mask)
            sensi_scores.append(sensi)

            # Calculate Specificity
            espec = eval_metrics.calculate_specificity(gt_mask[:, :, 0], pred_mask)
            espec_scores.append(espec)
            
            mc_pred_masks.append(pred_mask)

        ## Convert mc_pred_masks list to a numpy array
        #mc_pred_masks_array = np.array(mc_pred_masks)
        #sitk.WriteImage(sitk.GetImageFromArray(mc_pred_masks_array), output_masks_folder + 'segmentation.nii')
        
        # Compute mean and standard deviation of scores
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)

        mean_fpr = np.mean(fpr_scores)
        std_fpr = np.std(fpr_scores)

        mean_sensi = np.mean(sensi_scores)
        std_sensi = np.std(sensi_scores)

        mean_espec = np.mean(espec_scores)
        std_espec = np.std(espec_scores)
        
        # Compute mean and standard deviation of predicted masks
        mc_pred_masks = np.stack(mc_pred_masks, axis=-1)

        # Compute mean and standard deviation of segmentation masks
        mean_seg_mask = (np.sum(mc_pred_masks, axis=-1) / num_mc_samples) 
        filename = os.path.splitext(dataset.image_info[image_id]['id'])[0]
        mean_seg_mask_filename = f"{filename}_mean_seg_mask.nii"
        mean_seg_mask_filepath = os.path.join(output_masks_folder, mean_seg_mask_filename)  
        sitk.WriteImage(sitk.GetImageFromArray(mean_seg_mask), mean_seg_mask_filepath)

        std_seg_mask = np.std(mc_pred_masks, axis=-1)

        std_seg_mask_entera = np.round(std_seg_mask * 10).astype(np.uint8) # guardar la máscara como enteros entre 0-10


        # Save mean segmentation mask
        
        filtered_mean_seg = mean_seg_mask
        filtered_mean_seg[mean_seg_mask >= 0.95] = 1
        filtered_mean_seg[mean_seg_mask < 0.95] = 0
        filt_mean_seg_mask_filename = f"{filename}_FILTmean_seg_mask.nii"
        os.makedirs(output_masks_folder + 'filt/', exist_ok=True)
        filt_mean_seg_mask_filepath = os.path.join(output_masks_folder + 'filt/', filt_mean_seg_mask_filename)

        sitk.WriteImage(sitk.GetImageFromArray(filtered_mean_seg), filt_mean_seg_mask_filepath)

        # Save standard deviation of segmentation mask
        std_seg_mask_filename = f"{filename}_std_seg_mask.nii"
        std_seg_mask_filepath = os.path.join(output_masks_folder, std_seg_mask_filename)
        sitk.WriteImage(sitk.GetImageFromArray(std_seg_mask_entera), std_seg_mask_filepath)
    


        # Save results to DataFrame
        results_df = results_df.append({
            'ImageName': filename,
            'BoundingBox': r['rois'][0],
            'PredictionTime': t_prediction,
            'DiceScore': mean_dice,
            'DiceScoreStd': std_dice,
            'FPR': mean_fpr,
            'FPRStd': std_fpr,
            'Sensitivity': mean_sensi,
            'SensitivityStd': std_sensi,
            'Specificity': mean_espec,
            'SpecificityStd': std_espec
        }, ignore_index=True)

    return results_df



if __name__ == '__main__': 
     ####################################################
    ########### GENERAL SETTINGS #######################
    ####################################################

    DATASET_DIR = 
    DEFAULT_LOGS_DIR = 

    mode = 'inference'
    dataset_test = MyoDataset()
    dataset_test.load_scan(DATASET_DIR, 'test')
    dataset_test.prepare()

    config = InferenceConfig()
    model = modellib.MaskRCNN(
        mode=mode,
        config=config,
        model_dir=DEFAULT_LOGS_DIR, 
        apply_dropout=True
    )

    model_dir = 
    weights_path = 

    print('Loading weights...')
    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name = True)

    print('Evaluating results...')
    result_dir = model_dir + 'results/'

    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)
        os.makedirs(result_dir + 'seg/')
    
    results_df = evaluate_uncertainty(model, dataset_test, result_dir + 'seg/', num_mc_samples=100)
    results_df.to_excel(result_dir + 'evaluation_results.xlsx', index=False)
