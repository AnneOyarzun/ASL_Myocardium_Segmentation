import numpy as np

# y_true: gt_mask
# y_pred: predicted_mask

def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    dice = 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice

def precision(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    predicted_positives = np.sum(y_pred)
    return true_positives / (predicted_positives + 1e-10)  # Avoid division by zero

def false_positive_rate(y_true, y_pred):
    false_positives = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    actual_negatives = np.sum(np.logical_not(y_true))
    return false_positives / (actual_negatives + 1e-10)  # Avoid division by zero

def false_discovery_rate(y_true, y_pred):
    false_positives = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    return false_positives / (true_positives + false_positives + 1e-10)  # Avoid division by zero

def recall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    actual_positives = np.sum(y_true)
    return true_positives / (actual_positives + 1e-10)  # Avoid division by zero

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-10)  # Avoid division by zero

def calculate_sensitivity(predicted_mask, ground_truth_mask):
    # Convert masks to binary arrays (1 for foreground, 0 for background)
    predicted_mask_binary = (predicted_mask > 0).astype(np.int)
    ground_truth_mask_binary = (ground_truth_mask > 0).astype(np.int)
    
    # Calculate True Positive (TP) and False Negative (FN)
    TP = np.sum(np.logical_and(predicted_mask_binary == 1, ground_truth_mask_binary == 1))
    FN = np.sum(np.logical_and(predicted_mask_binary == 0, ground_truth_mask_binary == 1))
    
    # Calculate Sensitivity (Recall)
    sensitivity = TP / (TP + FN)
    
    return sensitivity

def calculate_specificity(predicted_mask, ground_truth_mask):
    # Convert masks to binary arrays (1 for foreground, 0 for background)
    predicted_mask_binary = (predicted_mask > 0).astype(np.int)
    ground_truth_mask_binary = (ground_truth_mask > 0).astype(np.int)
    
    # Calculate True Negative (TN) and False Positive (FP)
    TN = np.sum(np.logical_and(predicted_mask_binary == 0, ground_truth_mask_binary == 0))
    FP = np.sum(np.logical_and(predicted_mask_binary == 1, ground_truth_mask_binary == 0))
    
    # Calculate Specificity
    specificity = TN / (TN + FP)
    
    return specificity