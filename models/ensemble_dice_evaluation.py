"""
ensemble_dice_evaluation.py

Computes Dice coefficient for ensemble predictions from
multiple trained models (Random Forest, XGBoost, or EfficientNet-B2 U-Net).

Author: Jasmine Wang Thye Wei
Date: 2025-10-12
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
from tensorflow.keras.models import load_model


# ---------------------------------------------------
# Dice Coefficient Function
# ---------------------------------------------------
def compute_dice(im1, im2, empty_value=1.0):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch between input masks.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / im_sum


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
image_dir = "./data/test/images"
mask_dir = "./data/test/masks"
model_dir = "./trained_models"
output_dir = "./results"
model_type = "dl"   # set "ml" for Random Forest/XGBoost, or "dl" for EfficientNet-B2 U-Net

os.makedirs(output_dir, exist_ok=True)

image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))


# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
models = []

if model_type == "ml":
    model_paths = [os.path.join(model_dir, f"RF{i}add.sav") for i in range(1, 6)]
    models = [pickle.load(open(p, "rb")) for p in model_paths]
    print(f"Loaded {len(models)} machine learning models (.sav).")

elif model_type == "dl":
    model_paths = [os.path.join(model_dir, f"EfficientNetB2_UNet_fold{i}.keras") for i in range(1, 6)]
    models = [load_model(p, compile=False) for p in model_paths]
    print(f"Loaded {len(models)} deep learning models (.keras).")

else:
    raise ValueError("Invalid model_type. Choose 'ml' or 'dl'.")


# ---------------------------------------------------
# Ensemble Prediction and Dice Evaluation
# ---------------------------------------------------
dice_scores = []

for img_path, mask_path in zip(image_files, mask_files):
    print(f"Processing {os.path.basename(img_path)}...")

    img = nib.load(img_path).get_fdata()
    gt_mask = nib.load(mask_path).get_fdata()

    if model_type == "ml":
        # Handcrafted features for ML models
        features = extract_features(img)
        preds = [model.predict(features).reshape(img.shape) for model in models]

    elif model_type == "dl":
        # Expand dimensions for DL model input
        input_img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
        preds = [model.predict(input_img, verbose=0)[0, :, :, 0] for model in models]

    # Ensemble (average and threshold)
    ensemble_pred = np.mean(preds, axis=0)
    binary_pred = (ensemble_pred >= 0.5).astype(int)

    dice = compute_dice(gt_mask.flatten(), binary_pred.flatten())
    dice_scores.append(dice)

# ---------------------------------------------------
# Save Results
# ---------------------------------------------------
mean_dice = np.mean(dice_scores)
dice_df = pd.DataFrame({
    "Image_File": [os.path.basename(f) for f in image_files],
    "Dice_Score": dice_scores
})
dice_df.loc[len(dice_df)] = ["Mean Dice", mean_dice]

excel_path = os.path.join(output_dir, "ensemble_dice_scores.xlsx")
dice_df.to_excel(excel_path, index=False)

