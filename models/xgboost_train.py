"""
xgboost_train.py

Train an XGBoost model for stroke lesion segmentation
using handcrafted image features extracted from DWI scans.

Author: Jasmine Wang Thye Wei
Date: 2025-10-12
"""

# ----------------------------
# Required Libraries
# ----------------------------
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import xgboost as xgb


# ----------------------------
# Define Paths (edit as needed)
# ----------------------------
image_dir = "path_to_your_images"     
mask_dir = "path_to_your_masks"       

# ----------------------------
# Extract Features
# ----------------------------
all_features_df = pd.DataFrame()

image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii")))
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii")))

for img_file, mask_file in zip(image_files, mask_files):
    img = nib.load(img_file).get_fdata()
    mask = nib.load(mask_file).get_fdata()

    for i in range(img.shape[2]):
        slice_img = img[:, :, i]
        slice_mask = mask[:, :, i]

        if np.sum(slice_mask) == 0:
            continue

        features_df = extract_features(slice_img)
        features_df["Labels"] = slice_mask.reshape(-1)
        all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)

# ----------------------------
# Prepare Data
# ----------------------------
all_features_df = all_features_df[all_features_df["Original Image"] != 0]
X = all_features_df.drop(columns=["Labels"])
y = all_features_df["Labels"].values

# ----------------------------
# Train XGBoost
# ----------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)

# ----------------------------
# Save Model (pickle)
# ----------------------------
save_dir = "./trained_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "xgboost_model.sav")

with open(model_path, "wb") as f:
    pickle.dump(model, f)


