"""
intensity_standardization.py

Author: Jasmine Wang Thye Wei
Date: 2025-10-12

Intensity Standardization using Nyúl's Histogram Matching Method
---------------------------------------------------------------

This script standardizes the intensity distribution of diffusion-weighted imaging (DWI)
scans using a modified implementation of Nyúl’s histogram normalization method.
This process improves consistency across datasets from different scanners or centers,
enhancing the generalizability of stroke lesion segmentation models.

Original implementation adapted from:
https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/nyul.py

References:
1. Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000).
   New variants of a method of MRI scale standardization.
   IEEE Transactions on Medical Imaging, 19(2), 143–150.
2. Shah, M. et al. (2011).
   Evaluating intensity normalization on MRIs of human brain with multiple sclerosis.
   Medical Image Analysis, 15(2), 267–282.

Modifications in this version:
- Applied specifically to DWI brain scans.
- Standardization restricted to foreground voxels (non-zero mask regions).
- Used PCHIP interpolation for smooth histogram mapping.
- Added structured directory setup and reproducible outputs.
"""

# ============================================================
# Import Libraries
# ============================================================
import os
import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d, PchipInterpolator

# ============================================================
# Core Functions
# ============================================================

def get_landmarks(volume, percentiles):
    """Compute percentile landmarks of the input volume."""
    return np.percentile(volume, percentiles)


def nyul_train_standard_scale(
    volume_fns,
    mask_fns=None,
    i_min=0,
    i_max=100,
    i_s_min=1,
    i_s_max=4095,
    l_percentile=10,
    u_percentile=90,
    step=10,
):
    """
    Train the Nyúl standard scale from multiple DWI volumes.

    Parameters
    ----------
    volume_fns : list of str
        Paths to input DWI NIfTI files.
    mask_fns : list of str, optional
        Corresponding brain masks to limit standardization to the foreground.
    i_min, i_max : float
        Minimum and maximum percentile cutoffs.
    i_s_min, i_s_max : float
        Standardized intensity range.
    l_percentile, u_percentile : float
        Lower and upper percentile limits for landmarks.
    step : int
        Step size between percentile landmarks.

    Returns
    -------
    standard_scale : np.ndarray
        Averaged intensity landmarks defining the standard scale.
    percentiles : np.ndarray
        Percentile positions used to define landmarks.
    """

    percentiles = np.concatenate(([i_min], np.arange(l_percentile, u_percentile + 1, step), [i_max]))
    standard_scale = np.zeros(len(percentiles))
    landmarks_list = []

    for i, volume_fn in enumerate(volume_fns):
        print(f"[{i+1}/{len(volume_fns)}] Processing: {os.path.basename(volume_fn)}")

        volume_data = nib.load(volume_fn).get_fdata()
        mask_data = nib.load(mask_fns[i]).get_fdata() if mask_fns is not None else None

        # Extract only foreground voxels
        data = volume_data[mask_data > 0] if mask_data is not None else volume_data.flatten()

        landmarks = get_landmarks(data, percentiles)
        min_p, max_p = np.percentile(data, [i_min, i_max])

        # Normalize each subject’s landmarks to standard range
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        scaled_landmarks = f(landmarks)

        landmarks_list.append(scaled_landmarks)
        standard_scale += scaled_landmarks

    # Compute mean standard scale across subjects
    standard_scale /= len(volume_fns)

    return standard_scale, percentiles


def do_hist_normalization(volume, landmark_percs, standard_scale, mask=None):
    """Apply histogram normalization using PCHIP interpolation."""
    data = volume[mask > 0] if mask is not None else volume.flatten()
    landmarks = get_landmarks(data, landmark_percs)

    f = PchipInterpolator(landmarks, standard_scale)
    normalized_volume = f(volume.flatten()).reshape(volume.shape)
    return normalized_volume


def nyul_apply_standard_scale(input_volume, standard_hist_path, input_mask=None):
    """Apply pre-trained standard scale to a new DWI volume."""
    standard_scale, percentiles = np.load(standard_hist_path, allow_pickle=True)
    normalized_volume = do_hist_normalization(input_volume, percentiles, standard_scale, mask=input_mask)
    return normalized_volume

# ============================================================
# Example Usage (edit paths before running)
# ============================================================
if __name__ == "__main__":

    # Define directories
    volume_dir = "./data/dwi_images"
    mask_dir = "./data/masks"
    output_dir = "./data/standardized"
    standard_hist_path = "./data/standard_histogram.npy"

    os.makedirs(output_dir, exist_ok=True)

    # Collect file lists
    volume_files = sorted([os.path.join(volume_dir, f) for f in os.listdir(volume_dir) if f.endswith(".nii.gz")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

    # -------------------------------
    # Train standard scale
    # -------------------------------
    standard_scale, percentiles = nyul_train_standard_scale(volume_files, mask_fns=mask_files)

    np.save(standard_hist_path, [standard_scale, percentiles])
   

    # -------------------------------
    # Apply to each volume
    # -------------------------------
    for volume_fn, mask_fn in zip(volume_files, mask_files):
        vol = nib.load(volume_fn).get_fdata()
        mask = nib.load(mask_fn).get_fdata()

        normalized_vol = nyul_apply_standard_scale(vol, standard_hist_path, input_mask=mask)

        output_path = os.path.join(output_dir, os.path.basename(volume_fn))
        nib.save(nib.Nifti1Image(normalized_vol, affine=np.eye(4)), output_path)
      
