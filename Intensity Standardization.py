"""

Intensity Standardization using Nyul's Histogram Matching Method

This script implements intensity standardization for diffusion-weighted imaging (DWI) 
scans using a modified version of Nyul's histogram matching method. This ensures 
that intensity distributions across different datasets are standardized, enhancing the 
generalizability of stroke lesion segmentation models.

This code is modified from the implementation provided in:
https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/nyul.py

References:
1. Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000). New variants of a method of MRI scale 
   standardization. IEEE Transactions on Medical Imaging, 19(2), 143-150.
2. Shah, M., Xiao, Y., Subbanna, N., Francis, S., Arnold, D. L., Collins, D. L., & Arbel, T. (2011). 
   Evaluating intensity normalization on MRIs of human brain with multiple sclerosis. 
   Medical Image Analysis, 15(2), 267-282.

Modifications in this version:
- Applied standardization specifically to DWI scans.
- Added functionality to use foreground masks to exclude non-brain regions.
- Used PCHIP interpolation for smoother transformation.
- Structured the script for easier integration into processing pipelines.

"""


import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import os


def nyul_apply_standard_scale(input_volume, standard_hist, input_mask=None):
    standard_scale, percs = np.load(standard_hist)
    normalized_volume = do_hist_normalization(input_volume, percs, standard_scale, input_mask)
    return normalized_volume


def get_landmarks(volume, percs):
    landmarks = np.percentile(volume, percs)
    return landmarks


def nyul_train_standard_scale(volume_fns, mask_fns=None,
                              i_min=0,
                              i_max=100,
                              i_s_min=1,
                              i_s_max=4095,
                              l_percentile=10,
                              u_percentile=90,
                              step=10):

    percs = np.concatenate(([i_min],
                            np.arange(l_percentile, u_percentile+1, step),
                            [i_max]))
    standard_scale = np.zeros(len(percs))
    landmarks_list = []

    for i, volume_fn in enumerate(volume_fns):
        print('Processing volume ', volume_fn)
        volume_data = nib.load(volume_fn).get_fdata()  
        mask_fn = mask_fns[i] if mask_fns is not None else None  
        mask_data = nib.load(mask_fn).get_fdata() if mask_fn is not None else None  
        masked = volume_data[mask_data > 0] if mask_fn is not None else volume_data.flatten()  
        landmarks = get_landmarks(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])  
        landmarks_list.append(landmarks)  
        standard_scale += landmarks

    standard_scale = standard_scale / len(volume_fns)
    return standard_scale, percs


def do_hist_normalization(volume, landmark_percs, standard_scale, mask=None):
    masked = volume[mask > 0] if mask is not None else volume.flatten() 
    landmarks = get_landmarks(masked, landmark_percs)
    f = PchipInterpolator(landmarks,standard_scale)
    normalized_volume = f(volume.flatten()).reshape(volume.shape)
    return normalized_volume


volume_directory = 'DWI_directory'
mask_directory = 'Mask_directory'  
output_directory = 'Output_directory'
os.makedirs(output_directory, exist_ok=True)


volume_files = [os.path.join(volume_directory, filename) for filename in os.listdir(volume_directory) if filename.endswith('.nii.gz')]
mask_files = [os.path.join(mask_directory, filename) for filename in os.listdir(mask_directory) if filename.endswith('.nii.gz')]


standard_scale, percs = nyul_train_standard_scale(volume_files, mask_fns=mask_files)
standard_path = 'standard_path_directory'
np.save(standard_path, [standard_scale, percs])


for volume_fn, mask_fn in zip(volume_files, mask_files):
    input_volume = nib.load(volume_fn).get_fdata()
    input_mask = nib.load(mask_fn).get_fdata()  
    normalized_volume = nyul_apply_standard_scale(input_volume, standard_path, input_mask=input_mask)
    output = os.path.join(output_directory, os.path.basename(volume_fn).replace('.nii.gz', '.nii.gz'))
    normalized_nii = nib.Nifti1Image(normalized_volume, affine=None)  
    nib.save(normalized_nii, output)