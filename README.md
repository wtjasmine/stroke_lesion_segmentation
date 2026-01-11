# 🧠 Stroke Lesion Segmentation from DWI Scans

This framework performs automated ischemic stroke lesion segmentation on diffusion-weighted MRI using a hybrid approach that combines handcrafted feature-based machine learning (Random Forest and XGBoost) and deep learning (EfficientNet-B2 U-Net) approaches. The preprocessing pipeline includes an intensity standardization step, designed to reduce variability in image intensities across different scanners and centers, enabling more consistent and reproducible lesion segmentation.


## 🧪 Preprocessing: Intensity Standardization
<img width="2003" height="1134" alt="image" src="https://github.com/user-attachments/assets/5c3b5d44-0ccd-47c5-86f0-6891b725a24e" />

---
**Figure 1. Effect of intensity standardization on diffusion-weighted MRI.**  
Top: voxel intensity distributions across subjects before (left) and after (right) standardization, showing improved alignment of intensity profiles.  
Bottom: representative axial DWI slices from two datasets before and after standardization.  
The standardization step reduces inter-scanner and inter-subject intensity variability while preserving anatomical and lesion contrast.


## 🧠 Segmentation Framework & Outputs
<img width="949" height="1097" alt="image" src="https://github.com/user-attachments/assets/f595fa73-ed57-4199-b868-47480ac27c4d" />


---
**Figure 2. Example stroke lesion segmentation results.**  
Representative diffusion-weighted MRI slices (top rows) and corresponding lesion masks (bottom rows).  
Columns illustrate different anatomical locations and lesion characteristics.  
The framework captures both large territorial infarcts and small focal lesions, demonstrating robustness across heterogeneous stroke presentations.


---

## 📂 Repository Overview
stroke_lesion_segmentation/

├── data_preprocessing/ # Intensity standardization

├── feature_extraction/ # Handcrafted convolutional filters

├── models/ # Random Forest, XGBoost, EfficientNet-B2 U-Net training

└── README.md


## 📌 Notes on Reproducibility

- All experiments are conducted on 2D axial DWI slices extracted from 3D volumes.
- Subject-level data splitting is used to avoid data leakage between training and testing sets.
- Evaluation is performed using Dice similarity coefficient and related overlap metrics.






---
**Intellectual Property Notice**  
© Universiti Teknologi Malaysia (UTM).  
This repository is protected under the Malaysia Copyright Act 1987  
(Copyright Notification No. CRLY2025J10766).  
All rights reserved.
