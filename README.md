# 🧠 Stroke Lesion Segmentation from DWI Scans

This framework performs automated ischemic stroke lesion segmentation on diffusion-weighted MRI using a hybrid approach that combines handcrafted feature-based machine learning (Random Forest and XGBoost) and deep learning (EfficientNet-B2 U-Net) approaches. The preprocessing pipeline includes an intensity standardization step, designed to reduce variability in image intensities across different scanners and centers, enabling more consistent and reproducible lesion segmentation.


---

## 📂 Repository Overview
stroke_lesion_segmentation/

├── data_preprocessing/ # Intensity standardization

├── feature_extraction/ # Handcrafted convolutional filters

├── models/ # Random Forest, XGBoost, EfficientNet-B2 U-Net training

└── README.md









---
**Intellectual Property Notice**  
© Universiti Teknologi Malaysia (UTM).  
This repository is protected under the Malaysia Copyright Act 1987  
(Copyright Notification No. CRLY2025J10766).  
All rights reserved.
