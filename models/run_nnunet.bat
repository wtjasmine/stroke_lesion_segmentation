@echo off

REM ===============================
REM nnU-Net preprocessing
REM ===============================

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity


REM ===============================
REM nnU-Net training (5-fold CV)
REM ===============================

nnUNetv2_train 1 2d 0
nnUNetv2_train 1 2d 1
nnUNetv2_train 1 2d 2
nnUNetv2_train 1 2d 3
nnUNetv2_train 1 2d 4


REM ===============================
REM Prediction for center1 (internal)
REM ===============================

nnUNetv2_predict ^
-i C:\nnUNet\nnUNet_raw\Dataset001_ISLES2D\imagesTs_c1 ^
-o C:\nnUNet\nnUNet_results\Dataset001_ISLES2D\predictions_ensemble_c1 ^
-d 1 ^
-c 2d ^
-f 0 1 2 3 4


REM ===============================
REM Prediction for center2 (external)
REM ===============================

nnUNetv2_predict ^
-i C:\nnUNet\nnUNet_raw\Dataset001_ISLES2D\imagesTs_c2 ^
-o C:\nnUNet\nnUNet_results\Dataset001_ISLES2D\predictions_ensemble_c2 ^
-d 1 ^
-c 2d ^
-f 0 1 2 3 4

echo Finished running nnU-Net
pause