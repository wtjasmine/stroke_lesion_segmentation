"""
efficientnet-b2_u-net_train.py
===========================================================

Performs 2D slice-based stroke lesion segmentation using an EfficientNet-B2
backbone within the U-Net architecture.


Author: Jasmine Wang Thye Wei
Date: 2025-10-12
"""

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from segmentation_models import Unet
import segmentation_models as sm


# ------------------------------------------------------------------
# ⚙️ Reproducibility Settings
# ------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ------------------------------------------------------------------
# Directory Configuration (make these paths your own)
# ------------------------------------------------------------------
# Root dataset folder
DATA_ROOT = "./dataset"

# Sub-folders for training / validation / test splits
train_img_dir = os.path.join(DATA_ROOT, "train/images")
train_mask_dir = os.path.join(DATA_ROOT, "train/masks")
val_img_dir   = os.path.join(DATA_ROOT, "val/images")
val_mask_dir  = os.path.join(DATA_ROOT, "val/masks")

# Output folder for model checkpoints and logs
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Data Loader
# ------------------------------------------------------------------
def load_batch(img_dir, file_list):
    """Load a batch of .npy images."""
    return np.array([
        np.load(os.path.join(img_dir, f))
        for f in file_list if f.endswith(".npy")
    ])

def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """Yield batches of (image, mask) pairs."""
    L = len(img_list)
    while True:
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            X = load_batch(img_dir, img_list[start:end])
            Y = load_batch(mask_dir, mask_list[start:end])
            yield X, Y


# ------------------------------------------------------------------
# Dataset Setup
# ------------------------------------------------------------------
train_imgs = sorted(os.listdir(train_img_dir))
train_masks = sorted(os.listdir(train_mask_dir))
val_imgs = sorted(os.listdir(val_img_dir))
val_masks = sorted(os.listdir(val_mask_dir))

BATCH_SIZE = 32
EPOCHS = 80
STEPS_PER_EPOCH = len(train_imgs) // BATCH_SIZE
VAL_STEPS = len(val_imgs) // BATCH_SIZE

train_gen = image_loader(train_img_dir, train_imgs, train_mask_dir, train_masks, BATCH_SIZE)
val_gen   = image_loader(val_img_dir, val_imgs, val_mask_dir, val_masks, BATCH_SIZE)


# ------------------------------------------------------------------
# Model Definition – EfficientNet-B2 U-Net
# ------------------------------------------------------------------
base_unet = Unet(backbone_name="efficientnetb2", encoder_weights="imagenet")

inp = Input(shape=(None, None, 1))
x = Conv2D(3, (1, 1), kernel_initializer=GlorotUniform(seed=SEED))(inp)  # map 1→3 channels
out = base_unet(x)

model = Model(inp, out, name="EfficientNetB2_UNet")
model.summary()


# ------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------
LEARNING_RATE = 0.001
metric = sm.metrics.IOUScore(threshold=0.5)

model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[metric]
)

# Checkpoints (saved in `results/`)
CKPT_IOU  = os.path.join(OUTPUT_DIR, "best_iou_model_efficientnetb2.keras")
CKPT_LOSS = os.path.join(OUTPUT_DIR, "best_loss_model_efficientnetb2.keras")

callbacks = [
    ModelCheckpoint(CKPT_IOU, monitor="val_iou_score", save_best_only=True, mode="max", verbose=1),
    ModelCheckpoint(CKPT_LOSS, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    EarlyStopping(monitor="val_loss", patience=15, mode="min", restore_best_weights=True, verbose=1),
]


# ------------------------------------------------------------------
# Train Model
# ------------------------------------------------------------------
history = model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_gen,
    validation_steps=VAL_STEPS,
    epochs=EPOCHS,
    verbose=1,
    shuffle=False,
    callbacks=callbacks,
)


