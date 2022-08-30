import torch
import albumentations as alb
from albumentations import pytorch
import pathlib
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PHOTO_DIR = pathlib.Path("../data/gan-getting-started/photo_jpg")
MONET_DIR = pathlib.Path("../data/gan-getting-started/monet_jpg")
# print("Listdir:", os.listdir(PHOTO_DIR))
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_PHOTO = pathlib.Path("../states/genphoto.pth.tar")
CHECKPOINT_GEN_MONET = pathlib.Path("../states/genmonet.pth.tar")
CHECKPOINT_DISC_PHOTO = pathlib.Path("../states/discphoto.pth.tar")
CHECKPOINT_DISC_MONET = pathlib.Path("../states/discmonet.pth.tar")

TRANSFORMS = alb.Compose(
    [
        alb.Resize(height=256, width=256),
        alb.HorizontalFlip(p=0.5),
        alb.ColorJitter(p=0.2),
        alb.Normalize(mean=(0.5, 0.5, 0.5),
                      std=(0.5, 0.5, 0.5),
                      max_pixel_value=255),
        pytorch.transforms.ToTensorV2(),
    ],
    additional_targets={"image0": "image"}  # We can pass additional target and all augmentations will be applied to it
)

