from torch.utils import data
from PIL import Image
import numpy as np
from . import config


class MonetPhotoDataset(data.Dataset):
    """
    Pytorch style dataset class. Created for processing images of Claude Monet paintings
    and common photos of nature.
    ...
    Attributes
    ----------
    photo_files: np.ndarray
        Array of photo filenames (train or test)
    monet_files: np.ndarray
        Array of monet filenames (train or test)
    transforms: albumentations.Compose
        Transforms to apply on images data
    photo_len: int
        Number of photos
    monet_len: int
        Number of Monet paintings
    dataset_len: int
        Length of dataset
    ...
    Methods
    -------
    get_image: static
        Returns PIL.Image from filename
    """

    def __init__(self, photo_path, monet_path, train=True, test_size=0.2, transforms=None):
        np.random.seed(123)

        self.photo_files = np.array(list(photo_path.glob("*.jpg")))
        self.monet_files = np.array(list(monet_path.glob("*.jpg")))

        photo_mask = np.zeros((len(self.photo_files), ))
        monet_mask = np.zeros((len(self.monet_files), ))
        if train:
            photo_mask[np.random.choice(
                len(self.photo_files), int(len(self.photo_files) * (1 - test_size)), replace=False
            )] = 1
            monet_mask[np.random.choice(
                len(self.monet_files), int(len(self.monet_files) * (1 - test_size)), replace=False
            )] = 1
        else:
            photo_mask[np.random.choice(
                len(self.photo_files), int(len(self.photo_files) * test_size), replace=False
            )] = 1
            monet_mask[np.random.choice(
                len(self.monet_files), int(len(self.monet_files) * test_size), replace=False
            )] = 1
        photo_mask = photo_mask.astype(bool)
        monet_mask = monet_mask.astype(bool)
        self.photo_files = self.photo_files[photo_mask]
        self.monet_files = self.monet_files[monet_mask]
        self.transforms = transforms

        self.photo_len = len(self.photo_files)
        self.monet_len = len(self.monet_files)
        self.dataset_len = max(self.photo_len, self.monet_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        monet_img = np.array(self.get_image(self.monet_files[index % self.monet_len]))
        photo_img = np.array(self.get_image(self.photo_files[index % self.photo_len]))
        if self.transforms:
            augmented = self.transforms(image=monet_img, image0=photo_img)
            monet_img = augmented["image"]
            photo_img = augmented["image0"]
        return monet_img, photo_img

    @staticmethod
    def get_image(filename):
        image = Image.open(filename)
        return image


if __name__ == "__main__":
    ds_train = MonetPhotoDataset(config.PHOTO_DIR, config.MONET_DIR, transforms=config.transform)
    ds_test = MonetPhotoDataset(config.PHOTO_DIR, config.MONET_DIR, train=False, transforms=config.transform)
    train_monet, train_photo = ds_train[0]
    test_monet, test_photo = ds_test[0]
    print(f"Train dataset len: {len(ds_train)} | Monet shape: {train_monet.shape} | Photo shape: {train_photo.shape}")
    print(f"Test dataset len: {len(ds_test)} | Monet shape: {test_monet.shape} | Photo shape: {test_photo.shape}")
    print(f"Type: {type(test_photo)} | DType: {test_photo.dtype}")
