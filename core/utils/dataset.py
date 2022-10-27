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
    train: bool
        Train or test flag
    small: bool
        300 photos and 300 Monet paintings small dataset for experiments flag
    ...
    Methods
    -------
    get_image: static
        Returns PIL.Image from filename
    """

    def __init__(self, photo_path, monet_path, train=True, test_size=0.2, transforms=None, small=False):
        np.random.seed(123)

        self.photo_files = np.array(list(photo_path.glob("*.jpg")))
        self.monet_files = np.array(list(monet_path.glob("*.jpg")))

        photo_mask = np.zeros((len(self.photo_files), ))
        monet_mask = np.zeros((len(self.monet_files), ))
        if small:
            photo_mask[np.random.choice(
                len(self.photo_files), 300, replace=False
            )] = 1
            monet_mask[np.random.choice(
                len(self.monet_files), 300, replace=False
            )] = 1
        else:
            photo_mask[np.random.choice(
                len(self.photo_files), int(len(self.photo_files) * (1 - test_size)), replace=False
            )] = 1
            monet_mask[np.random.choice(
                len(self.monet_files), int(len(self.monet_files) * (1 - test_size)), replace=False
            )] = 1
        photo_mask = photo_mask.astype(bool)
        monet_mask = monet_mask.astype(bool)
        if train or small:
            self.photo_files = self.photo_files[photo_mask]
            self.monet_files = self.monet_files[monet_mask]
        elif not train and not small:
            self.photo_files = self.photo_files[~photo_mask]
            self.monet_files = self.monet_files[~monet_mask]
        else:
            raise AttributeError("Wrong combination of parameters 'train' and 'small'")
        self.transforms = transforms

        self.photo_len = len(self.photo_files)
        self.monet_len = len(self.monet_files)
        self.dataset_len = max(self.photo_len, self.monet_len)

        self.train = train
        self.small = small

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        try:
            monet_img = np.array(self.get_image(self.monet_files[index % self.monet_len]))
        except ZeroDivisionError:
            monet_img = np.array(self.get_image(self.photo_files[index % self.photo_len]))
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

    class NormalMalignantDataset(data.Dataset):
        """
        Pytorch style dataset class. Created for processing images of normal CT and malignant CT.
        ...
        Attributes
        ----------
        normal_files: np.ndarray
            Array of normal examples filenames (train or test)
        malignant_files: np.ndarray
            Array of malignant examples filenames (train or test)
        transforms: albumentations.Compose
            Transforms to apply on images data
        normal_len: int
            Number of normal examples
        malignant_len: int
            Number of malignant examples
        dataset_len: int
            Length of dataset
        train: bool
            Train or test flag
        ...
        Methods
        -------
        get_image: static
            Returns PIL.Image from filename
        """

        def __init__(self, normal_path, malignant_path, train=True, test_size=0.2, transforms=None):
            np.random.seed(123)

            self.normal_files = np.array(list(normal_path.glob("*.jpg")))
            self.malignant_files = np.array(list(malignant_path.glob("*.jpg")))

            normal_mask = np.zeros((len(self.normal_files), ))
            malignant_mask = np.zeros((len(self.malignant_files), ))
            normal_mask[np.random.choice(
                len(self.normal_files), int(len(self.normal_files) * (1 - test_size)), replace=False
            )] = 1
            malignant_mask[np.random.choice(
                len(self.malignant_files), int(len(self.malignant_files) * (1 - test_size)), replace=False
            )] = 1
            normal_mask = normal_mask.astype(bool)
            malignant_mask = malignant_mask.astype(bool)
            if train:
                self.normal_files = self.normal_files[normal_mask]
                self.malignant_files = self.malignant_files[malignant_mask]
            else:
                self.normal_files = self.normal_files[~normal_mask]
                self.malignant_files = self.malignant_files[~malignant_mask]
            self.transforms = transforms

            self.normal_len = len(self.normal_files)
            self.malignant_len = len(self.malignant_files)
            self.dataset_len = max(self.normal_len, self.malignant_len)

            self.train = train

        def __len__(self):
            return self.dataset_len

        def __getitem__(self, index):
            try:
                malignant_img = self.get_image(self.malignant_files[index % self.malignant_len])
            # except ZeroDivisionError:
            #     monet_img = self.get_image(self.photo_files[index % self.photo_len])
            normal_img = self.get_image(self.normal_files[index % self.normal_len])
            if self.transforms:
                augmented = self.transforms(image=malignant_img, image0=normal_img)
                malignant_img = augmented["image"]
                normal_img = augmented["image0"]
            return malignant_img, normal_img

        @staticmethod
        def get_image(filename):
            image = Image.open(filename)
            return image


if __name__ == "__main__":
    ds_train = MonetPhotoDataset(config.PHOTO_DIR, config.MONET_DIR, transforms=config.TRANSFORMS)
    ds_test = MonetPhotoDataset(config.PHOTO_DIR, config.MONET_DIR, train=False, transforms=config.TRANSFORMS)
    train_monet, train_photo = ds_train[0]
    test_monet, test_photo = ds_test[0]
    print(f"Train dataset len: {len(ds_train)} | Monet shape: {train_monet.shape} | Photo shape: {train_photo.shape}")
    print(f"Test dataset len: {len(ds_test)} | Monet shape: {test_monet.shape} | Photo shape: {test_photo.shape}")
    print(f"Type: {type(test_photo)} | DType: {test_photo.dtype}")
