import glob
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from gzoo.infra.config import PredictConfig, PreprocessConfig, TrainConfig

# from torchvision.utils import save_image


def pil_loader(path: Path) -> Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with path.open("rb") as f, Image.open(f) as img:
        return img.convert("RGB")


class GalaxyTrainSet(Dataset):
    """Train/Val dataset.

    Args:
        split (str): "train", "val"
        cfg (namespace): options from config

    Returns (__getitem__):
        image (torch.Tensor)
        label (torch.Tensor)
    """

    def __init__(self, split, cfg: TrainConfig):
        super().__init__()
        self.split = split
        self.val_split_ratio = cfg.dataset.val_split_ratio
        self.task = cfg.exp.task
        self.seed = cfg.compute.seed if cfg.compute.seed is not None else 0
        if not cfg.dataset.dir.exists():
            raise FileNotFoundError(
                "Please download them from "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data"
            )
        self.image_dir = cfg.dataset.train_images
        self.label_file = cfg.dataset.train_labels
        if cfg.exp.evaluate:
            self.label_file = cfg.dataset.test_labels

        df = pd.read_csv(self.label_file, header=0, sep=",")
        self.indexes, self.labels = self._split_dataset(df, cfg.exp.evaluate)
        self.image_tf = self._build_transforms(cfg.preprocess)

    def _split_dataset(self, df, evaluate):
        indexes = df.iloc[:, 0]
        labels = df.iloc[:, 1:]

        if self.task == "classification" and not evaluate:
            idx_train, idx_val, lbl_train, lbl_val = train_test_split(
                indexes,
                labels,
                test_size=self.val_split_ratio,
                random_state=self.seed,
                stratify=labels,
            )
            if self.split == "train":
                indexes = idx_train
                labels = lbl_train
            elif self.split == "val":
                indexes = idx_val
                labels = lbl_val

        elif self.task == "regression" and not evaluate:
            indices = np.random.RandomState(seed=self.seed).permutation(indexes.shape[0])
            val_len = int(len(indexes) * self.val_split_ratio)
            val_idx, train_idx = indices[:val_len], indices[val_len:]
            if self.split == "train":
                indexes = indexes[train_idx]
            elif self.split == "val":
                indexes = indexes[val_idx]

        return indexes.reset_index(drop=True), labels.reset_index(drop=True)

    def _build_transforms(self, cfg: PreprocessConfig):
        image_tf = []
        if self.split == "train" and cfg.augmentation:
            if cfg.rotate:
                image_tf.append(transforms.RandomRotation(180))
            if cfg.flip:
                image_tf.extend(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                    ]
                )
            if cfg.color_jitter:
                image_tf.extend(
                    [
                        transforms.ColorJitter(
                            brightness=cfg.color_jitter_factor,
                            contrast=cfg.color_jitter_factor,
                            # saturation=cfg.color_jitter_factor,
                            # hue=cfg.color_jitter_factor,
                        ),
                    ]
                )
        image_tf.extend(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        return transforms.Compose(image_tf)

    def __getitem__(self, idx):
        image_id = self.indexes.iloc[idx]
        path = self.image_dir / f"{image_id}.jpg"
        image = pil_loader(path)
        # -- DEBUG --
        # tens = transforms.ToTensor()
        # save_image(tens(image), f'logs/{idx}_raw.png')
        image = self.image_tf(image)
        # save_image(image, f'logs/{idx}_tf.png')
        # breakpoint()
        label = self.labels.iloc[idx]
        if self.task == "classification":
            label = torch.tensor(label).long()
        elif self.task == "regression":
            label = torch.tensor(label).float()
        return image, label

    def __len__(self):
        return len(self.indexes)


class GalaxyTestSet(Dataset):
    """Test dataset.

    Args:
        split (str): "train", "val"
        cfg (namespace): options from config

    Returns (__getitem__):
        image (torch.Tensor)
        image_id (int)
    """

    def __init__(self, cfg: PredictConfig):
        super().__init__()
        if not cfg.dataset.dir.exists():
            raise FileNotFoundError(
                "Please download them from "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data"
            )

        self.image_dir = cfg.dataset.test_images
        image_list = []
        for filename in glob.glob(f"{self.image_dir}/*.jpg"):
            idx = filename.split("/")[-1][:-4]
            image_list.append(idx)
        self.indexes = pd.Series(image_list)

        image_tf = []
        image_tf.extend(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        self.image_tf = transforms.Compose(image_tf)

    def __getitem__(self, idx):
        image_id = self.indexes.iloc[idx]
        path = self.image_dir / f"{image_id}.jpg"
        image = pil_loader(path)
        image = self.image_tf(image)
        return image, image_id

    def __len__(self):
        return len(self.indexes)


def imagenet(cfg: TrainConfig):
    traindir = cfg.dataset.dir / "train"
    valdir = cfg.dataset.dir / "val"
    # https://stackoverflow.com/questions/58151507
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    test_set = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    return train_set, test_set
