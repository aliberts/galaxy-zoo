from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose

from gzoo.infra.config import DatasetConfig, PreprocessConfig
from gzoo.infra.utils import pil_loader

# from torchvision.utils import save_image


class GalaxyTrainSet(Dataset):
    """Train/Val/Test dataset.

    Args:
        split (str): "train", "val", "test"
        cfg (namespace): options from config

    Returns (__getitem__):
        image (torch.Tensor)
        label (torch.Tensor)
    """

    def __init__(
        self, split: str, data_cfg: DatasetConfig, prepro_cfg: PreprocessConfig, dir: Path
    ):
        super().__init__()
        self.split = split
        if not data_cfg.dir.exists():
            raise FileNotFoundError(
                "Dataset not found. Please run `make dataset` or download it from: "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data"
            )
        labels_split_path = dir / data_cfg.clf_labels_split_file
        self.image_dir = dir / data_cfg.clf_images_dir.name

        try:
            labels_df = pd.read_csv(labels_split_path, header=0, sep=",", index_col="GalaxyID")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Classification labels not found. "
                "Run `poetry run python -m gzoo.app.make_labels` "
                "and then `poetry run python -m gzoo.app.split_data` first."
            )
        self.class_names = data_cfg.class_names
        self.indexes, self.labels = self._get_split(labels_df)
        self.image_tf = self._build_transforms(prepro_cfg)

    def _get_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df[df["Split"] == self.split]
        df = df.assign(**{"Label": df.loc[:, "Class"].apply(self._get_class_number)})
        indexes = df.index.to_list()
        labels = df["Label"].to_list()
        return indexes, labels

    def _get_class_number(self, class_name: str) -> int:
        return self.class_names.index(class_name)

    def _build_transforms(self, cfg: PreprocessConfig) -> Compose:
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

    def __getitem__(self, idx: int) -> tuple[Image.Image, torch.tensor]:
        image_id = self.indexes[idx]
        path = self.image_dir / f"{image_id}.jpg"
        image = pil_loader(path)
        # -- DEBUG --
        # tens = transforms.ToTensor()
        # save_image(tens(image), f'logs/{idx}_raw.png')
        image = self.image_tf(image)
        # save_image(image, f'logs/{idx}_tf.png')
        # breakpoint()
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image, label

    def __len__(self) -> int:
        return len(self.indexes)


class GalaxyPredictSet(Dataset):
    """Inference dataset.

    Args:
        cfg (namespace): options from config
        run: wandb run

    Returns (__getitem__):
        image (torch.Tensor)
        image_id (int)
    """

    def __init__(self, data_cfg: DatasetConfig):
        super().__init__()
        if not data_cfg.dir.exists():
            raise FileNotFoundError(
                "Dataset not found. Please run `make dataset` or download it from: "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data"
            )
        self.image_dir = data_cfg.clf_images

        self.indexes = []
        for filename in self.image_dir.glob("*.jpg"):
            self.indexes.append(filename.stem)

        self.image_tf = self._build_transforms()

    def _build_transforms(self) -> Compose:
        image_tf = []
        image_tf.extend(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        return transforms.Compose(image_tf)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        image_id = self.indexes[idx]
        path = self.image_dir / f"{image_id}.jpg"
        image = pil_loader(path)
        image = self.image_tf(image)
        return image, image_id

    def __len__(self) -> int:
        return len(self.indexes)
