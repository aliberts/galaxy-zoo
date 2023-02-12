import shutil
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose
from tqdm import tqdm

from gzoo.infra import utils
from gzoo.infra.config import DatasetConfig, PreprocessConfig

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
        image = utils.pil_loader(path)
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


class GalaxyRawSet(Dataset):
    """Inference dataset.

    Args:
        cfg (namespace): options from config
        run: wandb run

    Returns (__getitem__):
        image (torch.Tensor)
        image_id (int)
    """

    def __init__(self, dir: Path, types: list[str] | None = None):
        super().__init__()
        if not dir.exists():
            raise FileNotFoundError(
                "Dataset not found. Please run `make dataset` or download it from: "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data"
            )
        if types is None:
            types = [".jpg"]

        self.dir = dir
        self.index_list = [f for type in types for f in dir.glob(f"*{type}")]
        self.index_dict = {fname.stem: fname for fname in self.index_list}

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

    def get_pil(self, image_name: str | int):
        path = self.index_dict[str(image_name)]
        return Image.open(path)

    def copy_to(self, folder: Path, image_list: list[str | int] | None = None):
        folder.mkdir(parents=True, exist_ok=True)
        if image_list is not None:
            file_list = [self.index_dict[str(x)] for x in image_list if str(x) in self.index_dict]
        else:
            file_list = self.index_list

        print(f"Copying {len(file_list)} images from {self.dir} to {folder}")
        for file_in in tqdm(file_list):
            file_out = folder / file_in.name
            shutil.copy(file_in, file_out)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.indexes[idx]
        image_id = image_path.stem
        image = utils.pil_loader(image_path)
        return image, image_id

    def __len__(self) -> int:
        return len(self.indexes)
