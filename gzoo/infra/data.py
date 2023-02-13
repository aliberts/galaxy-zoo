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
    """
    Labeled dataset.

    Args:
        dir (Path): Path to the root directory of the dataset
            (containing the image folders and labels files).
        split (str): Partition to select ("train", "val" or "test").
        data_cfg (DatasetConfig): Dataset config.
        prepro_cfg (PreprocessConfig): Image preprocessing config.

    Returns (__getitem__):
        image (torch.Tensor)
        label (torch.Tensor)
    """

    def __init__(
        self, dir: Path, split: str, data_cfg: DatasetConfig, prepro_cfg: PreprocessConfig
    ):
        super().__init__()
        if not data_cfg.dir.exists():
            raise FileNotFoundError(
                "Dataset not found. Please run `make dataset` or download it from: "
                "https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data"
            )

        if split == "train" or split == "val":
            labels_path = dir / data_cfg.clf_labels_train_val_file
            self.image_dir = dir / data_cfg.clf_images_train_val_dir
        elif split == "test":
            labels_path = dir / data_cfg.clf_labels_test_file
            self.image_dir = dir / data_cfg.clf_images_test_dir
        else:
            raise ValueError
            "split must be either 'train', 'val' or 'test'."

        try:
            labels_df = pd.read_csv(labels_path, header=0, sep=",", index_col="GalaxyID")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Classification labels not found. "
                "Run `poetry run python -m gzoo.app.make_labels` "
                "and then `poetry run python -m gzoo.app.split_data` first."
            )

        self.split = split
        self.class_names = data_cfg.class_names
        self.index_list, self.labels = self._make_index(labels_df)

        if not self._validate_images_match_labels():
            raise FileNotFoundError(
                f"Some images in {self.image_dir} do not match with labels in {labels_path}"
            )

        self.image_tf = self._build_transforms(prepro_cfg)

    def _validate_images_match_labels(self) -> bool:
        return all(index.exists() for index in self.index_list)

    def _make_index(self, df: pd.DataFrame) -> tuple[list[Path], list[int]]:
        df = df[df["Split"] == self.split]
        df = df.assign(**{"Label": df.loc[:, "Class"].apply(self._get_class_number)})
        indexes = [(self.image_dir / str(idx)).with_suffix(".jpg") for idx in df.index.to_list()]
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.tensor]:
        image_path = self.index_list[idx]
        image = utils.pil_loader(image_path)
        image = self.image_tf(image)
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image, label

    def __len__(self) -> int:
        return len(self.index_list)


class GalaxyRawSet(Dataset):
    """
    Raw dataset (no labels).

    Args:
        dir (Path): Path to the directory containing the images.
        types (list[str] | None, optional): Images extensions. If None, defaults to ".jpg".

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
        self.index_list, self.index_dict = self._make_index(dir, types)

        self.image_tf = self._build_transforms()

    def _make_index(self, dir: Path, types: list[str]) -> tuple[list[Path], dict[str:Path]]:
        index_list = [f for type in types for f in dir.glob(f"*{type}")]
        index_dict = {fname.stem: fname for fname in index_list}
        return index_list, index_dict

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
        image_path = self.index_list[idx]
        image = utils.pil_loader(image_path)
        image_id = int(image_path.stem)
        return image, image_id

    def __len__(self) -> int:
        return len(self.index_list)
