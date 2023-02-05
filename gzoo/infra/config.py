"""
If you make changes here, you should also update the related .yaml config files in config/
by running 'poetry run python -m gzoo.app.update_config'
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from pyrallis import field

from gzoo.infra import utils


@dataclass
class ExpConfig:
    name: Optional[str] = None
    test: bool = False
    task: str = "classification"
    evaluate: bool = False


@dataclass
class WandBConfig:
    use: bool = False
    freq: int = 10
    entity: str = "aliberts"
    project: str = "galaxy-zoo"
    run_name: Optional[str] = None
    tags: Optional[list] = field(default_factory=lambda: ["baseline", "model exploration"])
    note: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str = "galaxy-zoo"
    dir: Path = field(default=Path("dataset/"))
    version: Optional[str] = None
    test_split_ratio: float = 0.1  # test / (train + val) ratio
    val_split_ratio: float = 0.1  # val / train ratio
    class_names: list = field(
        default_factory=lambda: [
            "completely_round_smooth",
            "in_between_smooth",
            "cigar_shaped_smooth",
            "edge_on",
            "spiral",
        ]
    )
    eda_table: str = "eda_table"
    eda_table_split: str = "eda_table_split"
    clf_labels_split_file: Path = field(default=Path("clf_labels_split.csv"))
    clf_images_dir: Path = field(default=Path("clf_images"))
    clf_labels_file: Path = field(default=Path("clf_labels.csv"))
    reg_images_train_dir: Path = field(default=Path("images_training_rev1"))
    reg_images_test_dir: Path = field(default=Path("images_test_rev1"))
    reg_labels_file: Path = field(default=Path("training_solutions_rev1.csv"))
    predictions_file: Path = field(default=Path("predictions.csv"))

    def __post_init__(self):
        if self.version is None:
            self.version = "latest"

    @property
    def clf_name(self) -> Path:
        return f"{self.name}-clf"

    @property
    def reg_name(self) -> Path:
        return f"{self.name}-reg"

    @property
    def clf_labels_split(self) -> Path:
        return self.dir / self.clf_labels_split_file

    @property
    def clf_images(self) -> Path:
        return self.dir / self.clf_images_dir

    @property
    def clf_labels(self) -> Path:
        return self.dir / self.clf_labels_file

    @property
    def reg_images_train(self) -> Path:
        return self.dir / self.reg_images_train_dir

    @property
    def reg_labels(self) -> Path:
        return self.dir / self.reg_labels_file

    @property
    def predictions(self) -> Path:
        return self.dir / "predictions" / self.predictions_file


@dataclass
class ModelConfig:
    arch: str = "resnet18"  # model architecture, 'resnetN' or 'customN' supported
    pretrained: bool = False
    freeze: bool = False
    output_constraints: bool = True
    path: Optional[Path] = field(default=None)  # path to model


@dataclass
class ComputeConfig:
    seed: Optional[int] = None  # seed for initializing training.
    epochs: int = 90
    start_epoch: int = 0  # manual epoch number (useful on restarts)
    use_cuda: bool = True
    workers: int = 8  # number of data loading workers
    batch_size: int = 128
    print_freq: int = 10
    resume: Optional[Path] = None  # path to latest checkpoint

    def __post_init__(self):
        if self.seed is not None:
            self.seed = int(self.seed)
            utils.set_random_seed(self.seed)

        if not self.use_cuda:
            torch.cuda.is_available = lambda: False


@dataclass
class DistributedConfig:
    # Use multi-processing distributed training to launch
    # N processes per node, which has N GPUs. This is the
    # fastest way to use PyTorch for either single node or
    # multi node data parallel training
    use: bool = False
    multiprocessing_distributed: bool = False
    world_size: int = -1  # number of nodes for distributed training
    rank: int = -1  # node rank for distributed training
    dist_url: str = "tcp://224.66.41.62:23456"  # url used to set up distributed training
    dist_backend: str = "nccl"  # distributed backend
    gpu: Optional[int] = None  # GPU id to use
    ngpus_per_node: Optional[int] = None

    def __post_init__(self):
        if self.gpu is not None:
            warnings.warn(
                "You have chosen a specific GPU. This will completely disable data parallelism."
            )
        if self.dist_url == "env://" and self.world_size == -1:
            self.world_size = int(os.environ["WORLD_SIZE"])
        self.use = self.world_size > 1 or self.multiprocessing_distributed
        self.ngpus_per_node = torch.cuda.device_count()


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 3.0e-4  # https://twitter.com/karpathy/status/801621764144971776
    lr_scheduler_freq: int = 30  # lr is divided by a factor of 10 each this number of epochs
    weight_decay: float = 1.0e-4
    momentum: float = 0.9  # for SGD only


@dataclass
class PreprocessConfig:
    augmentation: bool = True
    rotate: bool = True
    flip: bool = True
    color_jitter: bool = True
    color_jitter_factor: float = 0.1


@dataclass
class EnsemblingConfig:
    use: bool = False
    n_estimators: int = 50


@dataclass
class UploadConfig:
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: bool = False


@dataclass
class TrainConfig:
    exp: ExpConfig = field(default_factory=ExpConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    debug: bool = False


@dataclass
class PredictConfig:
    exp: ExpConfig = field(default_factory=lambda: ExpConfig(test=True, evaluate=True))
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    ensembling: EnsemblingConfig = field(default_factory=EnsemblingConfig)
    template: Path = field(default=Path("all_ones_benchmark.csv"))
    output: Path = field(default="predictions/predictions.csv")
    debug: bool = False
