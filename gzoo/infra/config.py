"""
If you make changes here, you should also update the related .yaml config files in config/
with a pyrallis.dump command that is present at the beginning of train.py and predict.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyrallis import field


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
    dir: Path = field(default=Path("/home/simon/datasets/galaxy_zoo/"))
    images: str = "images_training_rev1/"
    train_labels: str = "classification_labels_train_val.csv"
    test_labels: str = "classification_labels_test.csv"
    predictions: str = "predictions/training_solutions_rev1.csv"


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
    resume: Optional[str] = None  # path to latest checkpoint


@dataclass
class DistributedConfig:
    # Use multi-processing distributed training to launch
    # N processes per node, which has N GPUs. This is the
    # fastest way to use PyTorch for either single node or
    # multi node data parallel training
    multiprocessing_distributed: bool = False
    world_size: int = -1  # number of nodes for distributed training
    rank: int = -1  # node rank for distributed training
    dist_url: str = "tcp://224.66.41.62:23456"  # url used to set up distributed training
    dist_backend: str = "nccl"  # distributed backend
    gpu: Optional[int] = None  # GPU id to use


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
    colorjitter: bool = True


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


@dataclass
class PredictConfig:
    exp: ExpConfig = field(default_factory=ExpConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    template: str = "all_ones_benchmark.csv"
    output: Path = field(default="predictions/predictions.csv")
