import logging
import pprint
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from wandb.sdk.wandb_run import Run

import wandb
from gzoo.infra.config import TrainConfig
from gzoo.infra.logging import Log


def setup_wandb_run(cfg: TrainConfig) -> Run:
    run = wandb.init(
        name=cfg.wandb.run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        notes=cfg.wandb.note,
        tags=cfg.wandb.tags,
        config=asdict(cfg),
    )
    if cfg.wandb.run_name is not None:
        wandb.run_name = cfg.wandb.run_name

    return run


def setup_train_log(cfg: TrainConfig) -> Log:
    log = Log("train", cfg.exp, cfg.model.arch)
    log.toggle()
    logging.debug("arguments:")
    logging.debug(pprint.pformat(asdict(cfg)))
    return log


def set_random_seed(seed: int) -> None:
    # is pytorch dataloader with multi-threads deterministic ?
    # cudnn may not be deterministic anyway
    torch.manual_seed(seed)  # on CPU and GPU
    numpy.random.seed(seed)  # useful ? not thread safe
    random.seed(seed)  # useful ? thread safe
    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def pil_loader(path: Path) -> Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with path.open("rb") as f, Image.open(f) as img:
        return img.convert("RGB")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
