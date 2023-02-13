import logging
import os
import shutil
from pathlib import Path
from typing import TypeAlias, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from wandb.sdk.wandb_run import Run

import wandb
from gzoo.domain import models
from gzoo.infra import config, data

Loss: TypeAlias = Union[nn.CrossEntropyLoss, "RMSELoss"]


def setup_cuda(
    model: models.Model,
    cfg: config.TrainConfig | config.PredictConfig,
) -> tuple[models.Model, config.ComputeConfig]:
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif cfg.distributed.use:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.distributed.gpu is not None:
            torch.cuda.set_device(cfg.distributed.gpu)
            model.cuda(cfg.distributed.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.compute.batch_size = int(cfg.compute.batch_size / cfg.distributed.ngpus_per_node)
            cfg.compute.workers = int(
                (cfg.compute.workers + cfg.distributed.ngpus_per_node - 1)
                / cfg.distributed.ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.distributed.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.distributed.gpu is not None:
        torch.cuda.set_device(cfg.distributed.gpu)
        model = model.cuda(cfg.distributed.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_name = cfg.model.arch
        if model_name.startswith("alexnet") or model_name.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model, cfg.compute


def setup_distributed(
    gpu: int | None, dist_cfg: config.DistributedConfig
) -> config.DistributedConfig:
    if dist_cfg.gpu is not None:
        logging.info(f"Use GPU: {dist_cfg.gpu}")

    if dist_cfg.use:
        if dist_cfg.dist_url == "env://" and dist_cfg.rank == -1:
            dist_cfg.rank = int(os.environ["RANK"])
        if dist_cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            dist_cfg.rank = dist_cfg.rank * dist_cfg.ngpus_per_node + gpu
        dist.init_process_group(
            backend=dist_cfg.dist_backend,
            init_method=dist_cfg.dist_url,
            world_size=dist_cfg.world_size,
            rank=dist_cfg.rank,
        )
    return dist_cfg


def create_model(cfg: config.TrainConfig | config.PredictConfig) -> models.Model:
    if cfg.model.arch.startswith("resnet"):
        model = models.ResNet(cfg)
    elif cfg.model.arch.startswith("custom"):
        model = models.CustomNet(cfg)
    elif cfg.model.arch == "random":
        model = models.Random(cfg)
    else:
        raise NotImplementedError
    return model


def load_model(cfg: config.TrainConfig, model: models.Model) -> models.Model:
    if cfg.model.path:
        pth = cfg.model.path
    else:
        pth = Path(f"models/{cfg.model.arch}.pth.tar")

    if not pth.is_file():
        raise FileNotFoundError(f"=> model checkpoint not found at '{pth}'")

    if not cfg.compute.use_cuda:
        checkpoint = torch.load(pth, map_location=torch.device("cpu"))
        model = nn.DataParallel(model)
    elif cfg.distributed.gpu is None:
        checkpoint = torch.load(pth)
    else:
        # Map model to be loaded to specified single gpu.
        loc = f"cuda:{cfg.distributed.gpu}"
        checkpoint = torch.load(pth, map_location=loc)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> loaded model '{pth}'")
    return model


def make_train_dataset(
    cfg: config.TrainConfig, run: Run | None
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    dataset_dir = cfg.dataset.clf
    if run is not None:
        dataset_artifact = run.use_artifact(f"clf_train_val:{cfg.dataset.version}")
        dataset_dir = Path(dataset_artifact.download())

    train_dataset = data.GalaxyTrainSet(dataset_dir, "train", cfg.dataset, cfg.preprocess)
    val_dataset = data.GalaxyTrainSet(dataset_dir, "val", cfg.dataset, cfg.preprocess)

    if cfg.distributed.use:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.compute.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.compute.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.compute.batch_size,
        shuffle=False,
        num_workers=cfg.compute.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def make_test_dataset(cfg: config.PredictConfig, run: Run | None) -> DataLoader:
    dataset_dir = cfg.dataset.clf
    if run is not None:
        dataset_artifact = run.use_artifact(f"clf_test:{cfg.dataset.version}")
        dataset_dir = Path(dataset_artifact.download())

    test_dataset = data.GalaxyTrainSet(dataset_dir, "test", cfg.dataset, cfg.preprocess)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.compute.batch_size,
        shuffle=False,
        num_workers=cfg.compute.workers,
        pin_memory=True,
    )
    return test_loader


def make_criterion(cfg: config.TrainConfig | config.PredictConfig) -> Loss:
    if cfg.exp.task == "classification":
        if isinstance(cfg, config.TrainConfig):
            # https://discuss.pytorch.org/t/11455/10
            n_samples = [8014, 7665, 550, 3708, 7416]
            # weights = [max(n_samples) / x for x in n_samples]
            weights = [1.0 - x / sum(n_samples) for x in n_samples]
            weights = torch.FloatTensor(weights).cuda(cfg.distributed.gpu)
        else:
            weights = None
        criterion = nn.CrossEntropyLoss(weight=weights).cuda(cfg.distributed.gpu)
    elif cfg.exp.task == "regression":
        criterion = RMSELoss().cuda(cfg.distributed.gpu)
    return criterion


def make_optimizer(cfg: config.TrainConfig, model: models.Model) -> Optimizer:
    if cfg.model.arch == "random":
        optimizer = None
    elif cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), cfg.optimizer.lr)
    elif cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    return optimizer


def save_checkpoint(
    log_dir: Path, state: dict, model: models.Model, is_best: bool, cfg: config.TrainConfig
) -> None:
    file_path = log_dir / "checkpoint.pth.tar"
    torch.save(state, file_path)
    if is_best:
        model_path = log_dir / cfg.model.arch
        shutil.copyfile(file_path, model_path.with_suffix(".pth.tar"))
        if cfg.wandb.use:
            # onnx export to display in W&B
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dummy = Variable(torch.randn(1, 3, 224, 224))
            dummy = dummy.to(device)
            torch.onnx.export(model.module, dummy, model_path.with_suffix(".onnx"))
            wandb.save(str(model_path.with_suffix(".onnx")))


def resume_from_checkpoint(
    cfg: config.TrainConfig, model: models.Model, optimizer: Optimizer
) -> tuple[models.Model, Optimizer]:
    if not cfg.compute.resume.is_file(cfg.compute.resume):
        raise FileNotFoundError(f"=> no checkpoint found at '{cfg.compute.resume}'")

    print(f"=> loading checkpoint '{cfg.compute.resume}'")
    if cfg.distributed.gpu is None:
        checkpoint = torch.load(cfg.compute.resume)
    else:
        # Map model to be loaded to specified single gpu.
        loc = f"cuda:{cfg.distributed.gpu}"
        checkpoint = torch.load(cfg.compute.resume, map_location=loc)
    cfg.compute.start_epoch = checkpoint["epoch"]
    best_score = checkpoint["best_score"]
    if cfg.distributed.gpu is not None:
        # best_score may be from a checkpoint from a different GPU
        best_score = best_score.to(cfg.distributed.gpu)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"=> loaded checkpoint '{cfg.compute.resume}' (epoch {checkpoint['epoch']})")

    return model, optimizer


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = 1.0e-8
        loss = self.criterion(x, y)
        loss = loss.mean(axis=1)
        loss += eps
        loss = torch.sqrt(loss)
        loss = loss.mean()
        return loss
