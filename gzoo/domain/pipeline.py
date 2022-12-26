import shutil
from pathlib import Path
from typing import TypeAlias, Union

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from gzoo.domain.model import Model, create_model, load_model
from gzoo.infra.config import TrainConfig
from gzoo.infra.data import GalaxyTestSet, GalaxyTrainSet, imagenet

Loss: TypeAlias = Union[nn.CrossEntropyLoss, "RMSELoss"]


def setup_cuda(
    model: Model,
    cfg: TrainConfig,
):
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


def build_train(
    cfg: TrainConfig,
) -> tuple[Model, DataLoader, DataLoader, DistributedSampler | None, Loss, Optimizer]:
    model = create_model(cfg)
    model, cfg.compute = setup_cuda(model, cfg)

    # Make the data
    if cfg.dataset.name == "imagenet":
        train_dataset, val_dataset = imagenet(cfg)
    else:
        train_dataset = GalaxyTrainSet("train", cfg)
        val_dataset = GalaxyTrainSet("val", cfg)

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

    # Make loss function and optimizer
    # https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10
    if cfg.exp.task == "classification":
        n_samples = [8014, 7665, 550, 3708, 7416]
        # weights = [max(n_samples) / x for x in n_samples]
        weights = [1.0 - x / sum(n_samples) for x in n_samples]
        weights = torch.FloatTensor(weights).cuda(cfg.distributed.gpu)
        criterion = nn.CrossEntropyLoss(weight=weights).cuda(cfg.distributed.gpu)
    elif cfg.exp.task == "regression":
        criterion = RMSELoss().cuda(cfg.distributed.gpu)

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

    # Optionally resume from a checkpoint
    if cfg.compute.resume:
        model, optimizer = resume_from_checkpoint(cfg, model, optimizer)

    cudnn.benchmark = True
    return model, train_loader, val_loader, train_sampler, criterion, optimizer


def build_eval(cfg: TrainConfig) -> tuple[Model, DataLoader, Loss]:
    model = create_model(cfg)
    model, cfg.compute = setup_cuda(model, cfg)
    model = load_model(cfg, model)

    test_dataset = GalaxyTestSet(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.compute.batch_size,
        shuffle=False,
        num_workers=cfg.compute.workers,
        pin_memory=True,
    )
    # Make loss function and optimizer
    if cfg.exp.task == "classification":
        criterion = nn.CrossEntropyLoss().cuda(cfg.distributed.gpu)
    elif cfg.exp.task == "regression":
        criterion = RMSELoss().cuda(cfg.distributed.gpu)

    cudnn.benchmark = True
    return model, test_loader, criterion


def save_checkpoint(
    log_dir: Path, state: dict, model: Model, is_best: bool, cfg: TrainConfig
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


def resume_from_checkpoint(cfg: TrainConfig, model: Model, optimizer):
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

    def forward(self, x, y):
        eps = 1.0e-8
        loss = self.criterion(x, y)
        loss = loss.mean(axis=1)
        loss += eps
        loss = torch.sqrt(loss)
        loss = loss.mean()
        return loss
