import shutil
from os import path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import wandb
from gzoo.domain.model import create_model, load_model
from gzoo.infra import utils
from gzoo.infra.config import TrainConfig
from gzoo.infra.data import GalaxyTestSet, GalaxyTrainSet, imagenet


def build_train(cfg: TrainConfig, distribute: bool, ngpus_per_node: int):
    model = create_model(cfg)
    model, cfg.compute = utils.setup_cuda(
        model, cfg.compute, cfg.model.arch, cfg.distributed.gpu, distribute, ngpus_per_node
    )

    # Make the data
    if cfg.dataset.name == "imagenet":
        train_dataset, val_dataset = imagenet(cfg)
    else:
        train_dataset = GalaxyTrainSet("train", cfg)
        val_dataset = GalaxyTrainSet("val", cfg)

    if distribute:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
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


def build_eval(cfg: TrainConfig, distribute: bool, ngpus_per_node: int):
    model = create_model(cfg)
    model, cfg.compute = utils.setup_cuda(
        model, cfg.compute, cfg.model.arch, cfg.distributed.gpu, distribute, ngpus_per_node
    )
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


def save_checkpoint(log_dir, state: dict, model, is_best: bool, cfg: TrainConfig) -> None:
    filename = osp.join(log_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        model_name = osp.join(log_dir, cfg.model.arch)
        shutil.copyfile(filename, model_name + ".pth.tar")
        if cfg.wandb.use:
            # onnx export to display in W&B
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dummy = Variable(torch.randn(1, 3, 224, 224))
            dummy = dummy.to(device)
            torch.onnx.export(model.module, dummy, model_name + ".onnx")
            wandb.save(model_name + ".onnx")


def resume_from_checkpoint(cfg: TrainConfig, model, optimizer):
    if not osp.isfile(cfg.compute.resume):
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
