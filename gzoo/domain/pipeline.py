import shutil
from os import path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

import wandb
from gzoo.domain.model import create_model, load_model
from gzoo.infra import utils
from gzoo.infra.data import GalaxyTestSet, GalaxyTrainSet, imagenet


def build_train(opt, ngpus_per_node):
    model = create_model(opt)
    model, opt = utils.setup_cuda(model, opt, ngpus_per_node)

    # Make the data
    if opt.dataset.name == "imagenet":
        train_dataset, val_dataset = imagenet(opt)
    else:
        train_dataset = GalaxyTrainSet("train", opt)
        val_dataset = GalaxyTrainSet("val", opt)

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        num_workers=opt.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
    )

    # Make loss function and optimizer
    # https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10
    if opt.task == "classification":
        n_samples = [8014, 7665, 550, 3708, 7416]
        # weights = [max(n_samples) / x for x in n_samples]
        weights = [1.0 - x / sum(n_samples) for x in n_samples]
        weights = torch.FloatTensor(weights).cuda(opt.gpu)
        criterion = nn.CrossEntropyLoss(weight=weights).cuda(opt.gpu)
    elif opt.task == "regression":
        criterion = RMSELoss().cuda(opt.gpu)

    if opt.model.arch == "random":
        optimizer = None
    elif opt.optimizer.name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), opt.optimizer.lr)
    elif opt.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            opt.optimizer.lr,
            momentum=opt.optimizer.momentum,
            weight_decay=opt.optimizer.weight_decay,
        )

    # optionally resume from a checkpoint
    if opt.resume:
        model, optimizer = resume_from_checkpoint(opt, model, optimizer)

    cudnn.benchmark = True
    return model, train_loader, val_loader, train_sampler, criterion, optimizer


def build_eval(opt, ngpus_per_node):
    model = create_model(opt)
    model, opt = utils.setup_cuda(model, opt, ngpus_per_node)
    model = load_model(opt, model)

    test_dataset = GalaxyTestSet(opt)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
    )
    # Make loss function and optimizer
    if opt.task == "classification":
        criterion = nn.CrossEntropyLoss().cuda(opt.gpu)
    elif opt.task == "regression":
        criterion = RMSELoss().cuda(opt.gpu)

    cudnn.benchmark = True
    return model, test_loader, criterion


def save_checkpoint(log_dir, state, model, is_best, opt):
    filename = osp.join(log_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        model_name = osp.join(log_dir, opt.model.arch)
        shutil.copyfile(filename, model_name + ".pth.tar")
        if opt.wandb.use:
            # onnx export to display in W&B
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dummy = Variable(torch.randn(1, 3, 224, 224))
            dummy = dummy.to(device)
            torch.onnx.export(model.module, dummy, model_name + ".onnx")
            wandb.save(model_name + ".onnx")


def resume_from_checkpoint(opt, model, optimizer):
    if not osp.isfile(opt.resume):
        raise FileNotFoundError(f"=> no checkpoint found at '{opt.resume}'")

    print(f"=> loading checkpoint '{opt.resume}'")
    if opt.gpu is None:
        checkpoint = torch.load(opt.resume)
    else:
        # Map model to be loaded to specified single gpu.
        loc = f"cuda:{opt.gpu}"
        checkpoint = torch.load(opt.resume, map_location=loc)
    opt.start_epoch = checkpoint["epoch"]
    best_score = checkpoint["best_score"]
    if opt.gpu is not None:
        # best_score may be from a checkpoint from a different GPU
        best_score = best_score.to(opt.gpu)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"=> loaded checkpoint '{opt.resume}' (epoch {checkpoint['epoch']})")

    return model, optimizer


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x, y):
        eps = 1.0e-8
        loss = self.criterion(x, y)
        loss = loss.mean(axis=1)
        loss += eps
        loss = torch.sqrt(loss)
        loss = loss.mean()
        return loss
