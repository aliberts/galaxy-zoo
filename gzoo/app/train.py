"""
Inspired by https://github.com/pytorch/examples/blob/main/imagenet/main.py
"""

import logging
import os
import time
import warnings
from dataclasses import asdict

import pyrallis
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import wandb
from gzoo.domain import pipeline
from gzoo.infra import utils
from gzoo.infra.config import TrainConfig
from gzoo.infra.logging import Log
from gzoo.infra.utils import AverageMeter, ProgressMeter

NB_CLASSES = 5

best_score = 0
train_example_ct = 0
train_batch_ct = 0
val_example_ct = 0
val_batch_ct = 0


@pyrallis.wrap(config_path="config/train.yaml")
def main(opt: TrainConfig):

    # To update the config file, uncomment this line
    # pyrallis.dump(opt, open('config/train.yaml','w'))

    if opt.wandb.use:
        wandb.login()

    log = Log("train", opt)
    log.toggle()
    logging.debug("arguments:")
    logging.debug(opt)

    if opt.compute.seed is not None:
        opt.compute.seed = int(opt.compute.seed)
        utils.set_random_seed(opt.compute.seed)

    if not opt.compute.use_cuda:
        torch.cuda.is_available = lambda: False

    if opt.distributed.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism."
        )

    if opt.distributed.dist_url == "env://" and opt.distributed.world_size == -1:
        opt.distributed.world_size = int(os.environ["WORLD_SIZE"])

    opt.distribute = opt.distributed.world_size > 1 or opt.distributed.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if opt.distributed.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.distributed.world_size = ngpus_per_node * opt.distributed.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.distributed.gpu, ngpus_per_node, opt, log.dir)


def main_worker(gpu, ngpus_per_node, opt, log_dir):
    opt.distributed.gpu = gpu

    if opt.distributed.gpu is not None:
        logging.info(f"Use GPU: {opt.distributed.gpu} for training")

    if opt.distribute:
        if opt.distributed.dist_url == "env://" and opt.distributed.rank == -1:
            opt.distributed.rank = int(os.environ["RANK"])
        if opt.distributed.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.distributed.rank = opt.distributed.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=opt.distributed.dist_backend,
            init_method=opt.distributed.dist_url,
            world_size=opt.distributed.world_size,
            rank=opt.distributed.rank,
        )

    model, train_loader, val_loader, train_sampler, criterion, optimizer = pipeline.build_train(
        opt, ngpus_per_node
    )

    if opt.exp.evaluate:
        validate(val_loader, model, criterion, opt)
        return

    if opt.wandb.use:
        with wandb.init(
            name=opt.wandb.run_name,
            project=opt.wandb.project,
            entity=opt.wandb.entity,
            notes=opt.wandb.note,
            tags=opt.wandb.tags,
            config=asdict(opt),
        ):
            if opt.wandb.run_name is not None:
                wandb.run_name = opt.wandb.run_name

            wandb.watch(model, criterion, log="all", log_freq=10)
            for epoch in range(opt.compute.start_epoch, opt.compute.epochs):
                train_loop(
                    train_loader,
                    val_loader,
                    train_sampler,
                    model,
                    criterion,
                    optimizer,
                    epoch,
                    opt,
                    ngpus_per_node,
                    log_dir,
                )
    else:
        for epoch in range(opt.compute.start_epoch, opt.compute.epochs):
            train_loop(
                train_loader,
                val_loader,
                train_sampler,
                model,
                criterion,
                optimizer,
                epoch,
                opt,
                ngpus_per_node,
                log_dir,
            )


def train_loop(
    train_loader,
    val_loader,
    train_sampler,
    model,
    criterion,
    optimizer,
    epoch,
    opt,
    ngpus_per_node,
    log_dir,
):
    global best_score
    if opt.distribute:
        train_sampler.set_epoch(epoch)

    adjust_learning_rate(optimizer, epoch, opt)

    # train for one epoch
    train_loss, train_acc1, train_acc3 = train(
        train_loader, model, criterion, optimizer, epoch, opt
    )

    # evaluate on validation set
    val_loss, val_acc1, val_acc3 = validate(val_loader, model, criterion, opt)

    # remember best score and save checkpoint
    score = val_acc1
    is_best = score > best_score
    best_score = max(score, best_score)

    # Report metrics to w&b at a epoch time-step
    if opt.wandb.use:
        metrics = {
            "epoch": epoch,
            "train-loss": train_loss,
            "val-loss": val_loss,
        }
        if opt.exp.task == "classification":
            tmp = {
                "train-acc1": train_acc1,
                "train-acc3": train_acc3,
                "val-acc1": val_acc1,
                "val-acc3": val_acc3,
            }
            metrics = {**metrics, **tmp}
        wandb.log(metrics)
        wandb.run.summary["accuracy"] = best_score

    if (
        not opt.distributed.multiprocessing_distributed
        or (
            opt.distributed.multiprocessing_distributed
            and opt.distributed.rank % ngpus_per_node == 0
        )
    ) and opt.model.arch != "random":
        checkpoint = {
            "epoch": epoch + 1,
            "arch": opt.model.arch,
            "state_dict": model.state_dict(),
            "best_score": best_score,
            "optimizer": optimizer.state_dict(),
        }
        pipeline.save_checkpoint(
            log_dir,
            checkpoint,
            model,
            is_best,
            opt,
        )


def train(train_loader, model, criterion, optimizer, epoch, opt):
    global train_example_ct, train_batch_ct
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    metrics = [batch_time, data_time, losses]
    if opt.exp.task == "classification":
        top1 = AverageMeter("Acc@1", ":6.2f")
        top3 = AverageMeter("Acc@3", ":6.2f")
        metrics.extend([top1, top3])
    progress = ProgressMeter(len(train_loader), metrics, prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt.distributed.gpu is not None:
            images = images.cuda(opt.distributed.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.distributed.gpu, non_blocking=True)

        if opt.exp.task == "classification":
            target = target.view(-1)

        # forward pass
        output = model(images)
        loss = criterion(output, target)

        # measure record loss and accuracy
        losses.update(loss.item(), images.size(0))
        if opt.exp.task == "classification":
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            top1.update(acc1[0], images.size(0))
            top3.update(acc3[0], images.size(0))

        # Report metrics to w&b every n-th batch
        train_example_ct += len(images)
        train_batch_ct += 1
        if opt.wandb.use and ((train_batch_ct + 1) % opt.wandb.freq) == 0:
            batch_metrics = {
                "train_example_ct": train_example_ct,
                "train-loss (batch)": loss.to("cpu"),
            }
            if opt.exp.task == "classification":
                tmp = {
                    "train-acc1 (batch)": acc1.to("cpu"),
                    "train-acc3 (batch)": acc3.to("cpu"),
                }
                batch_metrics = {**batch_metrics, **tmp}
            wandb.log(batch_metrics)

        # backward pass and gradient descent
        if opt.model.arch != "random":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.compute.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg, top3.avg


def validate(val_loader, model, criterion, opt):
    global val_example_ct, val_batch_ct
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    metrics = [batch_time, losses]
    if opt.exp.task == "classification":
        top1 = AverageMeter("Acc@1", ":6.2f")
        top3 = AverageMeter("Acc@3", ":6.2f")
        metrics.extend([top1, top3])
    progress = ProgressMeter(len(val_loader), metrics, prefix="Test:")
    confusion_matrix = torch.zeros(NB_CLASSES, NB_CLASSES)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if opt.distributed.gpu is not None:
                images = images.cuda(opt.distributed.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.distributed.gpu, non_blocking=True)

            if opt.exp.task == "classification":
                target = target.view(-1)

            # forward pass
            output = model(images)
            loss = criterion(output, target)

            # measure record loss and accuracy
            losses.update(loss.item(), images.size(0))
            if opt.exp.task == "classification":
                acc1, acc3 = accuracy(output, target, topk=(1, 3))
                top1.update(acc1[0], images.size(0))
                top3.update(acc3[0], images.size(0))

            # confusion matrix
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            val_example_ct += len(images)
            val_batch_ct += 1
            if opt.wandb.use and ((val_batch_ct + 1) % opt.wandb.freq) == 0:
                batch_metrics = {
                    "val_example_ct": val_example_ct,
                    "val-loss (batch)": loss.to("cpu"),
                }
                if opt.exp.task == "classification":
                    tmp = {
                        "val-acc1 (batch)": acc1.to("cpu"),
                        "val-acc3 (batch)": acc3.to("cpu"),
                    }
                    batch_metrics = {**batch_metrics, **tmp}
                wandb.log(batch_metrics)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.compute.print_freq == 0:
                progress.display(i)

        logging.info(f" * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Loss {losses.avg:.3f}")
        logging.info(f"\n{confusion_matrix}")
        logging.info(f"\n{confusion_matrix.diag()/confusion_matrix.sum(1)}")

    return losses.avg, top1.avg, top3.avg


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if optimizer is None:
        return
    lr = opt.optimizer.lr * (0.1 ** (epoch // opt.optimizer.lr_scheduler_freq))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_reg(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        pred = output > 0.5
        targ = target > 0.5
        correct = pred.eq(targ)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
