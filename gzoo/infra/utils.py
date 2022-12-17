import os
import random
import warnings

import numpy
import torch
import torch.backends.cudnn as cudnn

from gzoo.infra.config import ComputeConfig, DistributedConfig


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


def setup_distribute(config: DistributedConfig) -> tuple[bool, DistributedConfig]:
    if config.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism."
        )
    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])
    distribute = config.world_size > 1 or config.multiprocessing_distributed
    return distribute, config.world_size


def setup_cuda(
    model,
    compute_config: ComputeConfig,
    model_name: str,
    gpu: int,
    distribute: bool,
    ngpus_per_node: int,
):
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif distribute:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            compute_config.batch_size = int(compute_config.batch_size / ngpus_per_node)
            compute_config.workers = int(
                (compute_config.workers + ngpus_per_node - 1) / ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if model_name.startswith("alexnet") or model_name.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model, compute_config


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
