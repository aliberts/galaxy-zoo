import os
import time
import warnings
from os import path as osp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from gzoo.domain import pipeline
from gzoo.infra import utils
from gzoo.infra.options import Options
from gzoo.infra.utils import AverageMeter, ProgressMeter


def main(yaml_path=None, run_cli=True):

    opt = Options(source=yaml_path, run_parser=run_cli)

    if opt.compute.seed is not None:
        opt.compute.seed = int(opt.compute.seed)
        utils.set_random_seed(opt.compute.seed)

    if not opt.compute.use_cuda:
        torch.cuda.is_available = lambda: False

    if opt.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism."
        )

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    opt.gpu = gpu

    if opt.gpu is not None:
        print(f"Use GPU: {opt.gpu} for training")

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method=opt.dist_url,
            world_size=opt.world_size,
            rank=opt.rank,
        )

    model, test_loader, criterion = pipeline.build_eval(opt, ngpus_per_node)
    validate(test_loader, model, criterion, opt)


def validate(test_loader, model, criterion, opt):
    batch_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(len(test_loader), [], prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    class_headers = [
        "GalaxyID",
        "completely_round_smooth",
        "in_between_smooth",
        "cigar_shaped_smooth",
        "edge_on",
        "spiral",
    ]

    output_file = osp.join(opt.dataset.dir, opt.output)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with torch.no_grad(), open(output_file, "w") as out:
        out.write(",".join(class_headers) + "\n")
        end = time.time()
        for i, (images, ids) in enumerate(test_loader):
            if opt.gpu is not None:
                images = images.cuda(opt.gpu, non_blocking=True)

            # compute output
            predictions = model(images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                progress.display(i)

            # write predictions
            for i, pred in enumerate(predictions):
                pred_id = ids[i] + ","
                out.write(pred_id + ",".join(map(str, pred.tolist())) + "\n")

    print(f"predictions writen in: {output_file}")


if __name__ == "__main__":
    main()
