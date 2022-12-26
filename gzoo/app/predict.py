import os
import time
from contextlib import suppress

import pyrallis
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from gzoo.domain import pipeline
from gzoo.infra.config import PredictConfig
from gzoo.infra.utils import AverageMeter, ProgressMeter


@pyrallis.wrap(config_path="config/predict.yaml")
def main(cfg: PredictConfig):

    if cfg.distributed.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.distributed.world_size = cfg.distributed.ngpus_per_node * cfg.distributed.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=cfg.distributed.ngpus_per_node,
            args=(cfg.distributed.use, cfg.distributed.ngpus_per_node, cfg),
        )
    else:
        # Simply call main_worker function
        main_worker(cfg.distributed.gpu, cfg)


def main_worker(gpu, cfg: PredictConfig):
    cfg.distributed.gpu = gpu

    if cfg.distributed.gpu is not None:
        print(f"Use GPU: {cfg.distributed.gpu} for training")

    if cfg.distributed.use:
        if cfg.distributed.dist_url == "env://" and cfg.distributed.rank == -1:
            cfg.distributed.rank = int(os.environ["RANK"])
        if cfg.distributed.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.distributed.rank = cfg.distributed.rank * cfg.distributed.ngpus_per_node + gpu
        dist.init_process_group(
            backend=cfg.distributed.dist_backend,
            init_method=cfg.distributed.dist_url,
            world_size=cfg.distributed.world_size,
            rank=cfg.distributed.rank,
        )

    model, test_loader, _ = pipeline.build_eval(cfg)
    validate(test_loader, model, cfg)


def validate(test_loader, model, cfg: PredictConfig):
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

    output_file = cfg.dataset.dir / cfg.output
    with suppress(FileExistsError):
        output_file.mkdir(exist_ok=True)

    with torch.no_grad(), output_file.open("w") as out:
        out.write(",".join(class_headers) + "\n")
        end = time.time()
        for i, (images, ids) in enumerate(test_loader):
            if cfg.distributed.gpu is not None:
                images = images.cuda(cfg.distributed.gpu, non_blocking=True)

            # compute output
            predictions = model(images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.compute.print_freq == 0:
                progress.display(i)

            # write predictions
            for i, pred in enumerate(predictions):
                pred_id = ids[i] + ","
                out.write(pred_id + ",".join(map(str, pred.tolist())) + "\n")

    print(f"predictions writen in: {output_file}")


if __name__ == "__main__":
    main()
