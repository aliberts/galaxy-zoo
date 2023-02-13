import time
from contextlib import suppress

import pyrallis
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from gzoo.domain import pipeline
from gzoo.domain.models import Model
from gzoo.infra.config import PredictConfig
from gzoo.infra.utils import AverageMeter, ProgressMeter


@pyrallis.wrap(config_path="config/predict.yaml")
def main(cfg: PredictConfig) -> None:
    if cfg.distributed.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.distributed.world_size = cfg.distributed.ngpus_per_node * cfg.distributed.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=cfg.distributed.ngpus_per_node,
            args=(cfg,),
        )
    else:
        # Simply call main_worker function
        main_worker(cfg.distributed.gpu, cfg)


def main_worker(gpu: int | None, cfg: PredictConfig) -> None:
    cfg.distributed.gpu = gpu
    cfg.distributed = pipeline.setup_distributed(gpu, cfg.distributed)

    model = pipeline.create_model(cfg)
    model, cfg.compute = pipeline.setup_cuda(model, cfg)
    model = pipeline.load_model(cfg, model)

    test_loader = pipeline.make_test_dataset(cfg)

    cudnn.benchmark = True

    validate(test_loader, model, cfg)


def validate(test_loader: DataLoader, model: Model, cfg: PredictConfig) -> None:
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

    with suppress(FileExistsError):
        cfg.dataset.predictions.mkdir(exist_ok=True)

    with torch.no_grad(), cfg.dataset.predictions.open("w") as out:
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

    print(f"predictions writen in: {cfg.dataset.predictions}")


if __name__ == "__main__":
    main()
