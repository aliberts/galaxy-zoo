import logging
import pprint
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import termgraph.module as tg
from PIL import Image
from tqdm import tqdm

import wandb
from gzoo.infra import config, data
from gzoo.infra.logging import Log


def setup_wandb_training_run(cfg: config.TrainConfig) -> wandb.run:
    resume = "must" if cfg.compute.resume is not None else None
    run = wandb.init(
        name=cfg.wandb.run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        notes=cfg.wandb.note,
        tags=cfg.wandb.tags,
        job_type="train",
        resume=resume,
        config=asdict(cfg),
    )
    return run


def setup_wandb_split_run(cfg: config.SplitConfig) -> wandb.run:
    test_ratio = cfg.dataset.test_split_ratio
    train_val_ratio = 1.0 - test_ratio
    val_ratio = train_val_ratio * cfg.dataset.val_split_ratio
    train_ratio = 1.0 - val_ratio
    note = f"test ({test_ratio:.0%}) / val ({val_ratio:.0%}) / train ({train_ratio:.0%})"
    split_config = {
        "from_raw": cfg.from_raw,
        "test_ratio": cfg.dataset.test_split_ratio,
        "val_ratio": cfg.dataset.val_split_ratio,
        "split_seed": cfg.seed,
    }
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        notes=note,
        job_type="data_split",
        config=split_config,
    )
    return run


def setup_train_log(cfg: config.TrainConfig) -> Log:
    log = Log("train", cfg.exp, cfg.model.arch)
    log.toggle()
    logging.debug("arguments:")
    logging.debug(pprint.pformat(asdict(cfg)))
    return log


def pil_loader(path: Path) -> Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with path.open("rb") as f, Image.open(f) as img:
        return img.convert("RGB")


def copy_images(image_names: list[int], from_: Path, to_: Path, suffix: str = ".jpg") -> None:
    print(f"Copying images from {from_} to {to_}")
    for image in tqdm(image_names):
        image_file = Path(str(image)).with_suffix(suffix)
        file_in = from_ / image_file
        file_out = to_ / image_file
        shutil.copy(file_in, file_out)


def purge_images(dir: Path, types: list[str] | None = None) -> None:
    if types is None:
        types = [".jpg"]
    image_list = [f for type in types for f in dir.glob(f"*{type}")]
    print(f"Removing images from {dir}")
    for image_path in tqdm(image_list):
        image_path.unlink()


def make_wandb_image_table(df: pd.DataFrame, image_folder: Path) -> wandb.Table:
    "Creates a W&B table from df and appends it a column image. Can take a while"

    df = df.sort_index()
    image_name_list = df.index.to_list()
    table = wandb.Table(dataframe=df.reset_index())
    dataset = data.GalaxyRawSet(image_folder)
    image_list = []

    print("Creating EDA table")
    for image_name in tqdm(image_name_list):
        image = dataset.get_pil(image_name)
        image_list.append(wandb.Image(image))
        image.close()
    table.add_column("Image", image_list)

    return table


def print_split_summary(labels_split: pd.DataFrame) -> None:
    """
    Displays a quick summary of the data split and the class distributions.
    """
    split_nb = labels_split.groupby(["Split", "Class"]).size()
    split_ratio = 100 * split_nb / split_nb.groupby("Split").transform("sum")
    split_ratio = split_ratio.apply(lambda x: round(x, 1))
    summary = pd.concat([split_nb, split_ratio], axis=1)
    summary.columns = ["examples", "% per split"]
    for split in ["train", "val", "test"]:
        total = labels_split.groupby(["Split"]).size().loc[split]
        print(f"----- {split} labels ({total}) -----")
        print(summary.loc[split], "\n")

    print_split_distribution(split_ratio)


def print_split_distribution(split_ratio: pd.DataFrame) -> None:
    """
    Displays class distributions on a horizontal bar graph.
    """
    chart_values = [[x] for x in split_ratio["test"].to_list()]
    chart_labels = split_ratio["test"].index.to_list()
    bar_data = tg.Data(chart_values, chart_labels)
    args = tg.Args(width=20)
    print("----- labels distribution for each split (%) -----")
    tg.BarChart(bar_data, args).draw()
    print("")


def make_dataset_artifact(name: str, items: list[Path]) -> wandb.Artifact:
    artifact = wandb.Artifact(name, type="dataset")
    for item in items:
        if item.is_file():
            artifact.add_file(item)
        elif item.is_dir():
            artifact.add_dir(item, name=item.name)

    return artifact


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
