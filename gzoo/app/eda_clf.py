"""
This script will upload the classification dataset to our Weights & Biases
project in order to perform EDA there.
"""
import pandas as pd
import pyrallis
from fastai.data.transforms import get_image_files
from PIL import Image
from tqdm import tqdm

import wandb
from gzoo.infra.config import EDAConfig

DEBUG = False


def create_wandb_table(cfg: EDAConfig, image_files):
    "Create a table with the dataset"
    class_names = cfg.dataset.class_names
    labels_df = pd.read_csv(cfg.dataset.labels, header=0, sep=",", index_col="GalaxyID")
    table = wandb.Table(columns=["ID", "Image", "Dataset", "Class"])

    print("Creating EDA table")
    for image_file in tqdm(image_files):
        image = Image.open(image_file)
        image_label = labels_df.loc[int(image_file.stem), "label"]
        table.add_data(
            image_file.stem,
            wandb.Image(image),
            cfg.dataset.name,
            class_names[image_label],
        )

    return table


@pyrallis.wrap()
def main(cfg: EDAConfig) -> None:

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="upload")
    raw_data_at = wandb.Artifact(cfg.dataset.name, type="raw_data")
    raw_data_at.add_file(cfg.dataset.train_labels, name=cfg.dataset.train_labels_file.name)
    raw_data_at.add_dir(cfg.dataset.images_clf, name=cfg.dataset.images_clf_dir.name)

    image_files = get_image_files(cfg.dataset.images_clf, recurse=False)
    if DEBUG:
        image_files = image_files[:10]
    table = create_wandb_table(cfg, image_files)

    raw_data_at.add(table, "eda_table")
    run.log_artifact(raw_data_at, aliases=["classification"])
    run.finish()


if __name__ == "__main__":
    main()
