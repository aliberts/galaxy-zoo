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
from gzoo.infra.config import UploadConfig


@pyrallis.wrap()
def main(cfg: UploadConfig) -> None:

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="upload")
    dataset = wandb.Artifact(cfg.dataset.clf_name, type="dataset")

    dataset.add_file(cfg.dataset.clf_labels, name=cfg.dataset.clf_labels_file.name)
    dataset.add_dir(cfg.dataset.clf_images, name=cfg.dataset.clf_images_dir.name)

    image_files = get_image_files(cfg.dataset.clf_images, recurse=False)
    if cfg.debug:
        image_files = image_files[:10]
    table = create_wandb_table(cfg, image_files)

    dataset.add(table, "eda_table")
    run.log_artifact(dataset, aliases=["classification"])
    run.finish()


def create_wandb_table(cfg: UploadConfig, image_files):
    "Create a table with the dataset"
    labels_df = pd.read_csv(cfg.dataset.clf_labels, header=0, sep=",", index_col="GalaxyID")
    table = wandb.Table(columns=["ID", "Image", "Dataset", "Class"])

    print("Creating EDA table")
    for image_file in tqdm(image_files):
        image = Image.open(image_file)
        image_label = labels_df.loc[int(image_file.stem), "Class"]
        table.add_data(
            image_file.stem,
            wandb.Image(image),
            cfg.dataset.name,
            image_label,
        )

    return table


if __name__ == "__main__":
    main()
