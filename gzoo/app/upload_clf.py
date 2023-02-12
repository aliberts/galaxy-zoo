"""
This script will upload the classification dataset to our Weights & Biases
project in order to perform EDA there.
"""
import pandas as pd
import pyrallis

import wandb
from gzoo.infra import config, utils


@pyrallis.wrap()
def main(cfg: config.UploadConfig) -> None:

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="upload")
    dataset = wandb.Artifact("clf_raw", type="dataset")

    dataset.add_file(cfg.dataset.clf_labels)
    dataset.add_dir(cfg.dataset.clf_images_raw, name=cfg.dataset.clf_images_raw_dir.name)

    labels_df = pd.read_csv(cfg.dataset.clf_labels, header=0, sep=",", index_col="GalaxyID")
    labels_df["Dataset"] = "clf_raw"
    table = utils.make_wandb_image_table(labels_df, cfg.dataset.clf_images_raw)

    dataset.add(table, cfg.dataset.raw_table)
    run.log_artifact(dataset)
    run.finish()


if __name__ == "__main__":
    main()
