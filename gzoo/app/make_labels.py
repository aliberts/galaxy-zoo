"""
Make Classification Labels
Run this script to generate the labels used for the classification version of the problem.
"""

import shutil
from pathlib import Path

import pandas as pd
import pyrallis
from tqdm import tqdm

from gzoo.infra.config import TrainConfig


@pyrallis.wrap(config_path="config/train.yaml")
def main(cfg: TrainConfig) -> None:
    reg_labels = pd.read_csv(cfg.dataset.reg_labels, sep=",", index_col="GalaxyID")
    clf_labels = pd.DataFrame()
    clf_labels = clf_labels.assign(
        # fmt: off
        **{
            # See https://arxiv.org/pdf/1308.3496.pdf, Table 3
            "completely_round_smooth":
                (reg_labels["Class1.1"] >= 0.469) & (reg_labels["Class7.1"] >= 0.5),
            "in_between_smooth":
                (reg_labels["Class1.1"] >= 0.469) & (reg_labels["Class7.2"] >= 0.5),
            "cigar_shaped_smooth":
                (reg_labels["Class1.1"] >= 0.469) & (reg_labels["Class7.3"] >= 0.5),
            "edge_on":
                (reg_labels["Class1.2"] >= 0.430) & (reg_labels["Class2.1"] >= 0.602),
            "spiral":
                (reg_labels["Class1.2"] >= 0.430) & (reg_labels["Class2.2"] >= 0.715)
                & (reg_labels["Class4.1"] >= 0.619),
        }
        # fmt: on
    )

    # Select only rows that have been assigned a class based on the above rule
    clf_labels = clf_labels.astype(int)
    clf_labels = clf_labels[clf_labels.sum(axis=1) > 0.0]
    clf_labels = clf_labels.idxmax(axis=1)
    clf_labels.name = "Class"

    image_list = clf_labels.index.to_list()
    copy_images(image_list, cfg.dataset.reg_images_train, cfg.dataset.clf_images)

    clf_labels.to_csv(cfg.dataset.clf_labels, sep=",")
    print(f"Classification labels writen to {cfg.dataset.clf_labels}.")

    # clf_labels_train_val, clf_labels_test = train_test_split(
    #     clf_labels,
    #     test_size=cfg.dataset.test_split_ratio,
    #     random_state=0,
    #     stratify=clf_labels,
    # )
    # print("--- train/val labels distribution ---")
    # print(clf_labels_train_val.sum())
    # print(f"\ntotal examples: {clf_labels_train_val.sum().sum()}\n")
    # print("----- test labels distribution -----")
    # print(clf_labels_test.sum())
    # print(f"\ntotal examples: {clf_labels_test.sum().sum()}\n")

    # # Actually write one column with the number of classes
    # clf_labels_train_val = get_classes_number(clf_labels_train_val)
    # clf_labels_test = get_classes_number(clf_labels_test)

    # clf_labels_train_val.to_csv(cfg.dataset.train_labels, sep=",")
    # print(f"classification labels writen to {cfg.dataset.train_labels}.")
    # clf_labels_test.to_csv(cfg.dataset.test_labels, sep=",")
    # print(f"classification labels writen to {cfg.dataset.test_labels}.")


def copy_images(image_names: list[int], from_: Path, to_: Path) -> None:
    to_.mkdir(exist_ok=True)
    print(f"Copying classification-labeled images from {from_} to {to_}")
    for image in tqdm(image_names):
        image_file = Path(str(image)).with_suffix(".jpg")
        file_in = from_ / image_file
        file_out = to_ / image_file
        shutil.copy(file_in, file_out)


if __name__ == "__main__":
    main()
