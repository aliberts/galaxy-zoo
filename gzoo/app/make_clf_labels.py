"""
Make Classification Labels
Run this script to generate the labels used for the classification version of the problem.
"""
import pandas as pd
import pyrallis

from gzoo.infra import config, data


@pyrallis.wrap()
def main(cfg: config.DatasetConfig) -> None:
    reg_labels = pd.read_csv(cfg.reg_labels, sep=",", index_col="GalaxyID")
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

    # Copy them to a "classification" folder
    image_list = clf_labels.index.to_list()
    reg_dataset = data.GalaxyRawSet(cfg.reg_images_train)
    reg_dataset.copy_to(cfg.clf_images_raw, image_list)

    clf_labels.to_csv(cfg.clf_labels, sep=",")
    print(f"Classification labels writen to {cfg.clf_labels}.")


if __name__ == "__main__":
    main()
