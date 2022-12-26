from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SPLIT_RATIO = 0.1

ABSTRACT = (
    "Make Classification Labels\n\n"
    "Run this script to generate the labels used for the\n"
    "classification version of the problem.\n"
    "To use this tool, please provide path to the dataset\n"
    "directory (which contains `training_solutions_rev1.csv`)\n"
    "as argument.\n"
)

parser = ArgumentParser(description=ABSTRACT, formatter_class=RawTextHelpFormatter)
parser.add_argument("data_dir", type=Path, metavar="PATH")


def main():
    args = parser.parse_args()
    in_path = args.data_dir / "training_solutions_rev1.csv"
    reg_labels = pd.read_csv(in_path, sep=",", index_col="GalaxyID")
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
    clf_labels = clf_labels.astype(int)
    clf_labels = clf_labels[clf_labels.sum(axis=1) > 0.0]

    clf_labels_train_val, clf_labels_test = train_test_split(
        clf_labels,
        test_size=TEST_SPLIT_RATIO,
        random_state=0,
        stratify=clf_labels,
    )
    print("--- train/val labels distribution ---")
    print(clf_labels_train_val.sum())
    print(f"\ntotal examples: {clf_labels_train_val.sum().sum()}\n")
    print("----- test labels distribution -----")
    print(clf_labels_test.sum())
    print(f"\ntotal examples: {clf_labels_test.sum().sum()}\n")

    # actually write one column with the number of classes
    clf_labels_train_val = get_classes_number(clf_labels_train_val)
    clf_labels_test = get_classes_number(clf_labels_test)

    out_path = args.data_dir / "classification_labels_train_val.csv"
    clf_labels_train_val.to_csv(out_path, sep=",")
    print(f"classification labels writen to {out_path}.")
    out_path = args.data_dir / "classification_labels_test.csv"
    clf_labels_test.to_csv(out_path, sep=",")
    print(f"classification labels writen to {out_path}.")


def get_classes_number(df):
    df.columns = np.arange(0, len(df.columns))
    df = df.dot(df.columns.T)
    return df


if __name__ == "__main__":
    main()
