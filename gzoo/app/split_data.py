from pathlib import Path

import pandas as pd
import pyrallis
from sklearn.model_selection import train_test_split
from termgraph.module import Args, BarChart, Data
from wandb.sdk.interface.artifacts import Artifact
from wandb.sdk.wandb_run import Run

import wandb
from gzoo.infra.config import DatasetConfig, TrainConfig


@pyrallis.wrap()
def main(cfg: TrainConfig) -> None:

    if cfg.wandb.use:
        run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="data_split")

        dataset = run.use_artifact(f"{cfg.dataset.clf_name}:{cfg.dataset.version}")
        clf_labels_path = Path(dataset.get_path(cfg.dataset.clf_labels_file.name).download())
    else:
        clf_labels_path = cfg.dataset.clf_labels

    clf_labels_df = pd.read_csv(clf_labels_path, header=0, sep=",", index_col="GalaxyID")
    clf_labels_split_df = split_data(
        clf_labels_df, cfg.dataset.test_split_ratio, cfg.dataset.val_split_ratio
    )
    print_split_summary(clf_labels_split_df)
    clf_labels_split_df.to_csv(cfg.dataset.clf_labels_split)
    print(f"Data split written to {cfg.dataset.clf_labels_split}")

    if cfg.wandb.use:
        upload_data_split(cfg.dataset, run, dataset, clf_labels_split_df)


def split_data(
    labels: pd.DataFrame, test_split_ratio: float, val_split_ratio: float
) -> pd.DataFrame:
    """
    Returns a DataFrame similar to labels with an additional "Split" column which contains
    the partition name (train / val / test) for each row. The split is done according to
    the ratios given as input.
    """
    train_val_df, test_df = train_test_split(
        labels,
        test_size=test_split_ratio,
        random_state=0,
        stratify=labels["Class"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split_ratio,
        random_state=0,
        stratify=train_val_df["Class"],
    )

    train_df["Split"] = "train"
    val_df["Split"] = "val"
    test_df["Split"] = "test"
    df = pd.concat([train_df, val_df, test_df]).sort_index()
    return df


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
    bar_data = Data(chart_values, chart_labels)
    args = Args(width=20)
    print("----- labels distribution for each split (%) -----")
    BarChart(bar_data, args).draw()


def upload_data_split(
    cfg: DatasetConfig, run: Run, dataset: Artifact, data_split: pd.DataFrame
) -> None:
    """
    Uploads the data split to Weights & Biases under a new version of the dataset artifact.
    """
    eda_table = dataset.get(cfg.eda_table)
    dataset_path = Path(dataset.download())

    data_split = data_split.reset_index()
    data_split = data_split.rename(columns={"GalaxyID": "ID"})
    data_split_table = wandb.Table(dataframe=data_split[["ID", "Split"]])
    join_table = wandb.JoinedTable(eda_table, data_split_table, "ID")

    split_dataset = wandb.Artifact(cfg.clf_name, type="dataset")
    split_dataset.add_file(cfg.clf_labels_split)
    split_dataset.add_dir(dataset_path)
    split_dataset.add(join_table, cfg.eda_table_split)

    run.log_artifact(split_dataset)
    run.finish()


if __name__ == "__main__":
    main()
