from pathlib import Path

import pandas as pd
import pyrallis
from sklearn.model_selection import train_test_split

import wandb
from gzoo.infra import config, data, utils


@pyrallis.wrap()
def main(cfg: config.SplitConfig) -> None:
    if cfg.wandb.use:
        run = utils.setup_wandb_split_run(cfg)
        parent_name = "clf_raw" if cfg.from_raw else "clf_train_val"
        parent_artifact = run.use_artifact(f"{parent_name}:{cfg.dataset.version}")
        dataset_path = Path(parent_artifact.download())
    else:
        dataset_path = cfg.dataset.clf

    if cfg.from_raw:  # Test / train / val split
        clf_labels_path = dataset_path / cfg.dataset.clf_labels_file
        clf_labels_df = pd.read_csv(clf_labels_path, header=0, sep=",", index_col="GalaxyID")
        clf_test_df, clf_train_val_df = split_test_train_val(
            clf_labels_df, cfg.dataset.test_split_ratio, cfg.dataset.val_split_ratio, cfg.seed
        )
    else:  # Reshuffle train / val
        clf_labels_path = dataset_path / cfg.dataset.clf_labels_train_val_file
        clf_labels_df = pd.read_csv(clf_labels_path, header=0, sep=",", index_col="GalaxyID")
        clf_train_val_df = shuffle_train_val(clf_labels_df, cfg.dataset.val_split_ratio, cfg.seed)
        clf_labels_test_path = dataset_path / cfg.dataset.clf_labels_test_file
        clf_test_df = pd.read_csv(clf_labels_test_path, header=0, sep=",", index_col="GalaxyID")

    clf_whole_df = pd.concat([clf_test_df, clf_train_val_df]).sort_index()
    utils.print_split_summary(clf_whole_df)
    write_data_split(clf_test_df, clf_train_val_df, dataset_path, cfg.dataset, cfg.from_raw)

    if cfg.wandb.use:
        write_artifacts(clf_whole_df, parent_artifact, cfg.dataset, cfg.from_raw, run)


def split_test_train_val(
    labels: pd.DataFrame, test_split_ratio: float, val_split_ratio: float, seed: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns 2 DataFrame (test and train_val) similar to labels with an additional "Split"
    column which contains the partition name (train / val / test) for each row.
    The split is done according to the ratios given as input.
    """
    train_val_df, test_df = train_test_split(
        labels,
        test_size=test_split_ratio,
        random_state=seed,
        stratify=labels["Class"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split_ratio,
        random_state=seed,
        stratify=train_val_df["Class"],
    )

    train_df["Split"] = "train"
    val_df["Split"] = "val"
    test_df["Split"] = "test"
    train_val_df = pd.concat([train_df, val_df]).sort_index()
    return test_df, train_val_df


def shuffle_train_val(
    labels: pd.DataFrame, val_split_ratio: float, seed: int | None
) -> pd.DataFrame:
    """
    Returns a DataFrame similar to labels with an additional "Split" column which contains
    the partition name (train & val / test) for each row. The split is done according to
    the ratios given as input.
    """
    train_df, val_df = train_test_split(
        labels,
        test_size=val_split_ratio,
        random_state=seed,
        stratify=labels["Class"],
    )

    train_df["Split"] = "train"
    val_df["Split"] = "val"
    df = pd.concat([train_df, val_df]).sort_index()
    return df


def write_data_split(
    test_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
    dataset_dir: Path,
    cfg: config.DatasetConfig,
    from_raw: bool,
) -> None:

    if from_raw:
        # Purge previous split image directories
        utils.purge_images(cfg.clf_images_test)
        utils.purge_images(cfg.clf_images_train_val)

        # Copy test and train_val images to respective directories
        clf_images_path = dataset_dir / cfg.dataset.clf_images_raw_dir
        dataset_raw = data.GalaxyRawSet(clf_images_path)
        dataset_raw.copy_to(cfg.clf_images_test, test_df.index.to_list())
        dataset_raw.copy_to(cfg.clf_images_train_val, train_val_df.index.to_list())

    # Write label files
    test_df.to_csv(cfg.clf_labels_test)
    train_val_df.to_csv(cfg.clf_labels_train_val)
    print(f"Test labels written to {cfg.clf_labels_test}")
    print(f"Train/val labels written to {cfg.clf_labels_train_val}")


def write_artifacts(
    labels_df: pd.DataFrame,
    parent_artifact: wandb.Artifact,
    cfg: config.DatasetConfig,
    from_raw: bool,
    run: wandb.run,
) -> None:
    train_val_items = [cfg.clf_labels_train_val, cfg.clf_labels_test, cfg.clf_images_train_val]
    train_val_artifact = utils.make_dataset_artifact("clf_train_val", train_val_items)

    if from_raw:
        test_items = [cfg.clf_labels_test, cfg.clf_images_test]
        test_artifact = utils.make_dataset_artifact("clf_test", test_items)

        left_table = parent_artifact.get(cfg.raw_table)
        labels_df["Dataset"] = "clf_train_val"
        labels_df.loc[labels_df["Split"] == "test", "Dataset"] = "clf_test"

        right_table = wandb.Table(dataframe=labels_df.reset_index())
        split_table = wandb.JoinedTable(left_table, right_table, "GalaxyID")
        train_val_artifact.add(split_table, name=cfg.split_table)

        run.log_artifact(test_artifact)

    run.log_artifact(train_val_artifact)
    run.finish()


if __name__ == "__main__":
    main()
