import pyrallis

from gzoo.infra import config


def main() -> None:
    """
    This will regenerate .yaml config files based on the config classes defined
    in gzoo.infra.config. This script should be run whenever changes in those
    classes have been made.
    """
    train_cfg = config.TrainConfig()
    predict_cfg = config.PredictConfig()
    with open("config/train.yaml", "w") as train_f, open("config/predict.yaml", "w") as predict_f:
        pyrallis.dump(train_cfg, train_f)
        pyrallis.dump(predict_cfg, predict_f)

    print("Config files updated.")


if __name__ == "__main__":
    main()
