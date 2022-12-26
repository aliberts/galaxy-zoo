import pyrallis

from gzoo.infra.config import PredictConfig, TrainConfig


def main():
    """
    This will regenerate .yaml config files based on the config classes defined
    in gzoo.infra.config. This script should be run whenever changes in those
    classes have been made.
    """
    train_cfg = TrainConfig()
    predict_cfg = PredictConfig()
    with open("config/train.yaml", "w") as train_f, open("config/predict.yaml", "w") as predict_f:
        pyrallis.dump(train_cfg, train_f)
        pyrallis.dump(predict_cfg, predict_f)


if __name__ == "__main__":
    main()
