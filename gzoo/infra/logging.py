import logging
from datetime import datetime
from pathlib import Path

from gzoo.infra import config


class Log:
    """Enables logging.

    Args:
        task (str): recorded task.
        exp (config.ExpConfig): experiment config.
        model_name (str): model architecture.
        level (_type_, optional): logging level. Defaults to logging.DEBUG.

    Methods:
        toggle: toggles logging
    """

    def __init__(
        self, step: str, exp: config.ExpConfig, model_name: str, level: int = logging.DEBUG
    ):
        self.task = step + "_" + exp.task
        self.exp_name = exp.name
        self.model_name = model_name
        self.format = "[%(asctime)s][%(levelname)s][%(module)s] - %(message)s"
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.level = level

        if exp.name:
            self.dir = Path(f"logs/{exp.name}/")
            self.fpath = self.dir / f"log_{self.task}_{datetime.now().strftime('%Y-%m-%d')}.txt"
        else:
            self.dir = Path(f"logs/{self.model_name}_{datetime.now().strftime('%Y-%m-%d')}")
            self.fpath = self.dir / f"log_{self.task}_{datetime.now().strftime('_%H%M')}.txt"

    def toggle(self) -> None:
        self.dir.mkdir(exist_ok=True)
        logging.basicConfig(
            format=self.format,
            datefmt=self.datefmt,
            level=self.level,
            filename=self.fpath,
        )

        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(module)s]: %(message)s")
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        with self.fpath.open("a") as f:
            first_line = "-" * 20 + "   " + self.task.upper()
            f.write(first_line)
            f.write("\n")
