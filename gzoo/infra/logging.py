import logging
import os
from datetime import datetime


class Log:
    """Enables logging.

    Args:
        task (str): recorded task
        level (str): logging level

    Methods:
        toggle: toggles logging
    """

    def __init__(self, step, opt, level=logging.DEBUG):
        self.task = step + "_" + opt.exp.task
        self.exp_name = opt.exp.name
        self.model_name = opt.model.arch
        self.format = "[%(asctime)s][%(levelname)s][%(module)s] - %(message)s"
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.level = level
        if opt.exp.name:
            self.dir = f"logs/{opt.exp.name}/"
            self.fpath = (
                self.dir + "log_" + self.task + datetime.now().strftime("_%Y-%m-%d") + ".txt"
            )
        else:
            self.dir = f'logs/{self.model_name}_{datetime.now().strftime("_%Y-%m-%d")}/'
            self.fpath = self.dir + "log_" + self.task + datetime.now().strftime("_%H%M") + ".txt"

    def toggle(self):
        os.system(f"mkdir -p {self.dir}")
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

        with open(self.fpath, "a") as f:
            first_line = "-" * 20 + "   " + self.task.upper()
            f.write(first_line)
            f.write("\n")
