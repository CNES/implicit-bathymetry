"""
Class Train step
"""

from pathlib import Path

import pandas as pd
import yaml

from implicit_bathy.orchestration.steps import implicit_bathy_step
from implicit_bathy.train.train_CLI import run


class TrainStep(implicit_bathy_step.Step):
    """
    Class for launching train step
    """

    resources_block_name = "train"

    def __init__(self, cfg, cfg_ressources_file, csv_file=None, working_directory=None):
        super().__init__(
            cfg,
            cfg_ressources_file,
            csv_file,
            self.resources_block_name,
        )
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config_train = (
            Path(data["dataprocessing"]["output_path"]) / "cfg" / "config_train.yaml"
        )

        if Path(csv_file).exists():
            df = pd.read_csv(csv_file, index_col=0)
            id_list = df.loc["id"].astype(int).tolist()
            for i in id_list:
                config_train = (
                    Path(data["dataprocessing"]["output_path"])
                    / f"out_path_{i}"
                    / "cfg"
                    / "config_train.yaml"
                )
                task = self.il_task(
                    task_name=f"train_{i}",
                    log_dir=self.log_step_dir,
                    execution_mode=self.execution_mode,
                    task_parameters={
                        "f": run,
                        "config_file": config_train,
                        "config_ressources": cfg_ressources_file,
                    },
                    task_resources=self.get_resources(),
                )

                self.add_task_to_il_processing_graph(
                    task,
                    task_group="train",
                    task_sub_group=f"train_{i}",
                    task_dep_dico={"dataset_pt": [f"dataset_pt_{i}"]},
                )
        else:
            task = self.il_task(
                task_name="train",
                log_dir=self.log_step_dir,
                execution_mode=self.execution_mode,
                task_parameters={
                    "f": run,
                    "config_file": config_train,
                    "config_ressources": cfg_ressources_file,
                },
                task_resources=self.get_resources(),
            )
            self.add_task_to_il_processing_graph(
                task,
                task_group="train",
                task_sub_group="train-dep_i",
                task_dep_dico={"dataset_pt": ["dataset_pt-dep"]},
            )

    @classmethod
    def step_description(cls):
        """Print a short description of the step's purpose."""
        description = "Training."
        return description
