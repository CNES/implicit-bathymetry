"""
Pytorch Dataset Creation
"""

from pathlib import Path

import pandas as pd
import yaml

from implicit_bathy.orchestration.steps import implicit_bathy_step
from implicit_bathy.preprocess.dataset_CLI import run


class DatasetPtStep(implicit_bathy_step.Step):
    """
    Class for launching creation dataset pt step
    """

    resources_block_name = "dataset_pt"

    def __init__(self, cfg, cfg_ressources_file, csv_file=None, working_directory=None):
        super().__init__(
            cfg,
            cfg_ressources_file,
            csv_file,
            self.resources_block_name,
        )
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if Path(csv_file).exists():
            df = pd.read_csv(csv_file, index_col=0)
            id_list = df.loc["id"].astype(int).tolist()
            for i in id_list:
                config_pt = (
                    Path(data["dataprocessing"]["output_path"])
                    / f"out_path_{i}"
                    / "cfg"
                    / "config_data_pt.yaml"
                )
                task = self.il_task(
                    task_name=f"dataset_pt_{i}",
                    log_dir=self.log_step_dir,
                    execution_mode=self.execution_mode,
                    task_parameters={
                        "f": run,
                        "config_file": config_pt,
                        "config_ressources": cfg_ressources_file,
                    },
                    task_resources=self.get_resources(),
                )
                self.add_task_to_il_processing_graph(
                    task,
                    task_group="dataset_pt",
                    task_sub_group=f"dataset_pt_{i}",
                    task_dep_dico={"dataset_gpkg": ["dataset_gpkg-dep"]},
                )
        else:
            config_pt = (
                Path(data["dataprocessing"]["output_path"])
                / "cfg"
                / "config_data_pt.yaml"
            )
            task = self.il_task(
                task_name="dataset_pt",
                log_dir=self.log_step_dir,
                execution_mode=self.execution_mode,
                task_parameters={
                    "f": run,
                    "config_file": config_pt,
                    "config_ressources": cfg_ressources_file,
                },
                task_resources=self.get_resources(),
            )

            self.add_task_to_il_processing_graph(
                task,
                task_group="dataset_pt",
                task_sub_group="dataset_pt-dep",
                task_dep_dico={"dataset_gpkg": ["dataset_gpkg-dep"]},
            )

    @classmethod
    def step_description(cls):
        """Print a short description of the step's purpose."""
        description = "Create Pytorch."
        return description
