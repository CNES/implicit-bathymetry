"""
Inference step
"""

from pathlib import Path

import pandas as pd
import yaml

from implicit_bathy.inference.inference_CLI import run
from implicit_bathy.orchestration.steps import implicit_bathy_step


class InferenceStep(implicit_bathy_step.Step):
    """
    Class for launching inference step
    """

    resources_block_name = "inference"

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
                config_inference = (
                    Path(data["dataprocessing"]["output_path"])
                    / f"out_path_{i}"
                    / "cfg"
                    / "config_inference.yaml"
                )
                task = self.il_task(
                    task_name=f"inference_{i}",
                    log_dir=self.log_step_dir,
                    execution_mode=self.execution_mode,
                    task_parameters={
                        "f": run,
                        "config_file": config_inference,
                        "config_ressources": cfg_ressources_file,
                    },
                    task_resources=self.get_resources(),
                )
                self.add_task_to_il_processing_graph(
                    task,
                    task_group="inference",
                    task_sub_group=f"inference_{i}",
                    task_dep_dico={"train": [f"train_{i}"]},
                )
        else:
            config_inference = (
                Path(data["dataprocessing"]["output_path"])
                / "cfg"
                / "config_inference.yaml"
            )
            task = self.il_task(
                task_name="inference",
                log_dir=self.log_step_dir,
                execution_mode=self.execution_mode,
                task_parameters={
                    "f": run,
                    "config_file": config_inference,
                    "config_ressources": cfg_ressources_file,
                },
                task_resources=self.get_resources(),
            )
            self.add_task_to_il_processing_graph(
                task,
                task_group="inference",
                task_sub_group="inference-i",
                task_dep_dico={"train": ["train-dep_i"]},
            )

    @classmethod
    def step_description(cls):
        """Print a short description of the step's purpose."""
        description = "Prediction."
        return description
