"""
Analyze step
"""

from pathlib import Path

import pandas as pd

from implicit_bathy.analysis.analyze_CLI import run
from implicit_bathy.orchestration.steps import implicit_bathy_step


class AnalyzeStep(implicit_bathy_step.Step):
    """
    Class for launching analyze step
    """

    resources_block_name = "analyze"

    def __init__(self, cfg, cfg_ressources_file, csv_file=None, working_directory=None):
        super().__init__(
            cfg,
            cfg_ressources_file,
            csv_file,
            self.resources_block_name,
        )
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file, index_col=0)
            id_list = df.loc["id"].astype(int).tolist()
            inference_i = [f"inference_{x}" for x in id_list]
            task = self.il_task(
                task_name="analyze",
                log_dir=self.log_step_dir,
                execution_mode=self.execution_mode,
                task_parameters={
                    "f": run,
                    "config_file": cfg,
                    "csv_file": csv_file,
                },
                task_resources=self.get_resources(),
            )
            self.add_task_to_il_processing_graph(
                task,
                task_group="analyze",
                task_sub_group="analyze",
                task_dep_dico={"inference": inference_i},
            )
        else:
            task = self.il_task(
                task_name="analyze",
                log_dir=self.log_step_dir,
                execution_mode=self.execution_mode,
                task_parameters={
                    "f": run,
                    "config_file": cfg,
                    "csv_file": csv_file,
                },
                task_resources=self.get_resources(),
            )
            self.add_task_to_il_processing_graph(
                task,
                task_group="analyze",
                task_sub_group="analyze",
                task_dep_dico={"inference": ["inference-i"]},
            )

    @classmethod
    def step_description(cls):
        """Print a short description of the step's purpose."""
        description = "Analyse."
        return description
