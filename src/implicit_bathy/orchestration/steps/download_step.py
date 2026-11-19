"""
Download data step
"""

from pathlib import Path

from implicit_bathy.config.create_configs_all import CreationConfigsAll
from implicit_bathy.download.data_setup_CLI import run
from implicit_bathy.orchestration.steps import implicit_bathy_step


def download_and_create_config(config_file, config_ressources, csv_file):
    """
    Args:
        config_file (str): Configuration file
        config_ressources (str): Configuration ressources file
        csv_file (str): Parameters csv file
    """
    run(config_file, config_ressources)
    if Path(csv_file).exists():
        create = CreationConfigsAll(config_file, csv_file)
        create.create_all_config_global()


class DownloadStep(implicit_bathy_step.Step):
    """
    Class for launching download step
    """

    resources_block_name = "download"

    def __init__(self, cfg, cfg_ressources_file, csv_file=None, working_directory=None):
        super().__init__(
            cfg,
            cfg_ressources_file,
            csv_file,
            self.resources_block_name,
        )
        task = self.il_task(
            task_name="download",
            log_dir=self.log_step_dir,
            execution_mode=self.execution_mode,
            task_parameters={
                "f": download_and_create_config,
                "config_file": cfg,
                "config_ressources": cfg_ressources_file,
                "csv_file": csv_file,
            },
            task_resources=self.get_resources(),
        )
        self.add_task_to_il_processing_graph(
            task,
            task_group="download",
            task_sub_group="download-dep",
            task_dep_dico={"first_task": []},
        )

    @classmethod
    def step_description(cls):
        """Print a short description of the step's purpose."""
        description = "Download SWOT PIXC and COP30."
        return description
