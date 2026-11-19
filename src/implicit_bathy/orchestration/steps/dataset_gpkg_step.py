"""
GPKG Dataset Creation
"""

from pathlib import Path

import yaml

from implicit_bathy.config.create_configs import CreationConfigs
from implicit_bathy.config.read_config_file import ReadConfigFile
from implicit_bathy.orchestration.steps import implicit_bathy_step
from implicit_bathy.preprocess.dataset_CLI import run


def create_gpkg_and_create_config(cfg, config_gpkg, cfg_resources_file, csv_file):
    """
    Args:
        cfg (str): Configuration file
        config_gpkg (str): Configuration gpkg file
        config_file_template (str): Configuration template file
        csv_file (str): Parameters csv file
    """
    run(config_gpkg, cfg_resources_file)
    if Path(csv_file).exists():
        cfg = ReadConfigFile(cfg)
        out = cfg.config.dataprocessing.output_path
        list_config_all = Path(out).glob("cfg_all/*.yaml")
        data_prqt_path = Path(out) / "data" / "dataset.parquet"
        for config_all in list_config_all:
            _ = CreationConfigs(config_all, data_prqt_path)


class DatasetGpkgStep(implicit_bathy_step.Step):
    """
    Class for launching creation dataset gpkg step
    """

    resources_block_name = "dataset_gpkg"

    def __init__(self, cfg, cfg_ressources_file, csv_file=None, working_directory=None):
        super().__init__(
            cfg,
            cfg_ressources_file,
            csv_file,
            self.resources_block_name,
        )
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config_gpkg = (
            Path(data["dataprocessing"]["output_path"])
            / "cfg"
            / "config_data_gpkg.yaml"
        )

        task = self.il_task(
            task_name="dataset_gpkg",
            log_dir=self.log_step_dir,
            execution_mode=self.execution_mode,
            task_parameters={
                "f": create_gpkg_and_create_config,
                "cfg": cfg,
                "config_gpkg": config_gpkg,
                "cfg_resources_file": cfg_ressources_file,
                "csv_file": csv_file,
            },
            task_resources=self.get_resources(),
        )
        self.add_task_to_il_processing_graph(
            task,
            task_group="dataset_gpkg",
            task_sub_group="dataset_gpkg-dep",
            task_dep_dico={"download": ["download-dep"]},
        )

    @classmethod
    def step_description(cls):
        """Print a short description of the step's purpose."""
        description = "Create GPKG."
        return description
