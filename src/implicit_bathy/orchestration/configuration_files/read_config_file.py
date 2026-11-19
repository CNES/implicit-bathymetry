"""
Read configuration file
"""

import importlib
import os
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, FilePath

from implicit_bathy.orchestration.common.file_utils import (
    file_search_and,
    get_implicit_bathy_project_dir,
)


class Dataprocessing(BaseModel):
    """
    Configuration model for Dataprocessing
    """

    output_path: Path
    dsm_path: FilePath | None = None
    dataset_path: Path | None = None
    roi_gpkg: FilePath
    start_date: date
    end_date: date
    id_feature_swot: int | None = None
    classes_to_keep: list
    min_wse: int | None = None
    max_wse: int | None = None
    land_phase_shift: float


class Train(BaseModel):
    """
    Configuration model for Train
    """

    seed: int
    bayesian_model: bool
    monitor_every_epoch: int
    ground_model: str
    x_freq_enc_dim: int | None = None
    weighted_by_phase_noise: bool
    weighted_function: str | None = None
    width1: int
    depth1: int
    gaussian_act_sigma: float
    water_model: str
    width2: int | None = None
    depth2: int | None = None
    softmax_factor: int | None = None
    water_probability: str
    ground_probability: str
    n_epochs: int
    plot_every_epoch: int
    lambda_label: int | None = None
    n_epochs_uncert: int | None = None
    lambda_uncert: int | None = None
    n_epochs_wloss: int
    lambda_wloss: int
    optimizer_name: str
    lr: float
    wl_optimizer_name: str
    wl_lr: float
    confidence_threshold: float
    weighted_by_phase_noise: bool
    weighted_function: str | None = None


class Inference(BaseModel):
    """
    Configuration model for Inference
    """

    plot_density: int
    fig_size: int
    vt_path: FilePath | None = None
    max_wse_for_X_plot: int
    min_wse_for_dsm: int
    max_wse_for_dsm: int
    id_feature_swot: int | None = None
    water_threshold_proba: float
    ground_threshold_proba: float
    surface_inference: str
    volume_inference: str
    p_w_filter: float | None = None


class TaskRetryLimits(BaseModel):
    """
    Configuration model for TaskRetryLimits
    """

    allowed_retry: int = 0
    maximum_ram: float = 180.0
    maximum_cpu: int = 4


class Builders(BaseModel):
    """
    Configuration model for Builders
    """

    builders_class_name: list[str] = ["ImplicitBathy"]


class Slurm(BaseModel):
    """
    Configuration model for Slurm
    """

    account: str


class Configuration(BaseModel):
    """
    Configuration model for all sections
    """

    dataprocessing: Dataprocessing
    train: Train
    inference: Inference
    task_retry_limits: TaskRetryLimits
    slurm: Slurm
    builders: Builders


class ReadConfigFile:
    """Configuration file reader."""

    def __init__(self, cfg_file: str):
        """
        ReadConfigFile constructor.
        Args:
        cfg_file (str) : configuration file
        """
        self.cfg_file = cfg_file
        with open(self.cfg_file, encoding="UTF-8") as conf_file:
            cfg_as_dict = yaml.safe_load(conf_file)
        self.config = Configuration(**cfg_as_dict)

    def get_builders(self) -> list:
        """Get list all builders involved."""

        def check_code_path(code_path):
            if code_path is None:
                return False
            if code_path.lower() == "none":
                return False
            if len(code_path) < 1:
                return False
            if not Path(code_path).is_file():
                raise ValueError(f"Error: {code_path} is not a correct path")
            return True

        def check_import(module_path):

            spec = importlib.util.spec_from_file_location(
                module_path.split(os.sep)[-1].split(".")[0], module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        def check_function_in_module(module, builder_name):
            try:
                builder_init = getattr(module, builder_name)
                return builder_init
            except AttributeError as exc:
                raise AttributeError(
                    f"{module.__name__} has no class {builder_name}"
                ) from exc

        modules_path = []

        il_mod_path = (
            Path(get_implicit_bathy_project_dir())
            / "orchestration"
            / "sequence_builders"
        )
        il_modules = file_search_and(il_mod_path, True, ".py")
        modules_path = list(
            filter(
                lambda x: ".pyc" not in x and "__init__" not in x, il_modules
            )
        )

        class_names = self.config.builders.builders_class_name
        builders = []
        for class_name in class_names:
            class_found = []
            for module_path in modules_path:
                module_path_valid = check_code_path(module_path)
                module = None
                init_builder = None
                if module_path_valid:
                    module = check_import(module_path)
                    try:
                        init_builder = check_function_in_module(
                            module, class_name
                        )
                        class_found.append(True)
                    except AttributeError:
                        class_found.append(False)
                        continue
                builders.append(init_builder)
                break
            if not any(class_found):
                raise ValueError(
                    f"builder class '{class_name}' can't be found in "
                    f"modules '{' '.join(modules_path)}'"
                )
        return builders
