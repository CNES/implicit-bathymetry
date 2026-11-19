#!/usr/bin/env python3
# =========================================================================
#   Program:   iota2
#
#   Copyright (c) CNES. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
#
# =========================================================================
"""Create the implicit bathy scheduling."""

import logging
from collections import OrderedDict
from functools import partial
from pathlib import Path

import pandas as pd

import implicit_bathy.orchestration.common.implicit_bathy_constants as il_const
from implicit_bathy.orchestration.sequence_builders.il_sequence_builder import (
    il_builder,
)
from implicit_bathy.orchestration.steps import (
    analyze_step,
    dataset_gpkg_step,
    dataset_pt_step,
    download_step,
    inference_step,
    train_step,
)
from implicit_bathy.orchestration.steps.implicit_bathy_step import StepContainer

LOGGER = logging.getLogger("distributed.worker")
IL_CONST = il_const.ImplicitBathyConstants()


class ImplicitBathy(il_builder):
    """Launch Implicit Bathymetry steps."""

    def __init__(
        self,
        cfg: str,
        config_resources: str,
        schduler_type: str,
        csv_file: str | None = None,
        restart: bool = False,
        tasks_states_file: str | None = None,
        hpc_working_directory: str = "TMPDIR",
    ):
        """Initialize the chain."""
        super().__init__(
            cfg,
            config_resources,
            schduler_type,
            csv_file,
            restart,
            tasks_states_file,
            hpc_working_directory,
        )

        # steps definitions
        self.config_resources = config_resources
        self.cfg = cfg
        self.csv_file = csv_file
        self.steps_group["download"] = OrderedDict()
        self.steps_group["dataset_gpkg"] = OrderedDict()
        self.steps_group["dataset_pt"] = OrderedDict()
        self.steps_group["train"] = OrderedDict()
        self.steps_group["inference"] = OrderedDict()
        self.steps_group["analyze"] = OrderedDict()
        # build steps
        self.sort_step()
        self.control_var = self.init_dict_control_variables()

    def build_steps(self, cfg, config_ressources=None):
        """Build steps."""
        # Init self.tiles for step communication

        # will contains all implicit_bathy steps
        # s_container_pre_processing = StepContainer("preprocess")
        s_container = StepContainer("featuresbuilder")
        s_container.append(
            partial(
                download_step.DownloadStep,
                self.cfg,
                self.config_resources,
                self.csv_file,
            ),
            "download",
        )
        s_container.append(
            partial(
                dataset_gpkg_step.DatasetGpkgStep,
                self.cfg,
                self.config_resources,
                self.csv_file,
            ),
            "dataset_gpkg",
        )
        s_container.append(
            partial(
                dataset_pt_step.DatasetPtStep,
                self.cfg,
                self.config_resources,
                self.csv_file,
            ),
            "dataset_pt",
        )
        s_container.append(
            partial(
                train_step.TrainStep,
                self.cfg,
                self.config_resources,
                self.csv_file,
            ),
            "train",
        )
        s_container.append(
            partial(
                inference_step.InferenceStep,
                self.cfg,
                self.config_resources,
                self.csv_file,
            ),
            "inference",
        )
        if Path(self.csv_file).exists():
            s_container.append(
                partial(
                    analyze_step.AnalyzeStep,
                    self.cfg,
                    self.config_resources,
                    self.csv_file,
                ),
                "analyze",
            )

        return [s_container]

    def init_dict_control_variables(self):
        """."""
        # control variable
        control_var = {}
        control_var["implicit_bathy_outputs_dir"] = (
            self.il_cfg_params.config.dataprocessing.output_path
        )
        control_var["remove_output_path"] = False
        return control_var

    def generate_output_directories(
        self, first_step_index: int, restart: bool, builder_index: int = 0
    ):
        """."""
        il_output_dir = self.control_var["implicit_bathy_outputs_dir"]

        _ = Path(il_output_dir) / IL_CONST.il_tasks_status_filename

        _ = generate_folders(
            self.il_cfg_params.config.dataprocessing.output_path, self.csv_file
        )


def generate_folders(out_path, csv_file):
    """Create dictionnary containing path for all products."""
    path = {
        "out_path": out_path,
        "data": Path(out_path) / "data",
        "dsm": Path(out_path) / "dsm",
        "inputs_nc": Path(out_path) / "inputs_nc",
        "cfg": Path(out_path) / "cfg",
    }
    if Path(csv_file).exists():
        df = pd.read_csv(csv_file, index_col=0)
        id_list = df.loc["id"].astype(int).tolist()
        path.update(
            {
                "cfg_all": Path(out_path) / "cfg_all",
                "synthesis": Path(out_path) / "synthesis",
            }
        )
        for launch_id in id_list:
            path.update(
                {
                    f"out_path_{launch_id}": Path(out_path) / f"out_path_{launch_id}",
                    f"cfg_{launch_id}": Path(out_path)
                    / f"out_path_{launch_id}"
                    / "cfg",
                    f"data_{launch_id}": Path(out_path)
                    / f"out_path_{launch_id}"
                    / "data",
                    f"train_{launch_id}": Path(out_path)
                    / f"out_path_{launch_id}"
                    / "train",
                    f"inference_{launch_id}": Path(out_path)
                    / f"out_path_{launch_id}"
                    / "inference",
                    f"logs_{launch_id}": Path(out_path)
                    / f"out_path_{launch_id}"
                    / "logs",
                }
            )
    else:
        path.update(
            {
                "logs": Path(out_path) / "logs",
                "train": Path(out_path) / "train",
                "inference": Path(out_path) / "inference",
            }
        )

    for _, folder in path.items():
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)
    return path
