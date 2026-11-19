#!/usr/bin/env python3
# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the implicit_bathy project:
#  https://.../implicit_bathy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Read config file
"""

import types
from datetime import date
from pathlib import Path
from typing import Union, get_args, get_origin

import yaml
from pydantic import BaseModel, FilePath


class Dataprocessing(BaseModel):
    """
    Configuration model for Dataprocessing
    """

    output_path: Path
    dsm_path: FilePath | None = None
    dataset_path: Path | None = None
    remove_nc_file: bool
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
    n_epochs_wloss: int
    lambda_wloss: int
    n_epochs_uncert: int | None
    lambda_uncert: int | None
    optimizer_name: str
    lr: float
    wl_optimizer_name: str
    wl_lr: float
    confidence_threshold: float
    save_model_epoch: list | None
    ground_model_mean_init: float | None
    ground_model_var_init: float | None
    water_model_mean_init: float | None
    water_model_var_init: float | None


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
    s2_tif_path: str | None = None


class Configuration(BaseModel):
    """
    Configuration model for Configuration
    """

    dataprocessing: Dataprocessing
    train: Train
    inference: Inference


def get_types_dict(model: type[BaseModel]) -> dict:
    """
    Get types
    """

    def clean_type(annotation):
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            args = [a for a in get_args(annotation) if a is not type(None)]
            if args:
                return clean_type(args[0])
            return None
        return annotation

    return {
        name: clean_type(field.annotation)
        for name, field in model.model_fields.items()
    }


class ReadConfigFile:
    """Configuration file reader."""

    def __init__(self, cfg_file: str):
        """ReadConfigFile constructor.

        Parameters
        ----------
        cfg_file
            configuration file
        """
        self.cfg_file = cfg_file
        with open(self.cfg_file, encoding="UTF-8") as conf_file:
            cfg_as_dict = yaml.safe_load(conf_file)
        self.config = Configuration(**cfg_as_dict)
