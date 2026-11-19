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
All class configuration
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    Dataset config class
    """

    dataset_path: str | None = None
    saving_path: str | None = None
    id_feature_swot: str | None = None
    land_phase_shift: float | None = None
    # to create .gpkg from many .nc files :
    roi_gpkg: str | None = None
    remove_nc_file: bool | None = None

    # to create .pt dataset from a .gpkg :
    min_wse: int | None = None
    max_wse: int | None = None
    convert_to_crs: str | None = None
    classes_to_keep: list = field(default_factory=list)

    # "old method", with many shapefiles
    area_path: str | None = None
    contour_path: str | None = None


@dataclass
class TrainingConfig:
    """
    Training config class
    """

    seed: int | None = None
    monitor_every_epoch: int | None = None
    # data parameters
    dataset_path: str | None = None
    area_path: str | None = None
    contour_path: str | None = None
    save_path: Path | None = None
    min_wse: int | None = None
    max_wse: int | None = None
    # model type
    bayesian_model: bool | None = None
    # ground model
    ground_model: str | None = None
    # ground_model_params:
    # if MLP : 1) width, 2) depth, 3) x_freq_enc_dim.
    # if GARF : 1) width, 2) depth, 3) gaussian_act_sigma
    ground_model_params: tuple | None = None
    #Value to initialize the ground model in normalized coordinates (0.5 default)
    ground_model_mean_init: float | None = None
    # water model
    water_model: str | None = None
    # water_model_params:
    # if 2 arguments : water_model_params, if only 1 = nb_dates
    water_model_params: tuple | None = None
    #Value to initialize the water model in normalized coordinates (0.5 default)
    water_model_mean_init: float | None = None
    # training
    lr: float | None = None
    optimizer_name: str | None = None
    n_epochs: int | None = None
    plot_every_epoch: int | None = None
    save_model_epoch: list = field(default_factory=list)
    weighted_by_phase_noise: bool | None = None
    weighted_function: bool | None = None


@dataclass
class StandardTrainingConfig(TrainingConfig):
    """
    Standard Training config class
    """

    # 2 branch model
    softmax_factor: int | None = None


@dataclass
class BayesianTrainingConfig(TrainingConfig):
    """
    Bayesian Training config class
    """

    # 2 branch model
    ground_prob: str | None = None
    # training
    confidence_threshold: float | None = None
    wl_optimizer_name: str | None = None
    wl_lr: float | None = None
    # parameters for training bayesian model
    lambda_label: int | None = None
    n_epochs_uncert: int | None = None
    lambda_uncert: int | None = None  # it's a ponderation
    n_epochs_wloss: int | None = None
    lambda_wloss: int | None = None  # it's a ponderation
    #Value to initialize the ground model variance in normalized coordinates (0.001 default)
    ground_model_var_init: float | None = None
    #Value to initialize the water model variance in normalized coordinates (0.001 default)
    water_model_var_init: float | None = None


@dataclass
class InferenceConfig:
    """
    Inference config class
    """

    dsm_path: str | None = None
    vt_path: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    swot_wse: list = field(default_factory=list)
    dataset_dir: str | None = None
    contour_path: str | None = None
    convert_to_crs: str | None = None
    dataset_path: str | None = None
    plot_density: int | None = None
    fig_size: int | None = None
    id_feature_swot: str | None = None
    save_path_for_plots: Path | None = None
    max_wse_for_X_plot: int | None = None
    min_wse_for_dsm: int | None = None
    max_wse_for_dsm: int | None = None
    model_path: str | None = None
    water_threshold_proba: float | None = None
    ground_threshold_proba: float | None = None
    plot_p_w_on_s1_images: float | None = None
    save_s2_tif: bool | None = None
    save_s1_tif: bool | None = None
    deg_polyfit: int | None = None
    surface_inference: str | None = None
    volume_inference: str | None = None
    p_w_filter: float | None = None
    s2_tif_path: str | None = None
