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
CLI inference
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from implicit_bathy.config.config import (
    BayesianTrainingConfig,
    DatasetConfig,
    InferenceConfig,
    StandardTrainingConfig,
)
from implicit_bathy.inference import (
    PlotterCrossSection,
    PlotterDSM,
    PlotterErrorHistograms,
    PlotterVS,
    PlotterWaterProbaOnS2,
)
from implicit_bathy.inference.inference_classes import Processor, SimulateInputs
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logging.info("The device detected on this machine is %s", device)


def load_config(filename):
    """
    Load config
    """
    with open(filename, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def run(config_file: Path, config_ressources: Path):
    """
    Run the different setup scripts
    """
    # retrieve infos from YAML for plot config -------------------------------
    yaml_file = load_config(config_file)

    plot_cross_section = yaml_file["plot_cross_section"]
    plot_dsm = yaml_file["plot_DSM"]
    plot_error_histograms = yaml_file["plot_error_histograms"]
    plot_p_w_on_s2_images = yaml_file["plot_p_w_on_s2_images"]
    plot_p_w_on_s1_images = yaml_file["plot_p_w_on_s1_images"]
    save_s2_tif = yaml_file["save_s2_tif"]
    save_s1_tif = yaml_file["save_s1_tif"]
    plot_volume_surface = yaml_file["plot_volume_surface"]

    contour_path = yaml_file["contour_path"]
    convert_to_crs = yaml_file["convert_to_crs"]
    model_path = yaml_file["model_path"]
    dsm_path = yaml_file["dsm_path"]
    vt_path = yaml_file["vt_path"]
    start_time = yaml_file["start_time"]
    end_time = yaml_file["end_time"]
    id_feature_swot = yaml_file["id_feature_swot"]
    dataset_path = yaml_file["dataset_path"]
    plot_density = yaml_file["plot_density"]
    deg_polyfit = yaml_file["deg_polyfit"]
    fig_size = yaml_file["fig_size"]
    save_path_for_plots = Path(yaml_file["save_path_for_plots"])
    max_wse_for_X_plot = int(yaml_file["max_wse_for_X_plot"])
    min_wse_for_dsm = int(yaml_file["min_wse_for_dsm"])
    max_wse_for_dsm = int(yaml_file["max_wse_for_dsm"])
    water_threshold_proba = float(yaml_file["water_threshold_proba"])
    ground_threshold_proba = float(yaml_file["ground_threshold_proba"])
    surface_inference = yaml_file["surface_inference"]
    volume_inference = yaml_file["volume_inference"]
    p_w_filter = yaml_file["p_w_filter"]
    s2_tif_path = yaml_file["s2_tif_path"]

    plot_params = InferenceConfig(
        contour_path=contour_path,
        convert_to_crs=convert_to_crs,
        dataset_path=dataset_path,
        model_path=model_path,
        dsm_path=dsm_path,
        vt_path=vt_path,
        start_time=start_time,
        end_time=end_time,
        id_feature_swot=id_feature_swot,
        plot_density=plot_density,
        fig_size=fig_size,
        save_path_for_plots=save_path_for_plots,
        max_wse_for_X_plot=max_wse_for_X_plot,
        min_wse_for_dsm=min_wse_for_dsm,
        max_wse_for_dsm=max_wse_for_dsm,
        water_threshold_proba=water_threshold_proba,
        ground_threshold_proba=ground_threshold_proba,
        plot_p_w_on_s1_images=plot_p_w_on_s1_images,
        save_s2_tif=save_s2_tif,
        save_s1_tif=save_s1_tif,
        deg_polyfit=deg_polyfit,
        surface_inference=surface_inference,
        volume_inference=volume_inference,
        p_w_filter=p_w_filter,
        s2_tif_path=s2_tif_path,
    )

    # retrieve infos from model_path file for training config -------------------------------
    bayesian_model = torch.load(
        model_path, map_location=device, weights_only=False
    )["bayesian_model"]
    ground_model = torch.load(
        model_path, map_location=device, weights_only=False
    )["ground_model"]
    ground_model_params = torch.load(
        model_path, map_location=device, weights_only=False
    )["ground_model_params"]
    water_model = torch.load(
        model_path, map_location=device, weights_only=False
    )["water_model"]
    water_model_params = torch.load(
        model_path, map_location=device, weights_only=False
    )["water_model_params"]
    ground_model_mean_init = torch.load(
        model_path, map_location=device, weights_only=False
    )["ground_model_mean_init"]
    ground_model_var_init = torch.load(
        model_path, map_location=device, weights_only=False
    )["ground_model_var_init"]
    water_model_mean_init = torch.load(
        model_path, map_location=device, weights_only=False
    )["water_model_mean_init"]
    water_model_var_init = torch.load(
        model_path, map_location=device, weights_only=False
    )["water_model_var_init"]

    if bayesian_model:
        ground_prob = torch.load(
            model_path, map_location=device, weights_only=False
        )["ground_prob"]
        training_params = BayesianTrainingConfig(
            bayesian_model=bayesian_model,
            ground_model=ground_model,
            ground_model_params=ground_model_params,
            ground_prob=ground_prob,
            water_model=water_model,
            water_model_params=water_model_params,
            ground_model_mean_init=ground_model_mean_init,
            ground_model_var_init=ground_model_var_init,
            water_model_mean_init=water_model_mean_init,
            water_model_var_init=water_model_var_init

        )
    else:
        training_params = StandardTrainingConfig(
            bayesian_model=bayesian_model,
            ground_model=ground_model,
            ground_model_params=ground_model_params,
            water_model=water_model,
            water_model_params=water_model_params,
        )
    # --------------------------------------------------------------------------------------
    # load dataset for ground truth points
    dataset_params = DatasetConfig(dataset_path=dataset_path)
    dataset = MultiDateLakeDataset(dataset_params)
    dataset.load_dataset()
    # Extract timestamp values from dataset (removed from config)
    t_values = np.unique(dataset.X[:, 2])
    # --------------------------------------------------------------------------------------
    simulation = SimulateInputs()
    processor = Processor(plot_params, training_params, dataset, simulation)

    if plot_dsm:
        logging.info("Plotting predicted DSM")
        plotter_dsm = PlotterDSM(processor)
        plotter_dsm.plot_dsm_water_z()

    if plot_cross_section:
        plotter_cross_section = PlotterCrossSection(processor)
        logging.info("Plotting cross sections along x axis")
        plotter_cross_section.plot_many_cross_sections(t_values, "x")
        logging.info("Plotting cross sections along y axis")
        plotter_cross_section.plot_many_cross_sections(t_values, "y")

    if plot_error_histograms:
        logging.info("Plotting error histograms")
        plotter_error_histrogram = PlotterErrorHistograms(processor)
        plotter_error_histrogram.plot_all_error_histograms(t_values)

    if plot_p_w_on_s2_images:
        logging.info("Plotting water probability on s2 images")
        plotter_water_proba_on_s2 = PlotterWaterProbaOnS2(processor)
        plotter_water_proba_on_s2.plot_water_proba_on_s2_images()

    if plot_volume_surface:
        logging.info("Plotting Volume/Surface")
        plotter_vs = PlotterVS(processor)
        plotter_vs.get_data_vt()
        plotter_vs.get_pred_data_t()
        plotter_vs.get_swot_wse(t_values)
        plotter_vs.plot_relative_wse_t_hydrocron()
        plotter_vs.plot_relative_wse_t()
        plotter_vs.plot_absolute_wse_t(t_values)
        plotter_vs.plot_absolute_wse_t_hydrocron()
        plotter_vs.plot_surface_t_hydrocron()
        plotter_vs.plot_volume_t()
        plotter_vs.plot_volume_area()
        plotter_vs.export_csv()
        plotter_vs.compute_indicators()


def arguments_cli() -> argparse.ArgumentParser:
    """Create parser for command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="Path to command file"
    )
    parser.add_argument(
        "-r",
        "--config_ressources",
        type=Path,
        required=False,
        help="Path to command file",
    )
    return parser


def main():
    """Main function"""
    args = arguments_cli().parse_args()
    run(args.config, args.config_ressources)


if __name__ == "__main__":
    sys.exit(main())
