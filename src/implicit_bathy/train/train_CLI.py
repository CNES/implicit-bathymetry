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
Train CLI
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from implicit_bathy.config.config import (
    BayesianTrainingConfig,
    StandardTrainingConfig,
)
from implicit_bathy.train.trainer import BayesianTrainer, StandardTrainer

logging.basicConfig(level=logging.INFO)


def load_config(filename):
    """read yaml config file"""
    with open(filename, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def run(config_file: Path, config_ressources: Path):
    """
    Run the different setup scripts
    """

    yaml_file = load_config(config_file)

    # regroup parameters for GROUND model
    if yaml_file["ground_model"].lower() == "mlp":
        ground_model_parameters = (
            yaml_file["width1"],
            yaml_file["depth1"],
            yaml_file["x_freq_enc_dim"],
        )
    elif yaml_file["ground_model"].lower() == "garf":
        ground_model_parameters = (
            yaml_file["width1"],
            yaml_file["depth1"],
            yaml_file["gaussian_act_sigma"],
        )
    else:
        raise ValueError("Ground model is not correctly defined.")

    # regroup parameters for WATER model
    if (
        yaml_file["width2"]
        and yaml_file["depth2"]
        and yaml_file["water_model"].lower() == "mlp"
    ):
        water_model_parameters = (yaml_file["width2"], yaml_file["depth2"])
    elif yaml_file["n_dates"] and yaml_file["water_model"].lower() == "lut":
        water_model_parameters = (yaml_file["n_dates"],)
    else:
        raise ValueError("Water model is not correctly defined.")

    if yaml_file["weighted_by_phase_noise"]:
        weighted_function = yaml_file["weighted_function"]
    else:
        weighted_function = None

    lambda_uncert = yaml_file.get("lambda_uncert")
    if lambda_uncert is None:
        lambda_uncert = None
        n_epochs_uncert = None
    else:
        lambda_uncert = yaml_file["lambda_uncert"]
        n_epochs_uncert = yaml_file["n_epochs_uncert"]

    lambda_label = yaml_file.get("lambda_label")
    if lambda_label is None:
        lambda_label = None
    else:
        lambda_label = yaml_file["lambda_label"]
    save_model_epoch = yaml_file.get("save_model_epoch")
    save_model_epoch = save_model_epoch or []

    # Retrieve all parameters and put them into configClass :
    # instanciate trainer based on the config
    if yaml_file["bayesian_model"]:
        config_params = BayesianTrainingConfig(
            seed=yaml_file["seed"],
            dataset_path=yaml_file["dataset_path"],
            save_path=Path(yaml_file["save_path"]),
            bayesian_model=yaml_file["bayesian_model"],
            ground_model=yaml_file["ground_model"],
            ground_model_params=ground_model_parameters,
            water_model=yaml_file["water_model"],
            water_model_params=water_model_parameters,
            water_model_mean_init=yaml_file["water_model_mean_init"],
            water_model_var_init=yaml_file["water_model_var_init"],
            ground_model_mean_init=yaml_file["ground_model_mean_init"],
            ground_model_var_init=yaml_file["ground_model_var_init"],
            ground_prob=yaml_file["ground_probability"],
            optimizer_name=yaml_file["optimizer_name"],
            lr=yaml_file["lr"],
            wl_optimizer_name=yaml_file["wl_optimizer_name"],
            wl_lr=yaml_file["wl_lr"],
            n_epochs=yaml_file["n_epochs"],
            plot_every_epoch=yaml_file["plot_every_epoch"],
            save_model_epoch=save_model_epoch,
            lambda_label=lambda_label,
            n_epochs_uncert=n_epochs_uncert,
            lambda_uncert=lambda_uncert,
            n_epochs_wloss=yaml_file["n_epochs_wloss"],
            lambda_wloss=yaml_file["lambda_wloss"],
            confidence_threshold=yaml_file["confidence_threshold"],
            weighted_by_phase_noise=yaml_file["weighted_by_phase_noise"],
            weighted_function=weighted_function,
        )
        trainer = BayesianTrainer(config_params)
    else:
        config_params = StandardTrainingConfig(
            seed=yaml_file["seed"],
            dataset_path=yaml_file["dataset_path"],
            save_path=Path(yaml_file["save_path"]),
            bayesian_model=yaml_file["bayesian_model"],
            ground_model=yaml_file["ground_model"],
            ground_model_params=ground_model_parameters,
            water_model=yaml_file["water_model"],
            water_model_params=water_model_parameters,
            water_model_mean_init=yaml_file["water_model_mean_init"],
            ground_model_mean_init=yaml_file["ground_model_mean_init"],
            softmax_factor=yaml_file["softmax_factor"],
            lr=yaml_file["lr"],
            optimizer_name=yaml_file["optimizer_name"],
            n_epochs=yaml_file["n_epochs"],
            plot_every_epoch=yaml_file["plot_every_epoch"],
            save_model=yaml_file["save_model"],
            weighted_by_phase_noise=yaml_file["weighted_by_phase_noise"],
            weighted_function=weighted_function,
        )

        trainer = StandardTrainer(config_params)

    # Do the training
    logging.info(
        "starting : training will be monitored every %s epochs",
        config_params.plot_every_epoch,
    )
    trainer.train(config_params)
    logging.info("training finished")


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
