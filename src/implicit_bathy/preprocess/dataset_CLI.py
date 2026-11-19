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
Dataset SWOT CLI
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset


def load_config(filename):
    """
    Args:
        filename (str): Configuration file
    """
    with open(filename, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def run(config_file: Path, config_ressources: Path):
    """
    Args:
        config_file (Path): Configuration file
        config_ressources (Path) : Configuration ressources file
    """
    logging.info(config_file)
    yaml_file = load_config(config_file)
    logging.info(yaml_file)
    my_path = Path(yaml_file["dataset_path"])
    if my_path.is_dir():
        # the user gave the directory where the .nc swot files are stored,
        # we create the gpkg
        dataset_params = DatasetConfig(
            dataset_path=yaml_file["dataset_path"],
            saving_path=yaml_file["saving_path"],
            roi_gpkg=yaml_file["roi_gpkg"],
            remove_nc_file=yaml_file["remove_nc_file"]
        )
    elif my_path.is_file():
        if my_path.suffix == ".pt":
            # the user gave the file of the dataset, we just open it :
            dataset_params = DatasetConfig(
                dataset_path=yaml_file["dataset_path"]
            )
        elif my_path.suffix == ".gpkg" or my_path.suffix == ".parquet" :
            # create pt file
            classes_to_keep = yaml_file["classes_to_keep"]
            if not classes_to_keep:  # empty object in config yaml
                raise ValueError(
                    "You must indicates at least one class to keep"
                )
            if not classes_to_keep[0]:  # empty list in config yaml
                raise ValueError(
                    "You must indicates at least one class to keep"
                )

            dataset_params = DatasetConfig(
                dataset_path=yaml_file["dataset_path"],
                convert_to_crs=yaml_file["convert_to_crs"],
                saving_path=yaml_file["saving_path"],
                roi_gpkg=yaml_file["roi_gpkg"],
                id_feature_swot=yaml_file["id_feature_swot"],
                land_phase_shift=yaml_file["land_phase_shift"],
                min_wse=yaml_file["min_wse"],
                max_wse=yaml_file["max_wse"],
                classes_to_keep=yaml_file["classes_to_keep"],
            )

    multi_dataset = MultiDateLakeDataset(dataset_params)
    multi_dataset.load_dataset()


def arguments_cli() -> argparse.ArgumentParser:
    """Create parser for command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "-r",
        "--config_ressources",
        type=Path,
        required=False,
        help="Path to configuration ressources file",
    )
    return parser


def main():
    """Main function"""
    args = arguments_cli().parse_args()
    run(args.config, args.config_ressources)


if __name__ == "__main__":
    sys.exit(main())
