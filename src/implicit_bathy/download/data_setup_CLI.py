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
CLI downloading data
"""

import argparse
import sys
from pathlib import Path

from implicit_bathy.config.create_configs import CreationConfigs
from implicit_bathy.download.automatic_download_dsm import download_dsm
from implicit_bathy.download.automatic_download_s2 import download_s2_tif
from implicit_bathy.download.automatic_download_swot import download_swot


def run(config_file: Path, config_ressources: Path):
    """
    Run the different setup scripts
    Args:
        config_file (str): Configuration file
        config_ressources (str): Configuration ressources file
    """
    download_dsm(config_file)
    download_swot(config_file)
    download_s2_tif(config_file)
    CreationConfigs(config_file)


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
        help="Path to ressource file",
    )
    return parser


def main():
    """Main function"""
    args = arguments_cli().parse_args()
    run(args.config, args.config_ressources)


if __name__ == "__main__":
    sys.exit(main())
