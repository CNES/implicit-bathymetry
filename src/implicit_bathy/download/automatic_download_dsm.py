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
Automatic download DSM
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
from bmi_topography import Topography

from implicit_bathy.config.read_config_file import ReadConfigFile

logging.basicConfig(level=logging.INFO)


def download_dsm(config_file: str) -> None:
    """
    Download DSM with bmi_topography
    Args:
        config_file (str): Configuration file
    """
    cfg = ReadConfigFile(config_file)
    out = cfg.config.dataprocessing.output_path
    if not cfg.config.dataprocessing.dsm_path:
        path_download_dsm = Path(out) / "dsm"
        if not (Path(path_download_dsm) / "COP30.tif").is_file():
            logging.info("No dsm found in folder, downloading COP30 product")
            gdf = gpd.read_file(cfg.config.dataprocessing.roi_gpkg)
            min_x, min_y, max_x, max_y = gdf.total_bounds
            params = {
                "dem_type": "COP30",
                "south": min_y,
                "north": max_y,
                "west": min_x,
                "east": max_x,
                "output_format": "GTiff",
                "cache_dir": path_download_dsm,
            }
            dem = Topography(**params)
            dem.fetch()
            dem_file = list(Path(path_download_dsm).glob("COP30*"))[0]
            dem_file.rename(Path(path_download_dsm) / "COP30.tif")
        else:
            logging.info("DSM COP30 already downloaded")


def arguments_cli() -> argparse.ArgumentParser:
    """Create parser for command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="Path to command file"
    )
    return parser


def main():
    """Main function"""
    args = arguments_cli().parse_args()
    download_dsm(args.config)


if __name__ == "__main__":
    sys.exit(main())
