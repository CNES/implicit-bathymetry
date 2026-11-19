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
Automatic download swot
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import earthaccess
import geopandas as gpd
import requests
import xarray as xr

from implicit_bathy.config.read_config_file import ReadConfigFile

logging.basicConfig(level=logging.INFO)


def download_granule_https(granule, output_dir):
    """
    Download .nc file earthaccess granule via HTTPS.
    Args:
        granule (str):
        output_dir (str) : Output path
    """

    https_links = granule.data_links(access="https")
    url = https_links[0]
    local_file = Path(output_dir) / os.path.basename(url)
    if not local_file.exists():
        logging.info("Download %s", url)
        download_https(url, str(local_file))


def download_https(url, out_path, session=None):
    """
    Download a file via HTTPS.
    Args:
        url (str):
        out_path (str) : Output path
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy="environment")
    req = session if session is not None else requests
    with req.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def remove_corrupted_nc(dataset_path):
    """
    Remove corrupted nc file
    """
    for p in dataset_path:
        try:
            ds = xr.open_dataset(p)
            ds.close()
        except OSError as e:
            if "NetCDF: HDF error" in str(e):
                print(f"Removing corrupted file: {p}")
                Path(p).unlink(missing_ok=True)
                continue
            raise


def download_swot(config_file: str) -> None:
    """
    Download nc from hydrowebnext with EODAG
    Args:
        config_file (str): Configuration file
    """
    cfg = ReadConfigFile(config_file)
    dataset_path = cfg.config.dataprocessing.dataset_path
    out = cfg.config.dataprocessing.output_path
    if not dataset_path:
        logging.info("No path specified, downloading")
        path_download_nc = os.path.join(out, "inputs_nc")
        gdf = gpd.read_file(cfg.config.dataprocessing.roi_gpkg)
        minx, miny, maxx, maxy = gdf.total_bounds
        granules = earthaccess.search_data(
            short_name="SWOT_L2_HR_PIXC_D",
            temporal=(
                cfg.config.dataprocessing.start_date,
                cfg.config.dataprocessing.end_date,
            ),
            bounding_box=(minx, miny, maxx, maxy),
        )
        for granule in granules:
            download_granule_https(granule, path_download_nc)
        nc_folder_content = Path(path_download_nc).rglob("**/*.nc")
        remove_corrupted_nc(nc_folder_content)

    if dataset_path:
        nc_folder_content = dataset_path.rglob("**/*.nc")
        logging.info("%s .nc files in dataset_path", len(list(nc_folder_content)))
        remove_corrupted_nc(nc_folder_content)
        if not nc_folder_content:
            logging.info("No .nc files -> Stop the workflow ")
            sys.exit(1)


def arguments_cli() -> argparse.ArgumentParser:
    """Create parser for command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to command file"
    )
    return parser


def main():
    """Main function"""
    args = arguments_cli().parse_args()
    download_swot(args.config)


if __name__ == "__main__":
    sys.exit(main())
