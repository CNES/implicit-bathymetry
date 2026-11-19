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
Automatic download S2
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import planetary_computer
import pystac_client
import rasterio as rio
from rasterio import features, warp, windows
from shapely.geometry import shape

from implicit_bathy.config.read_config_file import ReadConfigFile

if os.getenv("CURL_CA_BUNDLE") is not None:
    os.environ["CURL_CA_BUNDLE"] = os.getenv("CURL_CA_BUNDLE")

logging.basicConfig(level=logging.INFO)


def download_s2_tif(config_file: str) -> None:
    """
    Download S2 image
    Args:
        config_file (str): Configuration file
    """
    cfg = ReadConfigFile(config_file)
    out = cfg.config.dataprocessing.output_path
    path_download_s2 = Path(out) / "data" / "S2.tif"
    if not Path(path_download_s2).exists():
        logging.info("No s2 tif -> downloading")
        gdf = gpd.read_file(cfg.config.dataprocessing.roi_gpkg).geometry.iloc[0]
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        toi = f"{cfg.config.dataprocessing.start_date}/{cfg.config.dataprocessing.end_date}"
        collections = ["sentinel-2-l2a", "sentinel-2-l2b"]
        search_s2 = []

        for collection in collections:
            search = catalog.search(
                collections=[collection],
                intersects=gdf,
                datetime=toi,
            )
            search_s2.extend(search.items())

        if search_s2:
            for it in search_s2:
                if shape(it.geometry).contains(gdf):
                    aoi_bounds = features.bounds(gdf)
                    with rio.open(it.assets["B02"].href) as ds:
                        crs = ds.crs
                        img_transform = ds.transform
                        warped_aoi_bounds = warp.transform_bounds(
                            "epsg:4326", crs, *aoi_bounds
                        )
                        aoi_window = windows.from_bounds(
                            transform=ds.transform, *warped_aoi_bounds
                        )
                        new_transform = rio.windows.transform(
                            aoi_window, img_transform
                        )
                        data = ds.read(window=aoi_window)
                    with rio.open(
                        path_download_s2,
                        "w",
                        driver="GTiff",
                        height=data.shape[1],
                        width=data.shape[2],
                        count=1,
                        dtype=rio.float32,
                        crs=crs,
                        transform=new_transform,
                    ) as dst:
                        dst.write(data)
                    break

        if not search_s2:
            raise ValueError("No S2 detected...")


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
    download_s2_tif(args.config)


if __name__ == "__main__":
    sys.exit(main())
