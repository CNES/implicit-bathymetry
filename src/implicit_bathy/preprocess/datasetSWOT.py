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
MultiDateLakeDataset class
"""

import logging
from datetime import date, datetime
from io import StringIO
from pathlib import Path

from pyogrio import list_layers
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from pixcdust.converters.gpkg import Nc2GpkgConverter
from shapely.geometry import Point
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class MultiDateLakeDataset(Dataset):
    """
    The purpose of this class is to generate a PyTorch dataset from provided .nc SWOT files.
    The .nc SWOT data can be dowloaded from EarthAccess or HydroWeb_next (one per time date).
    Then, the .nc files are regrouped into one .gpkg file, and cropped around an ROI.
    Finally, the .gpkg is filtered/normalise and transformed in a Pytorch dataset .pt.

    The data is stored as follow: three inputs representing UTM30N coordinates (labelled x and y,
    contained in tensor X) and a time coordinate corresponding to the number of days
    between the acquisition and a fixed reference date (also in tensor X), as well as
    the corresponding "WSE" as the target (stored as wse in tensor Y) and the SWOT classification.

    All the inputs and targets are normalized between 0 and 1, as well as filtered for extreme values
    of WSE.

    Rq : water surface elevation (WSE) is the name chosenhere to represent a mathematical z for all
    points in the geodata, water or not. It is defined here as the difference between
    the distance from a point to the reference ellipsoid ("height" in the provided
    geodata) and the distance from the reference geoid to the reference ellipsoid
    ("geoid" in the provided geodata).
    """

    def __init__(self, config_params):
        """
        Based on the path given in input, the class will
        - if the path is a folder : constructs a .gpkg gathering all the .nc files in the folder.
        - if the path points to a .gpkg file : convert the gpk in PyTorch dataset
        - if the path points to a .pt file : load the pytorch dataset

        For the first 2 cases, the user needs to provide
        - the path to a countour file (either gpkg or shp) for the (x,y) normalisation between 0 and 1
        - the min and maw value of WSE (for filtering and WSE normalisation between 0 and 1)
        - a booleen to indicate if only points classified as Water by Swot will be kept
        """
        self.config_params = config_params
        # path of either 1) a folder containing .nc files, 2) a .gpkg file, 3) a .pt file
        self.dataset_path = config_params.dataset_path
        # first time date for normalisation
        self.reference_date = date.fromisoformat("2023-01-01")
        self.id_feature_swot = config_params.id_feature_swot
        self.land_phase_shift = config_params.land_phase_shift
        self.end_time = datetime.now().strftime("%Y-%m-%d")
        # vector that will contain the (normalized) dates of the acquisitions
        self.list_dates = []
        # vector that will contain the number of points kept in the dataset for each time date
        self.t_sizes = []
        # vector that will contain the number of water points at each time date
        self.nb_points = {
            "1": [],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "total": [],
        }
        # min and max for WSE (this info will be saved along with the dataset)
        self.max_wse = 0
        self.min_wse = 0
        # initalise empty dataset
        self.X = torch.zeros((0, 3))
        self.Y = torch.empty(0)
        self.L = torch.empty(
            0, dtype=torch.float32
        )  # Label = 1 for water and 0 for all other classes
        self.classification = torch.empty(0)  # classification = SWOT label
        self.phase_noise_level = torch.empty(0)
        self.dheight_dphase = torch.empty(0)
        self.roi_gpkg = config_params.roi_gpkg
        self.saving_path = config_params.saving_path

        my_path = Path(config_params.dataset_path)
        if my_path.is_file() and my_path.suffix == ".parquet":
            self.min_wse = config_params.min_wse
            self.max_wse = config_params.max_wse
            self.classes_to_keep = config_params.classes_to_keep
            self.convert_to_crs = config_params.convert_to_crs

    def load_dataset(self):
        my_path = Path(self.config_params.dataset_path)
        if my_path.is_dir():
            # the user gave the directory where the .nc swot files are stored,
            # we create the gpkg
            logging.info("Creating .gpkg file from found .nc files.")
            self.create_gpkg_from_pixC()
            self.remove_nc_file()
        elif my_path.is_file():
            if my_path.suffix == ".pt":
                # the user gave the file of the dataset,
                # we just open it :
                logging.info(
                    "Loading data from previously created PyTorch dataset."
                )
                self.load_dataset_from_path()

            elif my_path.suffix in (".gpkg", ".parquet"):
                # the user gave the file of the gpkg file,
                # we convert it to .pt
                # the .pt will be saved in the same folder
                logging.info("Creating PyTorch dataset from .gpkg file.")
                # self.contour_path = config_params.contour_path
                if self.min_wse is self.max_wse is None:
                    self.get_min_max_wse()
                self.create_dataset_from_gpkg()

    def create_gpkg_from_pixC(self):
        """
        the user gave the directory where the .nc swot files are stored,
        we create the gpkg
        """

        # read the area of interest polygon (could have been set)
        gdf_geom = gpd.read_file(self.roi_gpkg)
        self.saving_path_gpkg = self.saving_path.replace(".parquet", ".gpkg")
        # convert to gpkg format and cropping around the ROI
        # save to self.saving_path
        # by default, compute wse as well
        all_nc_files = Path(self.dataset_path).rglob("**/*.nc")
        logging.info(".nc files found : %s", all_nc_files)
        variables = [
            "height",
            "sig0",
            "classification",
            "phase_unwrapping_region",
            "phase_noise_std",
            "pole_tide",
            "solid_earth_tide",
            "load_tide_fes",
            "geoid",
            "dlatitude_dphase",
            "dlongitude_dphase",
            "dheight_dphase",
        ]

        pixc = Nc2GpkgConverter(
            all_nc_files,
            variables=variables,
            area_of_interest=gdf_geom,
        )
        pixc.database_from_nc(
            self.saving_path_gpkg,
            mode="o",
        )
        layers = list_layers(self.saving_path_gpkg)[:, 0]
        sorted_layers = sorted(
            layers,
            key=lambda x: x.split("_")[1],
        )
        with open(
            str(self.saving_path).split(".", maxsplit=1)[0] + "_id_swot.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(sorted_layers))
        variables.extend(["wse", "geometry", "longitude", "latitude"])

        gdf_all_layers = []

        for layer in layers:
            gdf = gpd.read_file(self.saving_path_gpkg, layer=layer)
            col_to_drop = [c for c in gdf.columns if c not in variables]
            gdf["layer"] = layer
            gdf = gdf.drop(columns=col_to_drop)
            gdf_all_layers.append(gdf)

        gdf_all = gpd.GeoDataFrame(
            pd.concat(gdf_all_layers, ignore_index=True),
            geometry="geometry",
            crs=gdf_all_layers[0].crs,
        )

        gdf_all.to_parquet(self.saving_path)
        Path(self.saving_path_gpkg).unlink(missing_ok=True)
        logging.info("Saved in %s", self.saving_path)

    def create_dataset_from_gpkg(self):
        """
        From a .gpkg file, this function will filter points,
        normalise values and save the dataset as a .pt.
        """

        # load all the data from the gpkg file
        self._add_raw_data()
        saving_path = self.saving_path

        # if asked, keep only points classified by swot as 4.0 or 5.0 etc
        if self.classes_to_keep:
            # all_indexes = ""
            # all_nb_points = []
            # for index in self.classes_to_keep:
            #     nb_points_this_class = self._filter_classification(float(index))
            #     all_nb_points.append(nb_points_this_class)
            #     all_indexes = all_indexes + "_" + str(index)

            # This modify the self.X and self.Y, by keeping only points we want
            all_nb_points, all_indexes = self._filter_classification()
            saving_path = (
                str(self.saving_path).split(".", maxsplit=1)[0]
                + f"_classes{all_indexes}.pt"
            )
            self.t_sizes = [sum(x) for x in zip(*all_nb_points)]
        else:
            self.t_sizes = self.nb_points["total"]

        # normalize (x, y, t, wse) between 0 and 1
        self._normalize_data()
        # save
        logging.info(
            f"dataset processed, t_sizes = {self.t_sizes}, dates = {len(self.list_dates)}, classes_to_keep = "
            + all_indexes
        )
        torch.save(
            {
                "data": torch.cat(
                    (
                        self.X,
                        self.Y.view(-1, 1),
                        self.classification.view(-1, 1),
                        self.phase_noise_level.view(-1, 1),
                        self.L.view(-1, 1),
                    ),
                    dim=1,
                ),
                "list_dates": self.list_dates,
                "t_sizes": self.t_sizes,
                "classes_to_keep": self.classes_to_keep,
                "max_wse": self.max_wse,
                "min_wse": self.min_wse,
            },
            saving_path,
        )

    def _find_layers(self):
        if Path(self.dataset_path).suffix == ".gpkg":
            layers = list_layers(self.dataset_path)[:, 0]

        if Path(self.dataset_path).suffix == ".parquet":
            gdf = gpd.read_parquet(self.dataset_path)
            layers = gdf["layer"].unique().tolist()
        return layers

    def _create_gdf(self, layer):
        if Path(self.dataset_path).suffix == ".gpkg":
            gdf = gpd.read_file(self.dataset_path, layer=layer)

        if Path(self.dataset_path).suffix == ".parquet":
            gdf = gpd.read_parquet(self.dataset_path)
            gdf = gdf[gdf["layer"] == layer]

        return gdf

    def _add_raw_data(self):
        """
        This function go trough all the layers in a .gpkg file. For each layer,
        the data is converted to tensor, then filtered based on WSE.
        The dsm mean is used to compute a vertical shift on ground points.
        """
        layers = self._find_layers()

        for layer in layers:
            logging.info("- - - - - processing layer %s", layer)

            # DATE
            # Extract date and convert to date object
            date_str = ((layer).split("_"))[0]
            file_date = date.fromisoformat(
                date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]
            )
            time_coord = float(
                str(file_date - self.reference_date).split(" ", maxsplit=1)[0]
            )
            if time_coord not in self.list_dates:
                self.list_dates.append(time_coord)

            gdf = self._create_gdf(layer)
            # convert distances from degrees to meters : "EPSG:32630" for Ebro or Vioreau
            gdf_proj = gdf.to_crs(self.convert_to_crs)
            gdf_proj["x"] = gdf_proj["longitude"]
            gdf_proj["y"] = gdf_proj["latitude"]
            gdf_proj["wse"] = (
                gdf_proj["height"]
                - gdf_proj["geoid"]
                - gdf_proj["solid_earth_tide"]
                - gdf_proj["load_tide_fes"]
                - gdf_proj["pole_tide"]
            )  # en m
            gdf_proj["t"] = time_coord
            gdf_simple = gdf_proj[
                [
                    "x",
                    "y",
                    "t",
                    "wse",
                    "classification",
                    "phase_noise_std",
                    "dheight_dphase",
                ]
            ]
            shift_vector = torch.from_numpy(
                gdf_proj[
                    ["dlongitude_dphase", "dlatitude_dphase", "dheight_dphase"]
                ].to_numpy()
            )
            # Convert to tensor
            data = gdf_simple.to_numpy()
            X_tensor = torch.from_numpy(data[:, :3])
            Y_tensor = torch.from_numpy(data[:, 3])
            classification = torch.from_numpy(data[:, 4])
            phase_noise_level = torch.from_numpy(data[:, 5])
            dheight_dphase = torch.from_numpy(data[:, 6])

            # Make sure z_shift is positive
            if torch.mean(shift_vector[:, 2]) < 0:
                shift_vector *= -1

            # Apply 1 phase shift to XY to Land and Dark Water
            X_tensor[:, 0:2] += (
                self.land_phase_shift
                * shift_vector[:, 0:2]
                * (
                    (classification[:, None] == 1.0)
                    | (classification[:, None] == 5.0)
                )
            )
            xy_points = [
                Point(X_tensor[i, 0], X_tensor[i, 1])
                for i in range(X_tensor.shape[0])
            ]
            gdf_xy_shift = gpd.GeoDataFrame(
                geometry=xy_points, crs="EPSG:4326"
            ).to_crs(self.convert_to_crs)
            X_tensor[:, 0] = torch.Tensor(gdf_xy_shift["geometry"].x.to_numpy())
            X_tensor[:, 1] = torch.Tensor(gdf_xy_shift["geometry"].y.to_numpy())

            # Apply 1 phase shift to Z
            Y_tensor += (
                self.land_phase_shift
                * shift_vector[:, 2]
                * (
                    (classification == 1.0)
                    | (classification == 2.0)
                    | (classification == 5.0)
                )
            )

            # WSE filtering
            valid_wse = torch.flatten(
                ((Y_tensor) >= self.min_wse) & ((Y_tensor) <= self.max_wse)
            )
            X_tensor = X_tensor[valid_wse]
            Y_tensor = Y_tensor[valid_wse]
            classification = classification[valid_wse]
            label = (classification == 3.0) | (
                classification == 4.0
            )  # 1 for water, 0 for everything else
            phase_noise_level = phase_noise_level[valid_wse]
            dheight_dphase = dheight_dphase[valid_wse]
            # Log nb of points in this layer (after wse filtering)
            # self.t_sizes.append(X_tensor.shape[0])
            self.nb_points["1"].append(X_tensor[classification == 1.0].shape[0])
            self.nb_points["2"].append(X_tensor[classification == 2.0].shape[0])
            self.nb_points["3"].append(X_tensor[classification == 3.0].shape[0])
            self.nb_points["4"].append(X_tensor[classification == 4.0].shape[0])
            self.nb_points["5"].append(X_tensor[classification == 5.0].shape[0])
            self.nb_points["6"].append(X_tensor[classification == 6.0].shape[0])
            self.nb_points["7"].append(X_tensor[classification == 7.0].shape[0])
            self.nb_points["total"].append(
                X_tensor[classification == 1.0].shape[0]
                + X_tensor[classification == 2.0].shape[0]
                + X_tensor[classification == 3.0].shape[0]
                + X_tensor[classification == 4.0].shape[0]
                + X_tensor[classification == 5.0].shape[0]
                + X_tensor[classification == 6.0].shape[0]
                + X_tensor[classification == 7.0].shape[0]
            )

            # Concatenate with results from other layers
            self.X = torch.cat((self.X, X_tensor), 0)
            self.Y = torch.cat((self.Y, Y_tensor))
            self.classification = torch.cat(
                (self.classification, classification)
            )
            self.L = torch.cat((self.L, label))
            self.phase_noise_level = torch.cat(
                (self.phase_noise_level, phase_noise_level)
            )
            self.dheight_dphase = torch.cat(
                (self.dheight_dphase, dheight_dphase)
            )

    def _filter_classification(self):
        """
        This function keeps only the points classified by swot as "class_to_keep".
        (ex : 4.0 for water)
        The other points are removed from the dataset.
        """
        # for index in self.classes_to_keep:
        #     nb_points_this_class = self._filter_classification(float(index))
        #     all_nb_points.append(nb_points_this_class)
        #     all_indexes = all_indexes + "_" + str(index)

        classes_mask = np.isin(self.classification, list(self.classes_to_keep))
        self.X = self.X[classes_mask]
        self.Y = self.Y[classes_mask]
        self.classification = self.classification[classes_mask]
        self.L = self.L[classes_mask]
        self.phase_noise_level = self.phase_noise_level[classes_mask]
        self.dheight_dphase = self.dheight_dphase[classes_mask]

        all_indexes = ""
        all_nb_points = []
        for index in self.classes_to_keep:
            if index:
                dict_key = str(int(index))
                all_nb_points.append(self.nb_points[dict_key])
                all_indexes = all_indexes + "_" + str(index)

        return all_nb_points, all_indexes
        # water_mask = self.classification == class_to_keep
        # self.X = self.X[water_mask]
        #
        # self.classification = self.classification[water_mask]
        # dict_key = str(int(class_to_keep))
        # return self.nb_points[dict_key]

    def _normalize_data(self):
        """
        normalize data between 0 and 1 for the model
        """

        logging.info("- - - - - data normalisation")

        # Time normalization
        self.min_date = min(self.list_dates)
        self.max_date = max(self.list_dates)

        # (X, Y, wse) normalization
        # generate self.min_x, self.max_x, self.min_y, self.max_y

        (self.X)[:, 0] = ((self.X)[:, 0] - min((self.X)[:, 0])) / (
            max((self.X)[:, 0]) - min((self.X)[:, 0])
        )

        (self.X)[:, 1] = ((self.X)[:, 1] - min((self.X)[:, 1])) / (
            max((self.X)[:, 1]) - min((self.X)[:, 1])
        )
        (self.X)[:, 2] = ((self.X)[:, 2] - self.min_date) / (
            self.max_date - self.min_date
        )

        self.Y = (self.Y - self.min_wse) / (self.max_wse - self.min_wse)

        # Cast data to float32 (default for PyTorch)
        self.X = (self.X).float()
        self.Y = (self.Y).float()
        self.L = (self.L).float()

    def load_dataset_from_path(self):
        """
        This function load a .pt pytorch dataset that has been created and saved previsouly.
        """
        dataset = torch.load(self.dataset_path)
        self.X = dataset["data"][:, :3].to(dtype=torch.float32)
        self.Y = dataset["data"][:, 3].view(-1, 1).to(dtype=torch.float32)
        self.classification = dataset["data"][:, 4]
        self.phase_noise_level = dataset["data"][:, 5]
        self.dheight_dphase = dataset["data"][:, 6]
        self.max_wse = dataset["max_wse"]
        self.min_wse = dataset["min_wse"]
        self.list_dates = dataset["list_dates"]
        self.t_sizes = dataset["t_sizes"]
        self.L = dataset["data"][:, 6].to(dtype=torch.float32)

    def shuffle(self):
        """
        Shuffle data
        """
        rand_idx = np.arange(self.X.shape[0])
        np.random.shuffle(rand_idx)
        rand_idx = torch.IntTensor(rand_idx).to(device)
        self.X = torch.index_select(self.X, 0, rand_idx)
        self.Y = torch.index_select(self.Y, 0, rand_idx)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

    def remove_nc_file(self):
        """Remove nc files after creating gpkg/geoparquet"""
        if self.config_params.remove_nc_file:
            folder_nc = Path(self.config_params.dataset_path)
            if folder_nc.exists() and any(folder_nc.iterdir()):
                for nc_file in folder_nc.glob("*.nc"):
                    try:
                        nc_file.unlink()
                    except:
                        logging.error("ERROR - not enough permissions to remove .nc files")
                        

    def select_input_from_dataset(self, target_date):
        """
        For plot purposes, select only the data points corresponding to a certain time date.
        """
        # Find index for the date
        mask_time = self.X[:, 2].to(dtype=torch.float32) == np.float32(
            target_date
        )
        # self.X = (x, y, t)
        data_this_time = self.X[mask_time]
        # self.Y = (h)
        wse_this_time = self.Y[mask_time]
        classification_this_time = self.classification[mask_time]
        phase_noise_level_this_time = self.phase_noise_level[mask_time]
        dheight_dphase_this_time = self.dheight_dphase[mask_time]

        return (
            data_this_time,
            wse_this_time,
            classification_this_time,
            phase_noise_level_this_time,
            dheight_dphase_this_time,
        )

    def get_min_max_wse(self):
        """
        Automatically get min_wse and max_wse with HYDROCRON API
        """
        logging.info(
            "--- Automatically get min_wse and max_wse with HYDROCRON "
        )
        hydrocron_response = requests.get(
            f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
            f"feature=PriorLake&feature_id={self.id_feature_swot}"
            f"&start_time={self.reference_date}T00:00:00Z"
            f"&end_time={self.end_time}T00:00:00Z"
            f"&output=csv"
            f"&fields=lake_id,time_str,wse,wse_u,area_total,quality_f",
            timeout=(10, 60),
        ).json()
        csv_str = hydrocron_response["results"]["csv"]
        df_hydrocron = pd.read_csv(StringIO(csv_str))
        df_hydrocron = df_hydrocron[
            (df_hydrocron["time_str"] != "no_data")
            & (df_hydrocron["quality_f"] != -999)
            & df_hydrocron["wse"]
            > 0
        ]
        self.min_wse = round(df_hydrocron["wse"].min())
        self.max_wse = round(df_hydrocron["wse"].max())
        logging.info("Min_wse : %s ", self.min_wse)
        logging.info("Max_wse : %s ", self.max_wse)

    def plot_data_slice(
        self,
        ax,
        xmin,
        xmax,
        dim,
        target_dates,
        with_classification=False,
        save_path_for_plots="",
    ):
        """
        This function add the swot data points on a cut-cross plot.
        The axis of the plot needs to be provided.
        xmin and xmax are used to filter data point and to "de-normalise"" the values (0-1) back to meter.
        that mus be the xmin/xmax used for the dataset creation.
        The boolean "with_classifictaion"=True will plot water point in cyan, dark water point in black
        and ground points in orange. Otherwise, all the points will be blue.
        """

        _dim = 0 if dim == "x" else 1

        for target_date in target_dates:
            # Filter for the time date
            (
                data_this_time,
                data_wse_this_time,
                classification_this_time,
                _,
                _,
            ) = self.select_input_from_dataset(target_date)

            # Filter the data where xmin < x and x < xmax
            mask = (xmin < data_this_time[:, _dim]) & (
                data_this_time[:, _dim] < xmax
            )
            data_this_time = data_this_time[mask]  # (x, y, t)
            data_wse_this_time = data_wse_this_time[mask]  # (h)

            # 'de-normalize" wse values
            wse_span = self.max_wse - self.min_wse
            data_wse_this_time = data_wse_this_time * wse_span + self.min_wse

            # Plot
            _dim = 0 if dim == "x" else 1
            if with_classification:
                classification_this_time = classification_this_time[mask]

                water = (classification_this_time == 4.0) | (
                    classification_this_time == 3.0
                )
                scatter1 = ax.scatter(
                    data_this_time[water, 1 - _dim],
                    data_wse_this_time[water],
                    c="cyan",
                    s=4,
                )

                land = (classification_this_time == 1.0) | (
                    classification_this_time == 2.0
                )
                scatter2 = ax.scatter(
                    data_this_time[land, 1 - _dim],
                    data_wse_this_time[land],
                    c="orange",
                    s=2,
                )

                dark_water = classification_this_time == 5.0
                scatter3 = ax.scatter(
                    data_this_time[dark_water, 1 - _dim],
                    data_wse_this_time[dark_water],
                    c="k",
                    s=2,
                )
                ax.set_title(
                    f"time = {round(target_date, 2)} - {sum(water)} water - {sum(land)} land - {sum(dark_water)} dark water"
                )
                ax.set_xlim([0, 1])
                ax.set_ylim([self.min_wse, self.max_wse])
                ax.legend(
                    [scatter1, scatter2, scatter3],
                    ["water", "ground", "dark water"],
                    loc="upper right",
                )
            else:
                ax.scatter(
                    data_this_time[:, 1 - _dim], data_wse_this_time, s=0.1
                )
                ax.set_xlim([0, 1])
                ax.set_ylim([self.min_wse, self.max_wse])

            if save_path_for_plots:
                plt.savefig(
                    save_path_for_plots
                    + f"plot_cross_section_time_{round(target_date, 2)}_dim_{dim}_onset_{xmin}_offset_{xmax}.png"
                )
                plt.close()
