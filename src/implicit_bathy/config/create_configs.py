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
Create configs
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import yaml

from implicit_bathy.config.read_config_file import ReadConfigFile


class CreationConfigs:
    """
    Creates configuration files.
    Attributes:
        cfg (ReadConfigFile) :  Configuration template file reader
        out (Path) : Ouput path
        cfg_path (Path) : Configurations files folder
        params_csv_file (Path) : Parameters csv file
        data_gpkg_path (str / None): Configuration gpkg file path
        input_nc_path (Path) : Input nc folder
        roi_path (Path) : Roi file path
    """

    def __init__(self, config_file, data_gpkg_path=None):
        """
        Args:
            config_file (str): Configuration template file
            data_gpkg_path (str): Configuration gpkg file path
        """
        self.cfg = ReadConfigFile(config_file)
        self.out = Path(self.cfg.config.dataprocessing.output_path)
        self.cfg_path = self.out / "cfg"
        self.data_prqt_path = data_gpkg_path
        self.input_nc_path = (
            self.out / "inputs_nc"
            if self.cfg.config.dataprocessing.dataset_path is None
            else Path(self.cfg.config.dataprocessing.dataset_path)
        )
        self.roi_path = Path(self.cfg.config.dataprocessing.roi_gpkg)
        self.create_config_gpkg()
        self.create_config_pt()
        self.create_config_train()
        self.create_config_inference()

    def create_config_gpkg(self):
        """
        Create gpkg config
        """
        self.dataprocessing_config = json.loads(
            self.cfg.config.dataprocessing.json()
        )
        self.data_path = self.out / "data"
        parameters = ["roi_gpkg", "dataset_path", "remove_nc_file"]
        config_gpkg = {
            k: self.dataprocessing_config[k]
            for k in parameters
            if k in self.dataprocessing_config
        }
        if self.data_prqt_path is None:
            self.data_prqt_path = self.data_path / "dataset.parquet"

        if not config_gpkg["dataset_path"]:
            config_gpkg["dataset_path"] = str(self.out / "inputs_nc")
        config_gpkg["saving_path"] = str(self.data_prqt_path)
        with open(
            (self.cfg_path / "config_data_gpkg.yaml"), "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(config_gpkg, f)
        logging.info("config_data_gpkg.yaml created")

    def create_config_pt(self):
        """
        Create pt config
        """
        parameters = ["output_path", "start_date", "end_date", "remove_nc_file"]
        config_pt = {
            k: self.dataprocessing_config[k]
            for k in self.dataprocessing_config
            if k not in parameters
        }
        self.data_pt = self.data_path / "dataset.pt"
        epsg = self.retrieve_epsg()
        config_pt["convert_to_crs"] = epsg
        if not config_pt["dsm_path"]:
            self.dsm_path = self.out / "dsm" / "COP30.tif"
            config_pt["dsm_path"] = str(self.dsm_path)
        else:
            self.dsm_path = config_pt["dsm_path"]
        if config_pt["id_feature_swot"]:
            self.id_feature_swot = config_pt["id_feature_swot"]
        else:
            self.id_feature_swot = None
        config_pt["dataset_path"] = str(self.data_prqt_path)
        config_pt["saving_path"] = str(self.data_pt)
        config_pt = {k: ("" if v is None else v) for k, v in config_pt.items()}
        if not config_pt["min_wse"] and not config_pt["max_wse"]:
            config_pt["min_wse"] = None
            config_pt["max_wse"] = None

        yaml_config = yaml.dump(config_pt, sort_keys=False)
        yaml_config = yaml_config.replace(": null", ":")
        Path(self.cfg_path / "config_data_pt.yaml").write_text(
            yaml_config, encoding="utf-8"
        )

        all_indexes = ""
        logging.info("config_data_pt.yaml created")
        for index in config_pt["classes_to_keep"]:
            all_indexes = all_indexes + "_" + str(index)
        self.data_pt_classes = (
            self.data_path / f"dataset_classes{all_indexes}.pt"
        )

    def create_config_train(self):
        """
        Create train config
        """
        config_train = json.loads(self.cfg.config.train.json())
        config_train["save_path"] = str(self.out / "train")
        config_train["dataset_path"] = str(self.data_pt_classes)
        config_train["save_model"] = False
        if config_train["water_model"] == "lut":
            config_train["n_dates"] = len(list(self.input_nc_path.iterdir()))
        else:
            config_train["n_dates"] = ""
        yaml_config = yaml.safe_dump(config_train, sort_keys=False)
        yaml_config = yaml_config.replace(": null", ":")
        Path(self.cfg_path / "config_train.yaml").write_text(
            yaml_config, encoding="utf-8"
        )

        logging.info("config_train.yaml created")

    def get_start_date_end_date(self):
        """
        Get first time and end time (nc)
        """
        logging.info(self.input_nc_path)
        list_date = [
            f.name.split("_")[7][0:8] for f in self.input_nc_path.iterdir()
        ]
        logging.info(list_date)
        start_date = min(list_date)
        end_date = max(list_date)
        start_date = f"{start_date[6:]}-{start_date[4:6]}-{start_date[:4]}"
        end_date = f"{end_date[6:]}-{end_date[4:6]}-{end_date[:4]}"
        return start_date, end_date

    def recreate_model_path(self):
        """
        Recreate model path name
        """
        config_train = json.loads(self.cfg.config.train.json())
        if config_train["bayesian_model"]:
            model = "bayesian_"
        else:
            model = "standard_"
        w1 = config_train["width1"]
        d1 = config_train["depth1"]
        ground_description = ""

        # GROUND description
        if config_train["ground_model"].lower() == "mlp":
            indim = 2 + 4 * config_train["x_freq_enc_dim"]
            ground_description = f"GROUND-mlp-w-{w1}-d-{d1}-inDim-{indim}"
        elif config_train["ground_model"].lower() == "garf":
            str_gauss_act_sigma = str(
                config_train["gaussian_act_sigma"]
            ).replace(".", "-")
            ground_description = (
                f"GROUND-garf-w-{w1}-d-{d1}-sigma-" + str_gauss_act_sigma
            )
        else:
            raise ValueError("Ground model not defined")

        # WATER description
        if config_train["water_model"] == "mlp":
            w2 = config_train["width2"]
            d2 = config_train["depth2"]
            water_description = f"WATER-mlp-w-{w2}-d-{d2}"
        elif config_train["water_model"] == "lut":
            n_dates = len(list(self.input_nc_path.iterdir()))
            water_description = f"WATER-lut-ndates-{n_dates}"
        else:
            raise ValueError("Water model not defined")

        number_epoch = config_train["n_epochs"] - 1
        description = model + ground_description + "_" + water_description
        model_path = description + f"_epoch{number_epoch}.pth"
        return description, model_path

    def retrieve_epsg(self):
        """
        Retrive utm crs
        """
        return str(gpd.read_file(self.roi_path).estimate_utm_crs())

    def create_config_inference(self):
        """
        Create inference config
        """
        config_inference = json.loads(self.cfg.config.inference.json())
        config_inference["dataset_path"] = str(self.data_pt_classes)
        description, model_path = self.recreate_model_path()
        config_inference["model_path"] = str(
            self.out / "train" / description / model_path
        )
        epsg = self.retrieve_epsg()
        config_inference["convert_to_crs"] = epsg
        start_date, end_date = self.get_start_date_end_date()
        config_inference["start_time"] = start_date
        config_inference["end_time"] = end_date
        config_inference["id_feature_swot"] = self.id_feature_swot
        config_inference["dsm_path"] = str(self.dsm_path)
        config_inference["contour_path"] = self.dataprocessing_config[
            "roi_gpkg"
        ]
        config_inference["save_path_for_plots"] = str(self.out / "inference")
        config_inference["plot_cross_section"] = True
        config_inference["plot_DSM"] = True
        config_inference["plot_error_histograms"] = True
        config_inference["plot_volume_surface"] = True
        config_inference["plot_p_w_on_s2_images"] = True
        config_inference["plot_p_w_on_s1_images"] = False
        config_inference["save_s1_tif"] = False
        config_inference["save_s2_tif"] = False
        config_inference["deg_polyfit"] = 2
        if not config_inference["s2_tif_path"]:
            config_inference["s2_tif_path"] = str(self.out / "data" / "S2.tif")
        yaml_config = yaml.dump(config_inference, sort_keys=False)
        yaml_config = yaml_config.replace(": null", ":")
        Path(self.cfg_path / "config_inference.yaml").write_text(
            yaml_config, encoding="utf-8"
        )
        logging.info("config_inference.yaml created")
