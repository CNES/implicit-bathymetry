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
Create configuration all
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from implicit_bathy.config.read_config_file import (
    Dataprocessing,
    Inference,
    ReadConfigFile,
    Train,
    get_types_dict,
)


class CreationConfigsAll:
    """
    Creates global configuration files.
    Attributes:
        cfg (ReadConfigFile) :  Configuration template file reader
        out (Path) : Global ouput path
        template_file (Path) : Configuration template file
        params_csv_file (Path) : Parameters csv file

    """

    def __init__(self, config_file_template, params_csv_file):
        """
        Args:
            config_file_template (str): Configuration template file
            params_csv_file (str): Parameters csv file
        """
        self.template_file = Path(config_file_template)
        self.cfg = ReadConfigFile(self.template_file)
        self.out = Path(self.cfg.config.dataprocessing.output_path)
        self.params_csv_file = Path(params_csv_file)

    def apply_new_param(self, template, dict_param):
        """
        Apply new values on parameters
        Args:
            template (dict): Dict of template
            dict_param (dict): Dict with parameters and values
        """
        for values in template.values():
            for key, value in dict_param.items():
                if key in values:
                    values[key] = value
        return template

    def convert_param_type(self, template, section, param, expected_type):
        """
        Convert parameter values to their expected types
        Args:
            template (dict): template's dict
            section (str): section to change
            param (str): parameter to change
            expected_type : expected type
        """
        if template[section][param]:
            if expected_type is bool:
                template[section][param] = bool(template[section][param])
            if expected_type is int:
                template[section][param] = int(template[section][param])
            if expected_type is float:
                template[section][param] = float(template[section][param])

    def create_all_config_global(self):
        """
        Create config global
        """
        global_all_config_path = self.out / "cfg_all"
        df_param = pd.read_csv(
            self.params_csv_file, index_col=0, dtype=str, keep_default_na=True
        )
        df_param = df_param.where(df_param.notna(), None)

        for _, col_data in df_param.items():
            with open(self.template_file, encoding="utf-8") as f:
                template = yaml.safe_load(f)

            dict_param = dict(zip(col_data.keys().to_list(), col_data.values))
            output_path = "out_path_" + col_data["id"]
            name_config = "config_" + col_data["id"] + ".yaml"
            config_path = global_all_config_path / name_config
            output_path = self.out / output_path
            dataset_path = self.out / "inputs_nc"
            cop30_path = self.out / "dsm" / "COP30.tif"

            template = self.apply_new_param(template, dict_param)

            if not template.get("dataprocessing").get("dsm_path"):
                template["dataprocessing"]["dsm_path"] = str(cop30_path)
            if not template["dataprocessing"]["dataset_path"]:
                template["dataprocessing"]["dataset_path"] = str(dataset_path)

            template["dataprocessing"]["output_path"] = str(output_path)
            template["inference"]["id_feature_swot"] = template[
                "dataprocessing"
            ]["id_feature_swot"]
            template["inference"]["s2_tif_path"] = str(
                self.out / "data" / "S2.tif"
            )

            classes_to_keep = template["dataprocessing"]["classes_to_keep"]
            save_model_epoch = template["train"]["save_model_epoch"]

            if isinstance(classes_to_keep, str):
                template["dataprocessing"]["classes_to_keep"] = [
                    int(x) for x in classes_to_keep.split("_")
                ]
            if isinstance(save_model_epoch, str):
                template["train"]["save_model_epoch"] = [
                    int(x) for x in save_model_epoch.split("_")
                ]

            for section, cl in (
                ("dataprocessing", Dataprocessing),
                ("train", Train),
                ("inference", Inference),
            ):
                for key, value in get_types_dict(cl).items():
                    self.convert_param_type(template, section, key, value)

            yaml_config = yaml.dump(template, sort_keys=False)
            yaml_config = yaml_config.replace(": null", ":")
            Path(config_path).write_text(yaml_config, encoding="utf-8")


def main():
    """
    Main entry point of the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("template_file", help="Template file")
    parser.add_argument("params_csv_file", help="Param csv file")
    args = parser.parse_args()

    cfg_all_config = CreationConfigsAll(
        args.template_file, args.params_csv_file
    )
    cfg_all_config.create_all_config_global()


if __name__ == "__main__":
    main()
