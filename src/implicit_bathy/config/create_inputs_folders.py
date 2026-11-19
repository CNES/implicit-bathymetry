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
Create inputs folders
"""

import argparse
import logging
import sys
from pathlib import Path

from implicit_bathy.config.read_config_file import ReadConfigFile

logging.basicConfig(level=logging.INFO)


def create_folder(config_file: str) -> None:
    """
    Create arbo expected
    """
    cfg = ReadConfigFile(config_file)
    out = cfg.config.dataprocessing.output_path
    if not Path(out).exists():
        Path(out).mkdir()
    list_folder = [
        "cfg",
        "data",
        "dsm",
        "inference",
        "inputs_nc",
        "logs",
        "train",
    ]
    for name in list_folder:
        folder = Path(out) / name.strip()
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)
    logging.info(
        "Folders cfg, data, dsm, inference, inputs_nc, logs, train created "
    )


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
    create_folder(args.config)


if __name__ == "__main__":
    sys.exit(main())
