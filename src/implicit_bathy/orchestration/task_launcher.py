#!/usr/bin/env python3
# =========================================================================
#   Program:   iota2
#
#   Copyright (c) CESBIO. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
#
# =========================================================================
"""
Task launcher
"""

import argparse
import logging
import os
import sys

import dill

LOGGER = logging.getLogger("distributed.worker")


def set_working_dir_parameter(t_kwargs: dict, worker_working_dir: str) -> dict:
    """."""
    new_t_kwargs = t_kwargs.copy()
    working_dir_names = [
        "working_directory",
        "pathWd",
        "workingDirectory",
        "working_dir",
        "path_wd",
        "WORKING_DIR",
    ]
    for working_dir_name in working_dir_names:
        if working_dir_name in new_t_kwargs:
            new_t_kwargs[working_dir_name] = worker_working_dir
    return new_t_kwargs


def task_launcher(func_pkl: str) -> None:
    """function dedicated to undill function and its kwargs then launch it

    useful to travel through cluster

    Parameters
    ----------
    func_pkl : str
        path to a function serialized thanks to dill
    func_kw_pkl : str
        path to a dictionary serialized thanks to dill
        which represents function kwargs
    """
    func, func_kwargs, logger_lvl = dill.load(open(func_pkl, "rb"))
    worker_logger = logging.getLogger("distributed.worker")
    worker_logger.setLevel(logger_lvl)
    if (tmpdir := os.environ.get("TMPDIR")) is not None:
        f_kwargs = set_working_dir_parameter(func_kwargs, tmpdir)
    else:
        f_kwargs = func_kwargs
    func(**f_kwargs)


def main():
    """Main"""
    parser = argparse.ArgumentParser(description="launch function thanks to kwargs")
    parser.add_argument(
        "-dill_file",
        dest="dill_file",
        help="function file pickled",
        required=True,
    )
    args = parser.parse_args()

    task_launcher(args.dill_file)


if __name__ == "__main__":
    sys.exit(main())
