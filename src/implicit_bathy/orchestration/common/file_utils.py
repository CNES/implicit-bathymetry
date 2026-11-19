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
"""This module contains a lot of useful file manipulation functions"""

import errno
import importlib
import logging
import os
from pathlib import Path

from implicit_bathy.orchestration.common.utils import RemoveInStringList

LOGGER = logging.getLogger("distributed.worker")


def verify_import(module_path: str):
    """Check if a module can be imported

    Parameters
    ----------
    module_path:
        the path to module

    """
    spec = importlib.util.spec_from_file_location(
        module_path.split(os.sep)[-1].split(".")[0], module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_writable_directory(directory_path: str) -> bool:
    """
    ensure a directory is writable
    Parameters
    ----------
    directory_path:
        the directory to be tested
    """
    out = True
    try:
        ensure_dir(directory_path)
    except OSError:
        out = False
    return out


def get_implicit_bathy_project_dir():
    """Get projet dir"""
    parent = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    )
    implicit_bathy_dir = os.path.abspath(os.path.join(parent, os.pardir))
    return implicit_bathy_dir


def ensure_dir(dirname: str, raise_exe: bool = True):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        Path(dirname).mkdir(parents=True)
    except OSError as err:
        if err.errno != errno.EEXIST:
            if raise_exe:
                raise


@RemoveInStringList(".aux.xml", ".sqlite-journal", ".pyc", "~")
def file_search_and(PathToFolder, AllPath, *names):
    """
    search all files in a folder or sub folder which contains all names in their name

    IN :
        - PathToFolder : target folder
                ex : /xx/xxx/xx/xxx
        - *names : target names
                ex : "target1","target2"
    OUT :
        - out : a list containing all file name (without extension) which are containing all name
    """
    out = []
    for path, _, files in os.walk(str(PathToFolder), followlinks=True):
        for _, file in enumerate(files):
            flag = 0
            for name in names:
                if file.count(name) != 0:
                    flag += 1
            if flag == len(names):
                if not AllPath:
                    out.append(file.split(".")[0])
                else:
                    path_out = path + "/" + file
                    out.append(path_out)
    return out
