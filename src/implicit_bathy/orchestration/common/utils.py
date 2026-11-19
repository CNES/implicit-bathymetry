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
"""This module contains decorators and utils functions."""

import argparse
import logging
import os
import subprocess
import time
from functools import wraps
from timeit import default_timer as timer

LOGGER = logging.getLogger("distributed.worker")


class RemoveInStringList:
    """Remove element decorator to remove strings in a list."""

    def __init__(self, *args: str):
        self.pattern_list = args

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, fun):
        @wraps(fun)
        def wrapped_f(*args):
            results = fun(*args)
            results_filtered = []
            for elem in results:
                pattern_found = False
                for pattern in self.pattern_list:
                    if pattern in elem:
                        pattern_found = True
                        break
                if pattern_found is False:
                    results_filtered.append(elem)
            return results_filtered

        return wrapped_f


class TimeIt:
    """Design a chronometer decorator."""

    def __init__(self, fun):
        self.func = fun
        self.time_elapse = 0

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):

        start = time.time()
        results = self.func(*args, **kwargs)
        end = time.time()
        self.time_elapse = end - start
        LOGGER.info(
            "ELAPSED time during the call of %s : %s [sec]",
            self.func.__name__,
            self.time_elapse,
        )
        return results


def run(
    cmd: str,
    desc: str | None = None,
    env=os.environ,
    logger: logging.Logger = LOGGER,
):
    """Launch a system command and raise an execption if fail.

    Parameters
    ----------
    cmd:
        the system command to be launched
    desc:
        an optional description of the command for log_dir
    env;
        the environ variable if None, os.environ is used
    logger:
        by default module LOGGER value is used
    """
    # Create subprocess
    start = timer()
    logger.debug("run command :", cmd)

    proc = subprocess.Popen(
        cmd,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Get output as strings
    out, err = proc.communicate()

    # Get return code
    rtc = proc.returncode

    stop = timer()

    # Log outputs
    if desc is not None:
        logger.debug(desc)
    if out is not None:
        logger.debug("out/err: ", out.rstrip())
    logger.debug(f"Done in {stop - start} seconds")

    # Log error code
    if rtc != 0:
        logger.error(f"Command {cmd}  exited with non-zero return code {rtc}")
        exception_msg = f"Launch command fail : {cmd} \n {out} \n {err}"
        raise Exception(exception_msg)
    return rtc


def str2bool(val: str) -> bool:
    """Convert a string vue to boolean.

    usage: use in argParse as function to parse options
    Parameters
    ----------
    val:
        the value to convert
    """
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if val.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def is_nomenclature_castable_to_int(labels_table: dict[int, int | str]):
    """Try to convert labels to int."""
    all_castable = []
    for _, user_label in labels_table.items():
        try:
            _ = int(user_label)
            all_castable.append(True)
        except ValueError:
            all_castable.append(False)
    return all(all_castable)
