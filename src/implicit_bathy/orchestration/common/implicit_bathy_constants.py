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
"""Provides the ImplicitBathyConstants dataclass."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImplicitBathyConstants:
    """Define some constant values for implicit bathymetry runs."""

    # general
    il_const_module: str = __file__
    il_tasks_status_filename: str = "ImplicitBathy_tasks_status.txt"
    il_builders_section: str = "builders"
