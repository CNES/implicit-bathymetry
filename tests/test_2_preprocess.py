#!/usr/bin/env python3

"""Prepocess unit tests."""

from pathlib import Path
import pytest
import os
import geopandas as gpd
import torch
from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset


@pytest.fixture
def dataset_param(tmp_path):
    dataset_params = DatasetConfig(
        dataset_path=os.path.join((tmp_path) / "output" / "inputs_nc"),
        saving_path=os.path.join(
            (tmp_path) / "output" / "data" / "dataset.parquet"
        ),
        roi_gpkg=os.path.join(tmp_path / "input" / "roi_cercey.gpkg"),
        remove_nc_file=False,
    )
    return dataset_params


@pytest.fixture
def dataset_param_pt(tmp_path):
    dataset_params = DatasetConfig(
        dataset_path=os.path.join(
            (tmp_path) / "output" / "data" / "dataset.parquet"
        ),
        roi_gpkg=os.path.join(tmp_path / "input" / "roi_cercey.gpkg"),
        id_feature_swot=2320106762,
        classes_to_keep=[1, 2, 3, 4],
        min_wse=376,
        max_wse=385,
        land_phase_shift=0.5,
        convert_to_crs="EPSG:32631",
        saving_path=os.path.join((tmp_path) / "output" / "data" / "dataset.pt"),
    )
    return dataset_params


def test_create_gpkg_from_pixC(dataset_param, tmp_path):

    path_dataset_parquet = Path(
        (tmp_path) / "output" / "data" / "dataset.parquet"
    )
    if path_dataset_parquet.exists():
        path_dataset_parquet.unlink()
    m = MultiDateLakeDataset(dataset_param)
    m.create_gpkg_from_pixC()
    assert path_dataset_parquet.exists()

    dataset_gpkg = gpd.read_parquet(path_dataset_parquet)
    expected_columns = [
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
        "wse",
        "geometry",
        "longitude",
        "latitude",
        "layer",
    ]
    missing = set(expected_columns) - set(dataset_gpkg.columns)

    assert not missing, f"Missing columns : {missing}"


def test_create_dataset_from_gpkg(dataset_param_pt, tmp_path):
    path_dataset_pt = Path(
        (tmp_path) / "output" / "data" / "dataset_classes_1_2_3_4.pt"
    )
    if path_dataset_pt.exists():
        path_dataset_pt.unlink()

    m = MultiDateLakeDataset(dataset_param_pt)
    m.create_dataset_from_gpkg()

    assert path_dataset_pt.exists()
    data_torch = torch.load(path_dataset_pt, map_location="cpu")

    x_normalized = data_torch["data"][:, 0]
    y_normalized = data_torch["data"][:, 1]
    t_normalized = data_torch["data"][:, 2]
    wse_normalized = data_torch["data"][:, 3]
    classif = data_torch["data"][:, 4]
    label = data_torch["data"][:, 6]

    # Tests for the _normalize_data() function
    assert x_normalized.min() >= 0, (
        f"Invalid minimum longitude : {x_normalized.min()}"
    )
    assert x_normalized.max() <= 1, (
        f"Invalid maximum longitude : {x_normalized.max()}"
    )

    assert y_normalized.min() >= 0, (
        f"Invalid minimum latitude : {y_normalized.min()}"
    )
    assert y_normalized.max() <= 1, (
        f"Invalid maximum latitude : {y_normalized.max()}"
    )

    assert t_normalized.min() >= 0, (
        f"Invalid minimum time : {t_normalized.min()}"
    )
    assert t_normalized.max() <= 1, (
        f"Invalid maximum time : {t_normalized.max()}"
    )

    assert wse_normalized.min() >= 0, (
        f"Invalid minimum wse : {wse_normalized.min()}"
    )
    assert wse_normalized.max() <= 1, (
        f"Invalid maximum wse : {wse_normalized.max()}"
    )

    # Tests for the _filter_classification() function
    assert classif.min() >= 1, f"Invalid minimum label : {classif.min()}"
    assert classif.max() <= 4, f"Invalid maximum label : {classif.max()}"

    assert label.min() >= 0, f"Invalid minimum label : {label.min()}"
    assert label.max() <= 1, f"Invalid maximum label : {label.max()}"


def test_get_min_max_wse(dataset_param_pt, tmp_path):

    m = MultiDateLakeDataset(dataset_param_pt)
    m.get_min_max_wse()
    assert isinstance(m.min_wse, int), f"Min wse is not an integer: {m.min_wse}"
    assert isinstance(m.max_wse, int), f"Min wse is not an integer: {m.max_wse}"
