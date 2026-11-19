#!/usr/bin/env python3

"""Plot VS unit tests."""

from pathlib import Path
import pytest
import os
import numpy as np
import pandas as pd
from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.inference.plot_vs import PlotterVS
from implicit_bathy.inference.inference_classes import Processor, SimulateInputs
from implicit_bathy.config.config import BayesianTrainingConfig, InferenceConfig


@pytest.fixture
def dataset_param(tmp_path):
    dataset_params = DatasetConfig(
        dataset_path=os.path.join(
            (tmp_path) / "output" / "data" / "dataset_classes_1_2_3_4.pt"
        ),
        saving_path=os.path.join(
            (tmp_path) / "output" / "data" / "dataset.gpkg"
        ),
        roi_gpkg=os.path.join(tmp_path / "input" / "roi_cercey.gpkg"),
    )
    return dataset_params


@pytest.fixture
def config_params_inference(tmp_path):
    dataset_inference_params = InferenceConfig(
        dsm_path=Path((tmp_path) / "output" / "dsm" / "COP30.tif"),
        vt_path=Path(tmp_path / "input" / "df_cercey.csv"),
        start_time="30-07-2023",
        end_time="31-08-2023",
        swot_wse=[],
        dataset_dir=None,
        contour_path=Path(tmp_path / "input" / "roi_cercey.gpkg"),
        convert_to_crs="EPSG:32631",
        dataset_path=os.path.join(
            (tmp_path) / "output" / "data" / "dataset_classes_1_2_3_4.pt"
        ),
        plot_density=128,
        fig_size=5,
        id_feature_swot=2320106722,
        save_path_for_plots=Path((tmp_path) / "output" / "inference"),
        max_wse_for_X_plot=410,
        min_wse_for_dsm=370,
        max_wse_for_dsm=400,
        model_path=Path(
            (tmp_path)
            / "output"
            / "train"
            / "bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4"
            / "bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4_epoch9999.pth"
        ),
        water_threshold_proba=0.99,
        ground_threshold_proba=0.1,
        plot_p_w_on_s1_images=False,
        save_s2_tif=False,
        save_s1_tif=False,
        deg_polyfit=2,
        surface_inference="bayesian",
        volume_inference="bayesian",
        p_w_filter=0.5,
        s2_tif_path=Path((tmp_path) / "output" / "data" / "S2.tif"),
    )
    return dataset_inference_params


@pytest.fixture
def config_params_bayesian(tmp_path):
    dataset_params = BayesianTrainingConfig(
        seed=3,
        monitor_every_epoch=10,
        dataset_path=Path(
            (tmp_path) / "output" / "data" / "dataset_classes_1_2_3_4.pt"
        ),
        save_path=Path((tmp_path) / "output" / "train"),
        bayesian_model=True,
        ground_model="garf",
        ground_model_params=(16, 4, 0.1),
        water_model="mlp",
        water_model_params=(16, 4, None),
        lr=0.005,
        optimizer_name="Adam",
        n_epochs=10000,
        plot_every_epoch=1000,
        weighted_by_phase_noise=True,
        weighted_function="inverse",
        ground_prob="cauchy",
        wl_optimizer_name="AdamW",
        wl_lr=0.0005,
        confidence_threshold=0.999,
        n_epochs_wloss=5500,
        lambda_wloss=1,
        lambda_uncert=None,
    )
    return dataset_params


@pytest.fixture
def simulate_inputs_class(dataset_param):
    simulate_inputs = SimulateInputs()
    return simulate_inputs


@pytest.fixture
def multidatelake_dataset(dataset_param):
    m = MultiDateLakeDataset(dataset_param)
    m.load_dataset()
    return m


@pytest.fixture
def processor_class(
    config_params_inference,
    config_params_bayesian,
    multidatelake_dataset,
    simulate_inputs_class,
):
    processor = Processor(
        config_params_inference,
        config_params_bayesian,
        multidatelake_dataset,
        simulate_inputs_class,
    )
    return processor


@pytest.fixture
def plot_vs_class(processor_class):
    plot_vs = PlotterVS(processor_class)
    return plot_vs


def test_compute_predicted_volume(plot_vs_class):
    chosen_time = 0
    vol = plot_vs_class._compute_predicted_volume(chosen_time)
    assert isinstance(vol, float)


def test_get_data_vt(plot_vs_class):
    plot_vs_class.get_data_vt()
    assert isinstance(plot_vs_class.data_vt, pd.DataFrame)


def test_get_data_hydrocron(plot_vs_class):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    assert isinstance(plot_vs_class.df_hydrocron, pd.DataFrame)


def test_get_swot_wse(plot_vs_class):
    t_values = [0, 1]
    plot_vs_class.get_swot_wse(t_values)
    assert isinstance(plot_vs_class.swot_wse, list)
    assert all(isinstance(i, float) for i in plot_vs_class.swot_wse)


def test_get_pred_data_t(plot_vs_class):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    assert isinstance(plot_vs_class.data_pred, pd.DataFrame)


def test_plot_relative_wse_t_hydrocron(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = (
        inference_pth / "relative_to_first_value_wse_time_serie_hydrocron.png"
    )
    svg_path = (
        inference_pth / "relative_to_first_value_wse_time_serie_hydrocron.svg"
    )
    paths = [png_path, svg_path]
    for path in paths:
        path.unlink(missing_ok=True)

    plot_vs_class.plot_relative_wse_t_hydrocron()
    assert png_path.exists()
    assert svg_path.exists()


def test_plot_absolute_wse_t(plot_vs_class, tmp_path):
    t_values = [0, 1]
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "absolute_wse_time_serie.png"
    png_path.unlink(missing_ok=True)
    plot_vs_class.plot_absolute_wse_t(t_values)
    assert png_path.exists()


def test_plot_absolute_wse_t_hydrocron(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "absolute_wse_time_serie_hydrocron.png"
    png_path.unlink(missing_ok=True)
    plot_vs_class.plot_absolute_wse_t_hydrocron()
    assert png_path.exists()


def test_plot_surface_t_hydrocron(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()

    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "absolute_surface_time_serie.png"
    svg_path = inference_pth / "absolute_surface_time_serie.svg"
    paths = [png_path, svg_path]
    for path in paths:
        path.unlink(missing_ok=True)
    plot_vs_class.plot_surface_t_hydrocron()
    assert png_path.exists()
    assert svg_path.exists()


def test_plot_volume_t(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "relative_volume_time_serie.png"
    svg_path = inference_pth / "relative_volume_time_serie.svg"
    paths = [png_path, svg_path]
    for path in paths:
        path.unlink(missing_ok=True)
    plot_vs_class.plot_volume_t()
    assert png_path.exists()
    assert svg_path.exists()


def test_extra_linear(plot_vs_class):
    h = np.array(
        [
            378.9627,
            378.95932,
            378.9552,
            378.9511,
            378.94696,
            378.94266,
            378.9376,
            378.93094,
            378.9246,
            378.916,
            378.9167,
            378.9174,
        ]
    )

    y_smooth = np.array(
        [
            0.43813072,
            0.43639952,
            0.43481527,
            0.43328153,
            0.4316284,
            0.4298972,
            0.42817519,
            0.42617765,
            0.4244786,
            0.42341784,
            0.4229219,
            0.42304129,
        ]
    )
    r = plot_vs_class.extra_linear(h, y_smooth)
    assert callable(r[0])
    assert isinstance(r[1], float)
    assert isinstance(r[2], float)


def test_plot_surface_wse(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "relative_surface_wse.png"
    png_path.unlink(missing_ok=True)
    surfaces = [
        0.43809398,
        0.43645463,
        0.4349117,
        0.43307948,
        0.43172942,
        0.43009007,
        0.42796855,
        0.42632919,
        0.42468983,
        0.42285761,
        0.42295404,
        0.42324334,
        0.42343621,
        0.42382194,
        0.42391837,
        0.42420767,
        0.42420767,
        0.42420767,
        0.42449697,
        0.42449697,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
        0.4245934,
    ]
    sigma_surface = plot_vs_class.plot_surface_wse(surfaces)
    assert isinstance(sigma_surface, list)
    assert png_path.exists()


def test_compute_dynamic_range_percentiles(plot_vs_class):
    z = plot_vs_class.compute_dynamic_range_percentiles(370, 380, 0.25)
    assert isinstance(z, float)


def test_plot_volume_wse(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "relative_volume_wse.png"
    png_path.unlink(missing_ok=True)
    delta_vol = [
        0.0,
        -0.05449275,
        -0.10800725,
        -0.16058475,
        -0.21115575,
        -0.30436913,
        -0.34994075,
        -0.39465538,
        -0.43861,
        -0.48179175,
        -0.52417462,
        -0.56572563,
        -0.6064865,
        -0.646505,
        -0.68576737,
        -0.7282865,
        -0.77053975,
        -0.8120995,
        -0.85279975,
        -0.8962915,
        -0.94224712,
        -0.98708775,
        -1.03079487,
        -1.07339575,
        -1.11484525,
        -1.15529275,
        -1.21254087,
        -1.2629065,
        -1.3114835,
        -1.35826806,
        -1.40330806,
        -1.44508644,
        -1.482257,
    ]
    plot_vs_class.plot_volume_wse(delta_vol)
    assert png_path.exists()


def test_plot_volume_area(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_path = inference_pth / "relative_volume_area.png"
    png_path.unlink(missing_ok=True)

    plot_vs_class.plot_volume_area()
    assert png_path.exists()


def test_export_csv(plot_vs_class, tmp_path):
    plot_vs_class.get_data_vt()
    plot_vs_class.get_data_hydrocron()
    plot_vs_class.get_pred_data_t()
    inference_pth = Path((tmp_path) / "output" / "inference")
    csv_path = inference_pth / "data.csv"
    csv_path.unlink(missing_ok=True)
    plot_vs_class.export_csv()
    assert csv_path.exists()


def test_compute_metrics(plot_vs_class):
    col_true = "relative_wse_vt"
    col_pred = "relative_wse_hc"
    df = pd.DataFrame(
        {
            "relative_wse_vt": [-0.294, -0.294, -1.963, -2.659],
            "relative_wse_hc": [-0.29, -0.29, -2.07, -2.669999],
        }
    )
    dict_indicators = plot_vs_class.compute_metrics(df, col_true, col_pred)
    assert isinstance(dict_indicators, dict)
