#!/usr/bin/env python3

"""Plot probability on s2 unit tests."""

from pathlib import Path
import pytest
import os
import numpy as np
import pandas as pd
from affine import Affine
from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.inference.inference_classes import (
    Processor,
    Plotter,
    SimulateInputs,
)
from implicit_bathy.config.config import BayesianTrainingConfig, InferenceConfig
import torch


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
    processor.load_model()
    return processor


@pytest.fixture
def plotter_class(processor_class):
    processor = Plotter(processor_class)
    return processor


def test_simulate_input_for_3D_grid(simulate_inputs_class):
    plot_density_x = 500
    plot_density_y = 500
    chosen_time = 0
    generated_input, x1, y1 = simulate_inputs_class.simulate_input_for_3D_grid(
        plot_density_x, plot_density_y, chosen_time
    )
    assert isinstance(generated_input, torch.Tensor)
    assert isinstance(x1, np.ndarray)
    assert isinstance(y1, np.ndarray)


def test_simulate_input_for_2D_cut(simulate_inputs_class):
    t_values = [0, 1]
    x = 0.2
    plot_density_x = 500
    plot_density_y = 500
    dim = "x"
    x0, generated_input = simulate_inputs_class.simulate_input_for_2D_cut(
        t_values, x, plot_density_x, plot_density_y, dim
    )
    assert isinstance(x0, np.ndarray)
    assert isinstance(generated_input, torch.Tensor)


def test_make_prediction(processor_class, simulate_inputs_class):
    plot_density_x = 500
    plot_density_y = 500
    chosen_time = 0
    generated_input, _, _ = simulate_inputs_class.simulate_input_for_3D_grid(
        plot_density_x, plot_density_y, chosen_time
    )
    out_dict = processor_class.make_prediction(generated_input)
    assert isinstance(out_dict, dict)


def test_format_model_output(processor_class, simulate_inputs_class):
    plot_density_x = 500
    plot_density_y = 500
    chosen_time = 0
    generated_input, _, _ = simulate_inputs_class.simulate_input_for_3D_grid(
        plot_density_x, plot_density_y, chosen_time
    )
    out_dict = processor_class.make_prediction(generated_input)
    output = processor_class.format_model_outputs(out_dict, "h", 500, 500)
    assert isinstance(output, np.ndarray)


def test_load_vt_vh(processor_class):
    dict_vt = processor_class._load_vt_vh()
    assert isinstance(dict_vt, dict)


def test_compute_roi_bounding_box_surface(processor_class):
    area = processor_class._compute_roi_bounding_box_surface()
    print(area, "area")
    assert isinstance(area, float)


def test_compute_predicted_surface(processor_class):
    chosen_time = 0
    roi_area = 1081011.0
    water_surface = processor_class.compute_predicted_surface(
        chosen_time, roi_area
    )
    assert isinstance(water_surface, float)


def test_rasterize_contour(processor_class):
    width = 38
    height = 44
    transform = Affine(1.0, 0.0, 4.50, 0.0, -1.0, 47.26)
    crs = "EPSG:4326"
    raster_contour = processor_class.rasterize_contour(
        width, height, transform, crs
    )
    assert isinstance(raster_contour, np.ndarray)


def test_mean_absolute_rel_error(processor_class):

    df = pd.DataFrame(
        {"vt": [0.45, 0.43, 0.46, 0.50], "pred": [0.43, 0.44, 0.47, 0.52]}
    )
    mare, mae, mre, me = processor_class.mean_absolute_rel_error(
        df["vt"], df["pred"]
    )
    assert all(isinstance(x, float) for x in [mare, mae, mre, me])


def test_mean_error_delta_vol(processor_class):

    df = pd.DataFrame(
        {"vt": [0.45, 0.43, 0.46, 0.50], "pred": [0.43, 0.44, 0.47, 0.52]}
    )
    mae, maern, maetv = processor_class.mean_error_delta_vol(
        df["vt"], df["pred"], 0.5
    )
    assert all(isinstance(x, float) for x in [mae, maern, maetv])


def test_quantile_68(processor_class):
    df = pd.DataFrame(
        {"vt": [0.45, 0.43, 0.46, 0.50], "pred": [0.43, 0.44, 0.47, 0.52]}
    )
    quantile_abs_rel_errors, quantile_abs_errors = processor_class.quantile_68(
        df["vt"], df["pred"]
    )
    assert all(
        isinstance(x, float)
        for x in [quantile_abs_rel_errors, quantile_abs_errors]
    )


def test_retrieve_gt_dsm(plotter_class):
    data_array, transform = plotter_class.retrieve_gt_dsm()
    assert isinstance(data_array, np.ndarray)
    assert isinstance(transform, Affine)
