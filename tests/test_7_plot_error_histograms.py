#!/usr/bin/env python3

"""Plot Error histograms unit tests."""

from pathlib import Path
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.inference.plot_error_histograms import (
    PlotterErrorHistograms,
)
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
def plot_error_histo_class(processor_class):
    plot_error_histo = PlotterErrorHistograms(processor_class)
    return plot_error_histo


def test_compute_model_error_and_sigma_prediction(plot_error_histo_class):
    target_date = 0
    water_error, ground_error, water_sigma, ground_sigma = (
        plot_error_histo_class._compute_model_error_and_sigma_prediction(
            target_date
        )
    )
    assert all(
        isinstance(x, np.ndarray)
        for x in (water_error, ground_error, water_sigma, ground_sigma)
    )


def test_plot_distrib(plot_error_histo_class):
    mean = 0.015
    var = 0.353
    vmin = -0.5
    vmax = 0.5
    color = "red"
    water = True
    fig, ax = plt.subplots()
    plot_error_histo_class._plot_distrib(mean, var, vmin, vmax, color, water)
    n_fig = len(plt.get_fignums())
    assert n_fig == 3
    assert len(ax.lines) == 1


def test_plot_error_histogram(plot_error_histo_class):
    data = np.array([0.2549719, 0.1048373, -0.56742895])
    sigma = np.array([0.35098687, 0.35098687, 0.35098687])
    title = "GT swot wse - model prediction"
    vmin = -0.5
    vmax = 0.5
    ax = plt.gca()
    plot_error_histo_class._plot_error_histogram(data, sigma, title, vmin, vmax)
    assert len(ax.lines) == 3


def test_plot_all_error_histograms(plot_error_histo_class, tmp_path):
    t_values = [0, 1]
    inference_pth = Path((tmp_path) / "output" / "inference")
    for img in Path(inference_pth).glob("error_histograms_time_*.png"):
        img.unlink()
    plot_error_histo_class.plot_all_error_histograms(t_values)
    assert any(Path(inference_pth).glob("error_histograms_time_*.png"))
