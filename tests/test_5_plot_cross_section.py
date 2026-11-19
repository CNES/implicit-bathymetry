#!/usr/bin/env python3

"""Plot cross section unit tests."""

from pathlib import Path
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.inference.plot_cross_section import PlotterCrossSection
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
def plot_cross_section_class(processor_class):
    plot_cross_section = PlotterCrossSection(processor_class)
    return plot_cross_section


def test_generate_cross_section_data(plot_cross_section_class):
    t_values = [0, 1]
    x = 0.5
    x0, hg, hw, out, sigma_g, sigma_w = (
        plot_cross_section_class._generate_cross_section_data(t_values, x, "x")
    )

    assert all(
        isinstance(x, np.ndarray) for x in (x0, hg, hw, out, sigma_g, sigma_w)
    )


def test_plot_cross_section(plot_cross_section_class):
    x0 = [
        0.0,
        0.02702703,
        0.05405405,
        0.08108108,
        0.10810811,
        0.13513514,
        0.16216216,
        0.18918919,
        0.21621622,
        0.24324324,
        0.27027027,
        0.2972973,
        0.32432432,
        0.35135135,
        0.37837838,
        0.40540541,
        0.43243243,
        0.45945946,
        0.48648649,
        0.51351351,
        0.54054054,
        0.56756757,
        0.59459459,
        0.62162162,
        0.64864865,
        0.67567568,
        0.7027027,
        0.72972973,
        0.75675676,
        0.78378378,
        0.81081081,
        0.83783784,
        0.86486486,
        0.89189189,
        0.91891892,
        0.94594595,
        0.97297297,
        1.0,
    ]
    hg = np.zeros((2, 38))
    hw = np.zeros((2, 38))
    out = np.zeros((2, 38))
    sigma_g = np.zeros((2, 38))
    sigma_w = np.zeros((2, 38))
    dim = "x"
    x = 0.2
    fig, ax = plt.subplots()
    plot_cross_section_class._plot_cross_section(
        ax, x0, x, dim, hg, hw, out, sigma_g, sigma_w
    )
    n_fig = len(plt.get_fignums())
    assert n_fig == 2
    assert len(ax.lines) == 7


def test_plot_many_cross_sections(plot_cross_section_class, tmp_path):
    t_values = [0, 1]
    plot_cross_section_class.plot_many_cross_sections(t_values, dim="x")
    inference_pth = Path((tmp_path) / "output" / "inference")
    assert any(
        Path(inference_pth).glob("plot_cross_section_width_0_02_x_time_0.000_*")
    )
    assert any(
        Path(inference_pth).glob("plot_cross_section_width_0_02_x_time_1.000_*")
    )
