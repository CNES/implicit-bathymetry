#!/usr/bin/env python3

"""Plot DSM unit tests."""

from pathlib import Path
import pytest
import os
import numpy as np
from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.inference.plot_dsm import PlotterDSM
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
def plot_dsm_class(processor_class):
    plot_dsm = PlotterDSM(processor_class)
    return plot_dsm


def test_write_dsm(plot_dsm_class, tmp_path):
    dsm = np.random.rand(43, 100)
    plot_dsm_class.write_dsm(dsm, "predicted_dsm")
    inference_pth = Path((tmp_path) / "output" / "inference")
    tif_dsm = inference_pth / "predicted_dsm_4326.tif"
    tif_dsm_rep = inference_pth / "predicted_dsm_32631.tif"
    assert tif_dsm.exists()
    assert tif_dsm_rep.exists()


def test_compute_raster_learning_point(plot_dsm_class):
    raster_learning_point = plot_dsm_class.compute_raster_learning_point()
    assert isinstance(raster_learning_point, np.ma.MaskedArray)


def test_get_min_h_level(plot_dsm_class):
    min_level_h, d_min = plot_dsm_class.get_min_h_level()
    assert isinstance(min_level_h, np.float32)
    assert isinstance(d_min, float)


def test_plot_dsm_water_z(plot_dsm_class, tmp_path):
    inference_pth = Path((tmp_path) / "output" / "inference")
    png_gt = inference_pth / "ground_truth_DSM.png"
    png_combi = inference_pth / "combinaison_predicted_z_COP30.png"
    png_dsm = inference_pth / "predicted_DSM.png"
    paths = [png_gt, png_combi, png_dsm]
    for path in paths:
        path.unlink(missing_ok=True)
    plot_dsm_class.plot_dsm_water_z()
    assert png_gt.exists()
    assert png_combi.exists()
    assert png_dsm.exists()
