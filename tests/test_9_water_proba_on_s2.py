#!/usr/bin/env python3

"""Plot probability on s2 unit tests."""

from pathlib import Path
import pytest
import os
import numpy as np
import geopandas as gpd
import planetary_computer
import pystac_client
from datetime import date

from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.inference.plot_water_proba_on_s2 import (
    ImageSatellite,
    PlotterWaterProbaOnS2,
)
from implicit_bathy.inference.inference_classes import Processor, SimulateInputs
from implicit_bathy.config.config import BayesianTrainingConfig, InferenceConfig

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


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
def plot_proba_on_s2_class(processor_class):
    plot_proba_on_s2 = PlotterWaterProbaOnS2(processor_class)
    return plot_proba_on_s2


@pytest.fixture
def plot_image_satellite_s2(processor_class):
    catalog = pystac_client.Client.open(
        PC_STAC, modifier=planetary_computer.sign_inplace
    )
    datemin = date(2023, 7, 30)
    datemax = date(2023, 8, 3)
    gdf = gpd.read_file(processor_class.plot_params.contour_path).geometry.iloc[
        0
    ]
    time_of_interest = "2023-07-30/2023-08-30"
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=gdf,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 30}},
    )
    if len(search_s2.item_collection()) > 0:
        it = search_s2.item_collection()[0]

        imagesatellite = ImageSatellite(
            it, datemin, datemax, "S2", gdf, processor_class
        )
        return imagesatellite
    else:
        raise RuntimeError(
            "No Sentinel-2 image found for the specified area and time period."
        )


def test_get_band(plot_image_satellite_s2):
    band = plot_image_satellite_s2.get_band("B02")
    assert isinstance(band, str)


def test_crop_by_roi(plot_image_satellite_s2):
    band = plot_image_satellite_s2.get_band("B02")
    img_sent = plot_image_satellite_s2.crop_by_roi(band)
    assert isinstance(img_sent, np.ndarray)


def test_get_image_s2_to_plot(plot_image_satellite_s2):
    image_to_plot = plot_image_satellite_s2.get_image_s2_to_plot()
    assert isinstance(image_to_plot, np.ndarray)


def test_get_mndwi(plot_image_satellite_s2):
    mndwi = plot_image_satellite_s2.get_mndwi()
    assert isinstance(mndwi, np.ndarray)


def test_get_adaptative_th_canny(plot_image_satellite_s2):
    mndwi = plot_image_satellite_s2.get_mndwi()
    mndwi_canny = plot_image_satellite_s2.get_adaptative_th_canny(mndwi)
    assert isinstance(mndwi_canny, np.ndarray)


def test_get_ndwi(plot_image_satellite_s2):
    image_to_plot = plot_image_satellite_s2.get_image_s2_to_plot()
    assert isinstance(image_to_plot, np.ndarray)


def test_compute_surface_s2(plot_image_satellite_s2):
    mndwi = plot_image_satellite_s2.get_mndwi()
    water_surface = plot_image_satellite_s2.compute_surface_s2(mndwi)
    assert isinstance(water_surface, (float, type(None)))


def test_grad_label(plot_image_satellite_s2):
    mndwi = plot_image_satellite_s2.get_mndwi()
    im = plot_image_satellite_s2._grad_label(mndwi)
    assert isinstance(im, np.ndarray)


def test_get_raster_contour(plot_image_satellite_s2):
    band = plot_image_satellite_s2.get_band("B02")
    img_sent = plot_image_satellite_s2.crop_by_roi(band)
    raster_contour = plot_image_satellite_s2.get_raster_contour()
    assert isinstance(raster_contour, np.ndarray)


def test_save_gpkg(plot_image_satellite_s2, tmp_path):
    image_to_plot = plot_image_satellite_s2.get_image_s2_to_plot()
    raster_contour = plot_image_satellite_s2.get_raster_contour()
    inference_pth = Path((tmp_path) / "output" / "inference")
    for img in Path(inference_pth).glob("water_area_*.gpkg"):
        img.unlink()
    plot_image_satellite_s2.save_gpkg(image_to_plot, raster_contour)
    assert any(Path(inference_pth).glob("water_area_*.gpkg"))


def test_plot_water_probability(plot_image_satellite_s2, tmp_path):
    image_to_plot = plot_image_satellite_s2.get_image_s2_to_plot()
    inference_pth = Path((tmp_path) / "output" / "inference")
    for img in Path(inference_pth).glob("plot_water_proba_*.png"):
        img.unlink()
    plot_image_satellite_s2.plot_water_probability(image_to_plot)
    assert any(Path(inference_pth).glob("plot_water_proba_*.png"))


def test_save_tif(plot_image_satellite_s2, tmp_path):
    image_to_plot = plot_image_satellite_s2.get_image_s2_to_plot()
    inference_pth = Path((tmp_path) / "output" / "inference")
    for img in Path(inference_pth).glob("image_*.tif"):
        img.unlink()
    plot_image_satellite_s2.save_tif(image_to_plot)
    assert any(Path(inference_pth).glob("image_*.tif"))


def test_plot_water_proba_on_s2_images(plot_proba_on_s2_class, tmp_path):
    png_path = Path(
        (tmp_path) / "output" / "inference" / "absolute_surface_water_s1_s2.png"
    )
    png_path.unlink(missing_ok=True)
    plot_proba_on_s2_class.plot_water_proba_on_s2_images()
    assert png_path.exists()
