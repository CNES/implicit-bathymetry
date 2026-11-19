#!/usr/bin/env python3
# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the implicit_bathy project:
#  https://.../implicit_bathy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plot water probability class
"""

import logging
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio as rio
from rasterio import features, warp, windows
from rasterio.features import shapes
from rasterio.warp import Resampling
from scipy import ndimage
from scipy.ndimage import zoom
from shapely.geometry import shape
from skimage import feature, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import square

from implicit_bathy.inference.inference_classes import Plotter

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
logging.basicConfig(level=logging.INFO)


class ImageSatellite:
    """
    Image Satellite S1/S2 class
    """

    def __init__(
        self, item, date_nc_min, date_nc_max, sensor_type, gdf, processor
    ):
        """
        item : STAC Item (Planetary Computer)
        date_nc_min: minimum swot date
        date_nc_min: maximum swot date
        sensor_type : S1/S2
        gdf: contour ROI
        processor : processor class
        simulation: simulation class
        """
        self.item = item
        self.date = item.datetime.date()
        self.geometry = item.geometry
        self.sensor_type = sensor_type
        self.gdf = gdf
        self.bands = {
            name
            for name, _ in item.assets.items()
            if name.startswith("B")
            or name.startswith("v")
            or name.startswith("SCL")
        }
        self.date_normalized = (self.date - date_nc_min) / (
            date_nc_max - date_nc_min
        )
        self.processor = processor
        self.aoi_bounds = features.bounds(self.gdf)
        self.simulation = self.processor.simulation

    def get_band(self, name):
        """
        Get band from s1/s1 image
        Args :
            name (str) : band's name
        """
        if name not in self.bands:
            raise ValueError(f"Band {name} not available")
        return self.item.assets[name].href

    def crop_by_roi(self, band):
        """
        Crop band with ROI
        Args :
            band (str) : band's url
        """
        self.aoi_bounds = features.bounds(self.gdf)
        with rio.open(band) as ds:
            self.crs = ds.crs
            self.dtype = ds.dtypes
            self.img_transform = ds.transform
            warped_aoi_bounds = warp.transform_bounds(
                "epsg:4326", self.crs, *self.aoi_bounds
            )
            aoi_window = windows.from_bounds(
                transform=ds.transform, *warped_aoi_bounds
            )
            img_sent = ds.read(window=aoi_window)
            if (
                self.sensor_type == "S2" and band != self.get_band("SCL")
            ) or self.sensor_type == "S1":
                self.new_transform = rio.windows.transform(
                    aoi_window, ds.transform
                )
                self.height = img_sent.shape[1]
                self.width = img_sent.shape[2]
        return img_sent

    def get_image_s2_to_plot(self):
        """
        Get image 3 bands (8,3,2)
        """
        image_to_plot = []
        list_band = ["B08", "B03", "B02"]
        for b in list_band:
            data_b = self.crop_by_roi(self.get_band(b))
            image_to_plot.append(data_b)
        image_to_plot = np.squeeze(image_to_plot, axis=1)
        image_to_plot = np.transpose(image_to_plot, (1, 2, 0))
        return image_to_plot

    def get_image_s1_to_plot(self):
        """
        Get image 3 bands (vv,vh,vv)
        """
        image_to_plot = []
        list_band = ["vv", "vh", "vv"]
        for b in list_band:
            data_b = self.crop_by_roi(self.get_band(b))
            image_to_plot.append(data_b)
        image_to_plot = np.squeeze(image_to_plot, axis=1)
        image_to_plot = np.transpose(image_to_plot, (1, 2, 0))
        return image_to_plot

    def get_mndwi(self):
        """
        Get mndwi
        """
        mndwi = []
        data_b = self.crop_by_roi(self.get_band("B03"))
        mndwi.append(data_b)
        with rio.open(self.get_band("B11")) as dst:
            warped_aoi_bounds = warp.transform_bounds(
                "epsg:4326", self.crs, *self.aoi_bounds
            )
            aoi_window = windows.from_bounds(
                transform=dst.transform, *warped_aoi_bounds
            )
            data = dst.read(
                1,
                window=aoi_window,
                out_shape=(self.height, self.width),  # corresponds to B03/B08
                resampling=Resampling.bilinear,
            )
            mndwi.append(data)
        mndwi = (mndwi[0].astype(float) - mndwi[1].astype(float)) / (
            mndwi[0] + mndwi[1]
        ).astype(float)
        mndwi = mndwi.squeeze()
        return mndwi

    def get_adaptative_th_canny(self, im):
        """
        Get adaptative threshold canny
        Args :
            im (array) : ndwi image
        """
        canny_low_th = 0.98
        canny_high_th = 0.999
        cdt = True
        previous_border_pixels = 0
        corr = 0
        max_correction = 0.05
        im = im.squeeze()
        im[im < -1] = -1
        im[im > 1] = 1
        im[morphology.binary_dilation(im == 0, footprint=square(4))] = 0
        while cdt:
            edges = feature.canny(
                im,
                sigma=2,
                use_quantiles=True,
                low_threshold=canny_low_th - corr,
                high_threshold=(canny_high_th - corr),
            )
            n_edge = np.count_nonzero(edges)
            if (n_edge != 0) and (previous_border_pixels == n_edge):
                cdt = False
            elif corr > 0.05:
                cdt = False
            else:
                cdt = n_edge < 5000
                previous_border_pixels = n_edge
            corr = corr + 0.003

        dilated_edges = morphology.binary_dilation(edges, footprint=square(3))
        if ((dilated_edges != 0).sum() > 0) and (corr < max_correction):
            threshold = threshold_otsu(im[dilated_edges == 1])
            logging.info("OtsuThreshold: %s", threshold)
        else:
            threshold = 0
        mndwi = (im > threshold).astype(np.uint8)
        return mndwi

    def get_ndwi(self, image_to_plot):
        """
        Get ndwi
        Args :
            image_to_plot (array) : S2 image
        """
        return (image_to_plot[:, :, 1] - image_to_plot[:, :, 0]) / (
            image_to_plot[:, :, 1] + image_to_plot[:, :, 0]
        )

    def compute_surface_s2(self, ndwi):
        """
        Compute sentinel 2 surface
        Args :
            ndwi (array) : NDWI image
        """
        raster_contour = self.get_raster_contour()
        data = self.crop_by_roi(self.get_band("SCL"))
        if not np.isin(data, [3, 8, 9, 10]).any():
            # 3: cloud shadows, 8: cloud medium pb, 9: cloud hight pb, 10: thin cirrus
            ndwi_image = (ndwi > 0).astype(np.uint8)
            ndwi_image[raster_contour == 0] = False
            water_surface = np.sum(ndwi_image) * 10**2 * 10 ** (-6)
            return water_surface
        return None

    def compute_surface_s1(self):
        """
        Compute sentinel 1 surface
        """
        data = self.crop_by_roi(self.get_band("vv"))
        image_vv = ((data < 0.03).astype(np.uint8)).squeeze(0)
        raster_contour = self.get_raster_contour()
        image_vv[raster_contour == 0] = False
        water_surface = np.sum(image_vv) * 10**2 * 10 ** (-6)
        return water_surface

    def _grad_label(self, im):
        """
        Get gradient
        Args :
            im (array)
        """
        return ndimage.sobel(im, axis=0, mode="constant") + ndimage.sobel(
            im, axis=1, mode="constant"
        )

    def _remove_ticks(self, ax):
        """
        Remove tickes
        Args :
            ax (ax) : axis
        """
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

    def get_raster_contour(self):
        """
        Rasterize contour
        """
        raster_contour = self.processor.rasterize_contour(
            self.height, self.width, self.new_transform, self.crs
        )
        return raster_contour

    def zoom_raster(self, image_to_plot):
        """
        Zoom raster for printing
        Args :
            image_to_plot (array) : S2 image
        """
        if (image_to_plot.shape[0] < 500) or (image_to_plot.shape[1] < 500):
            height_image, width_image, _ = image_to_plot.shape
            target_max = 500
            factor = target_max / max(height_image, width_image)
            zoom_factors = (factor, factor, 1)
            image_to_plot = zoom(image_to_plot, zoom=zoom_factors, order=0)
            zoom_factors = (factor, factor)
            self.raster_contour = zoom(
                self.raster_contour, zoom=zoom_factors, order=0
            )
            return image_to_plot, self.raster_contour
        return image_to_plot, self.raster_contour

    def save_gpkg(self, image_to_plot, raster_contour):
        """
        Save GPKG surface S2
        Args :
            image_to_plot (array) : S2 image
            raster_contour (array) : ROI raster contour
        """
        generated_input, _, _ = self.simulation.simulate_input_for_3D_grid(
            image_to_plot.shape[1], image_to_plot.shape[0], self.date_normalized
        )
        out_dict = self.processor.make_prediction(generated_input)

        if self.processor.plot_params.surface_inference == "bayesian":
            # SAVE GPKG
            p_w = self.processor.format_model_outputs(
                out_dict, "pw", image_to_plot.shape[0], image_to_plot.shape[1]
            )
            p = p_w > self.processor.plot_params.p_w_filter
            p[raster_contour == 0] = False
            mask_uint8 = p.astype("uint8")
            geoms = [
                shape(geom)
                for geom, value in shapes(
                    mask_uint8, transform=self.new_transform
                )
                if value == 1
            ]
            gdf = gpd.GeoDataFrame(geometry=geoms, crs=self.crs)
            gdf = gdf.to_crs("EPSG:4326")
            gdf.to_file(
                self.processor.plot_params.save_path_for_plots
                / f"water_area_{self.date}.gpkg",
                driver="GPKG",
            )

        if self.processor.plot_params.volume_inference == "mean":
            # SAVE GPKG
            max_label = self.processor.format_model_outputs(
                out_dict,
                "max_label",
                image_to_plot.shape[0],
                image_to_plot.shape[1],
            )
            p = 1 - max_label
            p[self.raster_contour == 0] = False
            mask_uint8 = p.astype("uint8")
            geoms = [
                shape(geom)
                for geom, value in shapes(
                    mask_uint8, transform=self.new_transform
                )
                if value == 1
            ]
            gdf = gpd.GeoDataFrame(geometry=geoms, crs=self.crs)
            gdf = gdf.to_crs("EPSG:4326")
            gdf.to_file(
                self.processor.plot_params.save_path_for_plots
                / f"water_area_{self.date}.gpkg",
                driver="GPKG",
            )

    def plot_water_probability(self, image_to_plot):
        """
        Plot water probability
        Args :
            image_to_plot (array) : S2 image
        """
        self.raster_contour = self.get_raster_contour()
        self.save_gpkg(image_to_plot, self.raster_contour)
        image_to_plot, self.raster_contour = self.zoom_raster(image_to_plot)
        generated_input, _, _ = self.simulation.simulate_input_for_3D_grid(
            image_to_plot.shape[1], image_to_plot.shape[0], self.date_normalized
        )
        out_dict = self.processor.make_prediction(generated_input)
        p_w = self.processor.format_model_outputs(
            out_dict, "pw", image_to_plot.shape[0], image_to_plot.shape[1]
        )
        if self.processor.plot_params.surface_inference == "bayesian":
            contour_mid = self._grad_label(
                p_w > self.processor.plot_params.p_w_filter
            )
            contour_mid[self.raster_contour == 0] = False
            t = f"Contour lines t=t_value \n Pw > {self.processor.plot_params.p_w_filter}"

        if self.processor.plot_params.surface_inference == "mean":
            max_label = self.processor.format_model_outputs(
                out_dict,
                "max_label",
                image_to_plot.shape[0],
                image_to_plot.shape[1],
            )
            max_label = (1 - max_label).astype(int)
            contour_mid = self._grad_label(max_label == 1)
            contour_mid[self.raster_contour == 0] = False
            t = "Contour lines t=t_value \n Hw > Hg"

        if np.nanmax(image_to_plot) - np.nanmin(image_to_plot) != 0:
            image_to_plot = (image_to_plot - np.nanmin(image_to_plot)) / (
                np.nanmax(image_to_plot) - np.nanmin(image_to_plot)
            )
        if self.sensor_type == "S1":
            image_to_plot = image_to_plot[:, :, 1:2]
            p2, p98 = np.percentile(image_to_plot, (2, 98))
            image_to_plot = np.clip((image_to_plot - p2) / (p98 - p2), 0, 1)

        overlay = np.clip(
            contour_mid[:, :, None] * np.array([0.0, 1.0, 0.0]) + image_to_plot,
            0,
            1,
        )
        d_norm = round(self.date_normalized, 2)
        if (image_to_plot.shape[0] < 500) or (image_to_plot.shape[1] < 500):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(overlay, interpolation="none")
            ax.set_title(t)
            self._remove_ticks(ax)

            ax2 = fig.add_subplot(1, 2, 2)
            p_w = np.where(self.raster_contour == 0, np.nan, p_w)
            im2 = ax2.imshow(p_w, vmin=0, vmax=1)
            self._remove_ticks(ax2)

            ax2.set_title("Probability of water")

            plt.colorbar(im2, fraction=0.047)
            plt.savefig(
                self.processor.plot_params.save_path_for_plots
                / f"plot_water_proba_{d_norm}_{self.date}_{self.sensor_type}.png"
            )
            plt.close()
        else:
            plt.figure(figsize=(17, 14))
            plt.imshow(overlay, interpolation="none")
            plt.title(t)
            plt.savefig(
                self.processor.plot_params.save_path_for_plots
                / f"plot_water_proba_{d_norm}_{self.date}_{self.sensor_type}.png"
            )
            plt.close()

    def save_tif(self, image_to_plot):
        """
        Save S1/S2 tif
        Args :
            image_to_plot (array) : S2 image
        """
        image_to_plot = np.transpose(image_to_plot, (2, 0, 1))
        with rio.open(
            self.processor.plot_params.save_path_for_plots
            / f"image_{self.sensor_type}_{self.date}.tif",
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=3,
            dtype=rio.float32,
            crs=self.crs,
            transform=self.new_transform,
        ) as dst:
            dst.write(image_to_plot)


class PlotterWaterProbaOnS2(Plotter):
    """
    Plot water probability on sentinel 2 class
    """

    def __init__(self, processor):
        super().__init__(processor)
        self.date_nc_min = datetime.strptime(
            self.processor.plot_params.start_time, "%d-%m-%Y"
        ).date()
        self.date_nc_max = datetime.strptime(
            self.processor.plot_params.end_time, "%d-%m-%Y"
        ).date()
        self.gdf = gpd.read_file(
            self.processor.plot_params.contour_path
        ).geometry.iloc[0]

    def plot_water_proba_on_s2_images(self):
        """
        Plot water probability on sentinel 2 image
        """
        roi_area = self.processor._compute_roi_bounding_box_surface()
        (
            s1_graph,
            s2_graph,
            s2_graph_canny,
            predicted_graph_s1,
            predicted_graph_s2,
            all_dict,
        ) = ({}, {}, {}, {}, {}, {})
        catalog = pystac_client.Client.open(
            PC_STAC, modifier=planetary_computer.sign_inplace
        )
        time_of_interest = f"{self.date_nc_min}/{self.date_nc_max}"

        collections = ["sentinel-2-l2a", "sentinel-2-l2b"]
        search_s2 = []

        for collection in collections:
            search = catalog.search(
                collections=[collection],
                intersects=self.gdf,
                datetime=time_of_interest,
                query={"eo:cloud_cover": {"lt": 30}},
            )
            search_s2.extend(search.items())

        if search_s2:
            for it in search_s2:
                img = ImageSatellite(
                    it,
                    self.date_nc_min,
                    self.date_nc_max,
                    "S2",
                    self.gdf,
                    self.processor,
                )
                if shape(img.geometry).contains(self.gdf):
                    image_to_plot = img.get_image_s2_to_plot()
                    img.plot_water_probability(image_to_plot)
                    mndwi = img.get_mndwi()
                    surface_image_canny = img.get_adaptative_th_canny(mndwi)
                    surface_image = img.get_ndwi(image_to_plot)
                    surface = img.compute_surface_s2(surface_image)
                    surface_canny = img.compute_surface_s2(surface_image_canny)
                    # if surface / surface canny is None -> Cloud
                    if surface or surface_canny:
                        s2_graph[img.date] = surface
                        s2_graph_canny[img.date] = surface_canny
                        predicted_graph_s2[img.date] = (
                            self.processor.compute_predicted_surface(
                                img.date_normalized, roi_area
                            )
                        )
                    if self.processor.plot_params.save_s2_tif:  # save_tif
                        img.save_tif(image_to_plot)
            dict_to_add = [
                ("s2_graph", s2_graph),
                ("s2_graph_canny", s2_graph_canny),
                ("predicted_graph_s2", predicted_graph_s2),
            ]
            for name, d in dict_to_add:
                all_dict[name] = d

        search_s1 = catalog.search(
            collections=["sentinel-1-rtc"],
            intersects=self.gdf,
            datetime=time_of_interest,
        )
        if items_s1 := search_s1.item_collection():
            for it in items_s1:
                img = ImageSatellite(
                    it,
                    self.date_nc_min,
                    self.date_nc_max,
                    "S1",
                    self.gdf,
                    self.processor,
                )
                if shape(img.geometry).contains(self.gdf):
                    image_to_plot = img.get_image_s1_to_plot()
                    if (
                        self.processor.plot_params.plot_p_w_on_s1_images
                    ):  # plot s1 images
                        img.plot_water_probability(image_to_plot)
                    if self.processor.plot_params.save_s1_tif:  # save_tif
                        img.save_tif(image_to_plot)
                    surface = img.compute_surface_s1()
                    s1_graph[img.date] = surface
                    predicted_graph_s1[img.date] = (
                        self.processor.compute_predicted_surface(
                            img.date_normalized, roi_area
                        )
                    )
            dict_to_add = [
                ("s1_graph", s1_graph),
                ("predicted_graph_s1", predicted_graph_s1),
            ]
            for name, d in dict_to_add:
                all_dict[name] = d
        all_dict = {
            k: [float(v[key]) for key in sorted(v.keys())]
            for k, v in all_dict.items()
        }
        _ = plt.figure(figsize=(10, 7))
        plt.plot(
            s1_graph.keys(),
            s1_graph.values(),
            label="S1 surface",
            marker="o",
            color="tab:blue",
        )
        # plt.plot(s2_graph.keys(),
        # s2_graph.values(),
        # label="S2 surface ndwi seuil > 0", marker="o", color='tab:red')
        plt.plot(
            s2_graph_canny.keys(),
            s2_graph_canny.values(),
            label="S2 surface canny",
            marker="o",
            color="tab:pink",
        )
        plt.plot(
            predicted_graph_s1.keys(),
            predicted_graph_s1.values(),
            label="Predicted surface",
            marker="o",
            color="tab:orange",
        )

        mare_s2, _, _, _ = self.processor.mean_absolute_rel_error(
            all_dict["s2_graph_canny"], all_dict["predicted_graph_s2"]
        )
        mare_s1, _, _, _ = self.processor.mean_absolute_rel_error(
            all_dict["s1_graph"], all_dict["predicted_graph_s1"]
        )
        q68_abs_s2, _ = self.processor.quantile_68(
            all_dict["s2_graph_canny"], all_dict["predicted_graph_s2"]
        )
        q68_abs_s1, _ = self.processor.quantile_68(
            all_dict["s1_graph"], all_dict["predicted_graph_s1"]
        )
        results = {
            "mare_s2": mare_s2,
            "mare_s1": mare_s1,
            "quantile_abs_s2": q68_abs_s2,
            "quantile_abs_s1": q68_abs_s1,
        }
        self.processor.insert_indicators_report(results)
        self.vt_dict = self.processor._load_vt_vh()
        if self.vt_dict:
            vt_t = [
                datetime.strptime(d, "%d-%m-%Y").date()
                for d in self.vt_dict["t"]
            ]
            vt_s = self.vt_dict["area_vt"]
            plt.scatter(vt_t, vt_s, label="In situ", color="tab:green")

            vt_s2 = [
                v
                for d, v in zip(vt_t, vt_s)
                if d in sorted(list(s2_graph_canny))
            ]
            vt_s1 = [
                v for d, v in zip(vt_t, vt_s) if d in sorted(list(s1_graph))
            ]
            mare_s2_insitu, _, _, _ = self.processor.mean_absolute_rel_error(
                vt_s2, all_dict["s2_graph_canny"]
            )
            mare_s1_insitu, _, _, _ = self.processor.mean_absolute_rel_error(
                vt_s1, all_dict["s1_graph"]
            )
            q68_abs_s2_insitu, _ = self.processor.quantile_68(
                vt_s2, all_dict["s2_graph_canny"]
            )
            q68_abs_s1_insitu, _ = self.processor.quantile_68(
                vt_s1, all_dict["s1_graph"]
            )

            results = {
                "mare_s2_insitu": mare_s2_insitu,
                "mare_s1_insitu": mare_s1_insitu,
                "quantile_abs_s2_instu": q68_abs_s2_insitu,
                "quantile_abs_s1_instu": q68_abs_s1_insitu,
            }
            self.processor.insert_indicators_report(results)

        plt.xlabel("t normalized")
        plt.ylabel("Surface (km2)")
        plt.title("Predicted surface in time")
        plt.legend(fontsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "absolute_surface_water_s1_s2.png"
        )
