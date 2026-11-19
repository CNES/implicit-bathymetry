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
Plotter DSM class
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from numpy import ma
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Point

from implicit_bathy.inference.inference_classes import Plotter


class PlotterDSM(Plotter):
    """
    Plot DSM class
    """

    def write_dsm(self, dsm, name_tif):
        """
        Write DSM
        Args :
            dsm (array)
            name_tif (str)
        """
        epsg_dsm = str(self.dsm_crs).split(":")[1]
        out_tif = (
            self.processor.plot_params.save_path_for_plots
            / f"{name_tif}_{epsg_dsm}.tif"
        )
        with rio.open(
            out_tif,
            "w",
            driver="GTiff",
            height=dsm.shape[0],
            width=dsm.shape[1],
            count=1,
            dtype=rio.float32,
            crs=self.dsm_crs,
            transform=self.gt_transform,
        ) as dst:
            dst.write(dsm, 1)

        epsg = self.processor.plot_params.convert_to_crs.split(":")[1]
        out_tif_reproject = (
            self.processor.plot_params.save_path_for_plots
            / f"{name_tif}_{epsg}.tif"
        )
        self.reproject_tif(
            out_tif,
            out_tif_reproject,
            self.processor.plot_params.convert_to_crs,
        )

    def reproject_tif(self, in_tif, out_tif, dst_crs):
        """
        reproject DSM
        Args :
            in_tif (str) : path to the DSM
            out_tif (str) : path to the DSM reprojected
            dst_crs (str) : new CRS
        """
        with rio.open(in_tif) as src:
            data = src.read(1)
            temp_nodata = -9999
            data_temp = data.copy()
            data_temp[np.isnan(data_temp)] = temp_nodata
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": temp_nodata,
                }
            )

            dst_array = np.empty((height, width), dtype=data_temp.dtype)
            reproject(
                source=data_temp,
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=temp_nodata,
                dst_nodata=temp_nodata,
            )
            dst_array = dst_array.astype(float)
            dst_array[dst_array == temp_nodata] = np.nan
            with rio.open(out_tif, "w", **kwargs) as dst:
                dst.write(dst_array, 1)

    def compute_raster_learning_point(self):
        """
        Compute the raster of learning point
        """
        t_dates = (self.dataset.X)[:, 2].unique()
        params = self.processor.plot_params
        contour_gdf = gpd.read_file(params.contour_path)
        contour_gdf = contour_gdf.to_crs(params.convert_to_crs)

        min_x = contour_gdf.bounds["minx"].iloc[0]
        min_y = contour_gdf.bounds["miny"].iloc[0]
        max_x = contour_gdf.bounds["maxx"].iloc[0]
        max_y = contour_gdf.bounds["maxy"].iloc[0]

        raster = np.zeros(
            (self.plot_density_x, self.plot_density_y), dtype=np.uint8
        )
        for d in t_dates:
            filter = (self.dataset.X)[:, 2] == d
            data_filter = self.dataset.X[filter]
            x = data_filter[:, 0] * (max_x - min_x) + min_x
            y = data_filter[:, 1] * (max_y - min_y) + min_y
            geometry = [Point(xy) for xy in zip(x, y)]
            gdf = gpd.GeoDataFrame(geometry=geometry, crs=params.convert_to_crs)
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.drop_duplicates(subset="geometry")

            shapes = ((geom, 1) for geom in gdf.geometry)
            raster_for_date = rasterize(
                shapes=shapes,
                out_shape=(self.gt_dsm.shape[0], self.gt_dsm.shape[1]),
                transform=self.transform,
                fill=0,
                dtype="uint8",
            )
            raster = np.where((raster == 1) | (raster_for_date == 1), 1, 0)
        raster_learning_point = ma.masked_where(
            raster == 1, raster.astype(float)
        )
        return raster_learning_point

    def get_min_h_level(self):
        """
        Get minimum height during the period
        """
        t_dates = (self.dataset.X)[:, 2].unique()
        min_level_h = 1
        d = 0
        for d in t_dates.tolist():
            generated_input, _, _ = self.simulation.simulate_input_for_3D_grid(
                self.plot_density_y, self.plot_density_x, d
            )
            out_dict = self.processor.make_prediction(generated_input)
            predicted_h = self.processor.format_model_outputs(
                out_dict, "hw", self.plot_density_x, self.plot_density_y
            )
            if min_level_h > predicted_h[0][0]:
                min_level_h = predicted_h[0][0]
                d_min = d
        return min_level_h, d_min

    def plot_dsm_water_z(self):
        """
        Plot dsm water
        """
        params = self.processor.plot_params
        _, d_min = self.get_min_h_level()
        generated_input, _, _ = self.simulation.simulate_input_for_3D_grid(
            self.plot_density_y, self.plot_density_x, d_min
        )
        out_dict = self.processor.make_prediction(generated_input)

        predicted_max_label = self.processor.format_model_outputs(
            out_dict, "max_label", self.plot_density_x, self.plot_density_y
        )
        predicted_v = self.processor.format_model_outputs(
            out_dict, "vg", self.plot_density_x, self.plot_density_y
        )
        predicted_z = self.processor.format_model_outputs(
            out_dict, "hg", self.plot_density_x, self.plot_density_y
        )
        predicted_z = np.where((predicted_max_label == 0), np.nan, predicted_z)
        predicted_v = np.where((predicted_max_label == 0), np.nan, predicted_v)

        if params.dsm_path is not None:
            raster_contour = self.processor.rasterize_contour(
                self.gt_dsm.shape[0],
                self.gt_dsm.shape[1],
                self.transform,
                "EPSG:4326",
            )
            predicted_z_masked = np.where(
                (raster_contour == 0), np.nan, predicted_z
            )
            predicted_z_combined_cop30 = np.where(
                (raster_contour == 0), self.gt_dsm, predicted_z
            )
            predicted_v_masked = np.where(
                (raster_contour == 0), np.nan, predicted_v
            )
            self.write_dsm(predicted_z_masked, "predicted_dsm")
            self.write_dsm(predicted_v_masked, "predicted_std")
            raster_learning_point = self.compute_raster_learning_point()
            plt.figure(figsize=(2 * params.fig_size, params.fig_size))
            im0 = plt.imshow(
                self.gt_dsm * self.mask_nan_values,
                vmin=params.min_wse_for_dsm,
                vmax=params.max_wse_for_dsm,
                cmap="Spectral",
            )
            plt.title(
                f"GT COP30 DTM \n vmin={params.min_wse_for_dsm}, vmax={params.max_wse_for_dsm}"
            )
            plt.colorbar(im0)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(params.save_path_for_plots / "ground_truth_DSM.png")
            plt.close()

            plt.figure(figsize=(2 * params.fig_size, params.fig_size))
            im_cop30 = plt.imshow(
                predicted_z_combined_cop30 * self.mask_nan_values,
                vmin=params.min_wse_for_dsm,
                vmax=params.max_wse_for_dsm,
                cmap="Spectral",
            )
            plt.title(
                "Combinaison COP30 DTM / Predicted z \n"
                f"vmin={params.min_wse_for_dsm}, vmax={params.max_wse_for_dsm}"
            )
            plt.colorbar(im_cop30)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                params.save_path_for_plots / "combinaison_predicted_z_COP30.png"
            )
            plt.close()

        _, axs = plt.subplots(
            1, 2, figsize=(4 * params.fig_size, params.fig_size)
        )
        im0 = axs[0].imshow(
            predicted_z * self.mask_nan_values,
            vmin=params.min_wse_for_dsm,
            vmax=params.max_wse_for_dsm,
            cmap="Spectral",
        )
        if params.dsm_path is not None:
            _ = axs[0].imshow(
                raster_learning_point * self.mask_nan_values,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        axs[0].set_title(
            f"Predicted z \n vmin={params.min_wse_for_dsm}, vmax={params.max_wse_for_dsm}"
        )
        plt.colorbar(im0)

        im1 = axs[1].imshow(predicted_v * self.mask_nan_values, cmap="Spectral")
        if params.dsm_path is not None:
            _ = axs[1].imshow(
                raster_learning_point * self.mask_nan_values,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        axs[1].set_title(
            f"Predicted v \n min={np.nanmin(predicted_v):.2f} max={np.nanmax(predicted_v):.2f}"
        )
        plt.colorbar(im1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(params.save_path_for_plots / "predicted_DSM.png")
        plt.close()
