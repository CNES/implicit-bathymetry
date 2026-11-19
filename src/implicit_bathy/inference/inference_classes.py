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
Inference classes
"""

import logging
import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from rasterio.features import rasterize
from shapely.geometry import box

from implicit_bathy.train.model import (
    BayesianTwoBranchModel,
    StandardTwoBranchModel,
)

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.getenv("CURL_CA_BUNDLE") is not None:
    os.environ["CURL_CA_BUNDLE"] = os.getenv("CURL_CA_BUNDLE")

logging.basicConfig(level=logging.INFO)


class SimulateInputs:
    """
    Class for simulate inputs
    """

    def __init__(self):
        logging.info("Simulating inputs")

    def simulate_input_for_3D_grid(
        self, plot_density_x, plot_density_y, chosen_time
    ):
        """
        simulate input for 3D grid
        Args:
            plot_density_x (int): pixel density in the x direction
            plot_density_y (int): pixel density in the y direction
            chosen_time (float): DSM time
        """

        # generate N (=128 by default) dots between 0 and 1 for x and for y
        x = np.linspace(0, 1, plot_density_x)
        y = np.linspace(0, 1, plot_density_y)
        x1, y1 = np.meshgrid(x, y)
        generated_x = x1.flatten()
        generated_y = y1.flatten()

        # simulate time
        generated_t = np.ones(plot_density_x * plot_density_y) * chosen_time

        generated_input = (
            torch.Tensor(np.array([generated_x, generated_y, generated_t]))
            .transpose(0, 1)
            .to(device)
        )

        return generated_input, x1, y1

    def simulate_input_for_2D_cut(
        self, t_values, x, plot_density_x, plot_density_y, dim
    ):
        """
        simulate input for 2D grid
        Args:
            t_values (list) : all swot times
            plot_density_x (int): pixel density in the x direction
            plot_density_y (int): pixel density in the y direction
            dim (str): cross-section along th x or y axis
        """

        if dim == "x":
            # generate N dots between 0 and 1 for one spatial dimension
            x0 = np.linspace(0, 1, plot_density_x)

            # the other spatial dimension is fixed, eg 0.5
            fixed_x = torch.tensor([x], dtype=torch.float32)
            generated_x = fixed_x.expand(
                len(t_values), plot_density_x
            ).flatten()

            # expand for time dimension
            generated_y = (
                torch.tensor(x0, dtype=torch.float32)
                .expand(len(t_values), plot_density_x)
                .flatten()
            )

            generated_t = (
                torch.tensor(t_values, dtype=torch.float32)[None, :]
                .T.expand(len(t_values), plot_density_x)
                .flatten()
            )

            generated_input = (
                torch.stack([generated_x, generated_y, generated_t])
                .transpose(0, 1)
                .to(device)
            )

        elif dim == "y":
            # generate N dots between 0 and 1 for one spatial dimension
            x0 = np.linspace(0, 1, plot_density_y)
            # the other spatial dimension is fixed, eg 0.5
            fixed_x = torch.tensor([x], dtype=torch.float32)

            # expand for time dimension
            generated_y = (
                torch.tensor(x0, dtype=torch.float32)
                .expand(len(t_values), plot_density_y)
                .flatten()
            )
            generated_x = fixed_x.expand(
                len(t_values), plot_density_y
            ).flatten()
            generated_t = (
                torch.tensor(t_values, dtype=torch.float32)[None, :]
                .T.expand(len(t_values), plot_density_y)
                .flatten()
            )

            generated_input = (
                torch.stack([generated_y, generated_x, generated_t])
                .transpose(0, 1)
                .to(device)
            )
        else:
            logging.info("Dim error")

        return x0, generated_input


class Processor:
    """
    Class for proccess model and compute data
    """

    def __init__(self, plot_params, training_params, dataset, simulation):
        self.simulation = simulation
        self.plot_params = plot_params
        self.training_params = training_params
        out = Path(self.plot_params.save_path_for_plots).parent
        self.path_report = Path(out) / "report.csv"
        self.dataset = dataset

    def load_model(self):
        """
        Loading model
        """
        if self.training_params.bayesian_model:
            print(self.training_params)
            self.model = BayesianTwoBranchModel(self.training_params)
        else:
            self.model = StandardTwoBranchModel(self.training_params)

        logging.info("Load model %s", self.model.description)
        logging.info("Model set to %s", device)
        self.model.to(device)
        self.model.ground_net.to(device)
        self.model.water_net.to(device)

        model_save_dict = torch.load(
            self.plot_params.model_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(model_save_dict["model_state_dict"])
        self.model.water_net.weight = model_save_dict["water_embedding"]

        self.model.eval()

    def make_prediction(self, input):
        """
        Args:
            input (torch.Tensor) : simulation 3D grid
        """
        input_device = input.to(device)
        with torch.no_grad():
            out_dict = self.model(input_device, self.training_params)
        return out_dict

    def format_model_outputs(
        self,
        out_dict,
        key,
        dim1,
        dim2,
    ):
        """
        Args:
            out_dict (dict) : simulation 3D grid
            key (str) : key to access required data ("h", "hg")
            dim1 (int) : x dimension
            dim2 (int) : y dimension
        """
        output = out_dict[key].cpu().detach().numpy()
        output = np.reshape(
            output,
            (dim1, dim2),
        )
        output = output[::-1, :]

        if key in ("h", "hg"):
            # for standard model : we can plot the z predicted
            # between 0 and 1 -> need to multiply it back to meter values
            generated_z = (
                output * (self.dataset.max_wse - self.dataset.min_wse)
                + self.dataset.min_wse
            )
            output = np.maximum(
                generated_z,
                np.ones_like(generated_z) * self.dataset.min_wse,
            )

        return output

    def _load_vt_vh(self):
        """
        Load vt from csv file
        """
        if vt_path := self.plot_params.vt_path:
            df = pd.read_csv(vt_path, sep=",", header=None)
            t = df[0].to_numpy()[:-1]

            start_index = np.where(t == self.plot_params.start_time)[0][0]
            end_index = np.where(t == self.plot_params.end_time)[0][0] + 1
            h = df[1].to_numpy()[:-1]
            s = df[2].to_numpy()[:-1]
            v = df[3].to_numpy()[:-1]

            t_normalized = np.linspace(0.0, 1.0, end_index - start_index)
            t = t[start_index:end_index]

            h = h[start_index:end_index]
            h = h.astype(np.float32)

            s = s[start_index:end_index]
            s = s.astype(np.float32)

            v = v[start_index:end_index]
            v = v.astype(np.float32)
            v_normalized = v - v[~np.isnan(v)][0]

            return {
                "t_normalized": t_normalized,
                "t": t,
                "wse_vt": h,
                "delta_vol_vt": v_normalized * 1e-6,
                "vol_vt": v * 1e-6,
                "area_vt": s * 1e-6,
            }

    def _compute_roi_bounding_box_surface(self):
        """
        Compute ROI bounding box
        """
        contour_gdf = gpd.read_file(self.plot_params.contour_path)
        contour_gdf = contour_gdf.to_crs(self.plot_params.convert_to_crs)
        minx, miny, maxx, maxy = contour_gdf.total_bounds
        bbox = box(minx, miny, maxx, maxy)
        area = bbox.area
        logging.info("bounding box area is %s", area)
        return area

    def compute_predicted_surface(self, chosen_time, roi_area):
        """
        Compute ROI bounding box

        Args:
            chosen_time (float): DSM time
            roi_area (float): area ROI
        """
        params = self.plot_params
        s2_tif_path = params.s2_tif_path
        if Path(s2_tif_path).exists():
            with rio.open(s2_tif_path) as src:
                raster_contour = self.rasterize_contour(
                    src.height, src.width, src.transform, src.crs
                )
                # Generate input data that covers the whole range of x and y for a chosen time
                generated_input, _, _ = (
                    self.simulation.simulate_input_for_3D_grid(
                        src.width, src.height, chosen_time
                    )
                )
                out_dict = self.make_prediction(generated_input)
                if self.plot_params.surface_inference == "mean":
                    # Get predictions from the model
                    max_label = self.format_model_outputs(
                        out_dict, "max_label", src.height, src.width
                    )
                    max_label = (1 - max_label).astype(int)
                    max_label[raster_contour == 0] = 0
                    water_surface = np.mean(max_label) * roi_area * 1e-6
                    return water_surface
                if self.plot_params.surface_inference == "bayesian":
                    p_w = self.format_model_outputs(
                        out_dict, "pw", src.height, src.width
                    )
                    p_w_mask = (p_w > self.plot_params.p_w_filter).astype(int)
                    p_w_mask[raster_contour == 0] = 0
                    water_surface = np.mean(p_w_mask) * roi_area * 1e-6
                    return water_surface
        else:
            raise FileNotFoundError(
                f"No TIFF file found ({s2_tif_path}): unable to calculate the area."
            )

    def rasterize_contour(self, width, height, transform, crs):
        """
        Rasterize ROI
        Args:
            width (int): x dimension
            height (int): y dimension
            transform (Affine) : Affine transform
            crs (CRS) : coordinate reference system
        """
        contour_gdf = gpd.read_file(self.plot_params.contour_path)
        contour_gdf = contour_gdf.to_crs(crs)
        shp_contour = ((geom, 1) for geom in contour_gdf.geometry)
        raster_contour = rasterize(
            shapes=shp_contour,
            out_shape=(width, height),
            transform=transform,
            fill=0,
            dtype="float",
        )
        return raster_contour

    def mean_absolute_rel_error(self, vt_data, pred_data):
        """
        Compute different indicators :
        - MARE: mean absolute relative error
        - MRE: mean relative error
        - MAE: mean absolute error
        - ME: mean error
        Args:
            vt_data (series): vt data (height, area, volume)
            pred_data (series): pred data (height, area, volume)
        """
        vt_data = np.array(vt_data, dtype=float)
        pred_data = np.array(pred_data, dtype=float)
        mask = ~np.isnan(vt_data) & ~np.isnan(pred_data) & (vt_data != 0)
        if not vt_data[mask].size == 0:
            mare = np.mean(
                np.abs((vt_data[mask] - pred_data[mask]) / vt_data[mask])
            )
            mre = np.mean((vt_data[mask] - pred_data[mask]) / vt_data[mask])
            mae = np.mean(np.abs(vt_data[mask] - pred_data[mask]))
            me = np.mean(pred_data[mask] - vt_data[mask])
            return mare.round(4), mae.round(4), mre.round(4), me.round(4)
        return (np.nan,) * 4

    def mean_error_delta_vol(self, vt_data, pred_data, vol_max):
        """
        Compute different indicators :
        ME: mean error
        MAE: mean absolute error
        MAERN : mean Abs Error Range Normalized
        MAETV: mean Abs Error TotalVol Normalized
        Args:
            vt_data (series): vt data (height, area, volume)
            pred_data (series): pred data (height, area, volume)
        """
        vt_data = np.array(vt_data, dtype=float)
        pred_data = np.array(pred_data, dtype=float)
        mask = ~np.isnan(vt_data) & ~np.isnan(pred_data) & (vt_data != 0)
        if not vt_data[mask].size == 0:
            mae = np.mean(np.abs(vt_data[mask] - pred_data[mask]))
            maern = np.mean(np.abs(vt_data[mask] - pred_data[mask])) / (
                vt_data[mask].max() - vt_data[mask].min()
            )
            maetv = np.mean(np.abs(vt_data[mask] - pred_data[mask])) / vol_max
            return mae.round(4), maern.round(4), maetv.round(4)
        return (np.nan,) * 3

    def quantile_68(self, vt_data, pred_data):
        """
        Compute different indicators :
        Quantile absolute relative error
        Quantile absolute error
        Args:
            vt_data (series): vt data (height, area, volume)
            pred_data (series): pred data (height, area, volume)
        """
        vt_data = np.array(vt_data, dtype=float)
        pred_data = np.array(pred_data, dtype=float)
        mask = ~np.isnan(vt_data) & ~np.isnan(pred_data) & (vt_data != 0)
        abs_rel_errors = np.abs(
            (vt_data[mask] - pred_data[mask]) / vt_data[mask]
        )
        abs_errors = abs(vt_data[mask] - pred_data[mask])
        if not vt_data[mask].size == 0:
            quantile_abs_rel_errors = pd.Series(abs_rel_errors).quantile(0.68)
            quantile_abs_errors = pd.Series(abs_errors).quantile(0.68)
            return quantile_abs_rel_errors.round(4), quantile_abs_errors.round(
                4
            )
        return (np.nan,) * 2

    def insert_indicators_report(self, list_stat):
        """
        Insert indicators in report
        Args:
            list_stat (list) : name indicators
        """
        if Path(self.path_report).exists():
            report = pd.DataFrame(list_stat.items())
            add_in_report = pd.read_csv(
                self.path_report, index_col=0, header=None
            )
            add_in_report.columns = ["indicators", "value"]
            report.columns = ["indicators", "value"]
            add_in_report = pd.concat(
                [add_in_report, report], ignore_index=True
            )
            add_in_report["value"] = (
                add_in_report["value"].astype(str).str.strip()
            )
            add_in_report = add_in_report.drop_duplicates(
                subset=["indicators"], keep="last"
            ).reset_index(drop=True)
            add_in_report.to_csv(self.path_report, index=True, header=False)


class Plotter:
    """
    Class for plotting data
    """

    def __init__(self, processor):
        self.processor = processor
        self.simulation = self.processor.simulation
        self.dataset = self.processor.dataset

        # loading model
        self.processor.load_model()

        if self.processor.plot_params.dsm_path is not None:
            self.gt_dsm, self.gt_transform = self.retrieve_gt_dsm()
            self.extend_mask = self.processor.rasterize_contour(
                self.gt_dsm.shape[0],
                self.gt_dsm.shape[1],
                self.transform,
                self.dsm_crs,
            )
            self.extend_mask[self.extend_mask == 0] = np.nan

            # the resolution of the DSM will impose
            # the resolution of the plots -> plot_params.plot_density
            # that could also be the S2 image
            # TO BE DEFINED
            self.plot_density_x = self.gt_dsm.shape[0]  # 298
            self.plot_density_y = self.gt_dsm.shape[1]  # 936
            self.mask_nan_values = ~np.isnan(
                self.gt_dsm
            )  # 1 if real number, 0 if nan
            self.mask_nan_values = self.mask_nan_values * (self.gt_dsm > 0)
            self.mask_nan_values = self.mask_nan_values.astype("float")
            self.mask_nan_values[self.mask_nan_values == 0] = (
                np.nan
            )  # 1 if real number, nan if nan
            # reverse
        else:
            self.plot_density_x = self.processor.plot_params.plot_density
            self.plot_density_y = self.processor.plot_params.plot_density
            self.mask_nan_values = np.ones(
                (self.plot_density_x, self.plot_density_y)
            )
            self.extend_mask = self.mask_nan_values

    def retrieve_gt_dsm(self):
        """
        Retrieve ground truth DSM
        """
        dsm_dataset = rio.open(self.processor.plot_params.dsm_path)
        self.transform = dsm_dataset.transform
        self.dsm_crs = dsm_dataset.crs
        # simulate input with shape of the DSM
        data_array = dsm_dataset.read()[0]
        return data_array, dsm_dataset.transform
