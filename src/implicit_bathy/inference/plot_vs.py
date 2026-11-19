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
Plot Height/Surface/Volume class
"""

import logging
import math
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import requests
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from numpy.linalg import LinAlgError

from implicit_bathy.inference.inference_classes import Plotter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


class PlotterVS(Plotter):
    """
    Plot Volume/Surface class
    """

    def __init__(self, processor):
        super().__init__(processor)
        ref = self.dataset.reference_date
        data = torch.load(self.dataset.dataset_path)
        list_nb_j = sorted(data["list_dates"])
        self.list_dates = [ref + timedelta(days=d) for d in list_nb_j]
        if self.processor.plot_params.vt_path:
            self.vt_dict = self.processor._load_vt_vh()
        else:
            self.vt_dict = {}

    def _compute_predicted_volume(self, chosen_time):
        params = self.processor.plot_params
        s2_tif_path = params.s2_tif_path
        if Path(s2_tif_path).exists():
            with rio.open(s2_tif_path) as src:
                raster_contour = self.processor.rasterize_contour(
                    src.height, src.width, src.transform, src.crs
                )
                # Generate input data that covers the whole range of x and y for a chosen time
                generated_input, _, _ = (
                    self.simulation.simulate_input_for_3D_grid(
                        src.width, src.height, chosen_time
                    )
                )

                # Get predictions from the model
                out_dict = self.processor.make_prediction(generated_input)
                result_h = self.processor.format_model_outputs(
                    out_dict, "h", src.height, src.width
                )
                result_hg = self.processor.format_model_outputs(
                    out_dict, "hg", src.height, src.width
                )
                result_hhg = result_h - result_hg
                result_hhg[raster_contour == 0] = np.nan

                if params.volume_inference == "mean":
                    max_label = self.processor.format_model_outputs(
                        out_dict, "max_label", src.height, src.width
                    )
                    result_hhg[max_label == 1] = np.nan
                    return np.nansum(result_hhg * 100) * 1e-6
                if params.volume_inference == "bayesian":
                    result_pw = self.processor.format_model_outputs(
                        out_dict, "pw", src.height, src.width
                    )
                    result_hhg[result_pw < params.p_w_filter] = np.nan
                    return np.nansum(result_hhg * 100) * 1e-6

    def get_data_vt(self):
        """
        Get vt
        """
        if self.vt_dict:
            self.data_vt = pd.DataFrame(self.vt_dict)
            self.data_vt["relative_wse_vt"] = (
                self.data_vt["wse_vt"] - self.data_vt["wse_vt"].iloc[0]
            )
            self.data_vt["date"] = pd.to_datetime(
                self.data_vt["t"], format="%d-%m-%Y", errors="coerce"
            )
            self.data_vt["date"] = self.data_vt["date"].dt.strftime("%Y-%m-%d")

    def get_data_hydrocron(self):
        """
        Get wse hydrocron
        """
        if self.processor.plot_params.id_feature_swot:
            id_feature_swot = self.processor.plot_params.id_feature_swot
            start_time = datetime.strptime(
                self.processor.plot_params.start_time, "%d-%m-%Y"
            ).strftime("%Y-%m-%d")
            end_time = (
                datetime.strptime(
                    self.processor.plot_params.end_time, "%d-%m-%Y"
                )
                + timedelta(days=1)
            ).strftime("%Y-%m-%d")

            hydrocron_response = requests.get(
                f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
                f"feature=PriorLake&feature_id={id_feature_swot}"
                f"&start_time={start_time}T00:00:00Z"
                f"&end_time={end_time}T00:00:00Z"
                f"&output=csv"
                f"&fields=time_str,wse,area_total,quality_f,partial_f,ds1_l,ds1_q",
                timeout=(5, 30),
            ).json()
            csv_str = hydrocron_response["results"]["csv"]
            df_hydrocron = pd.read_csv(StringIO(csv_str))
            # We remove when quality = -999 and partial f != 1
            df_hydrocron = df_hydrocron[
                (df_hydrocron["time_str"] != "no_data")
                & (df_hydrocron["quality_f"] != -999)
                & (df_hydrocron["quality_f"] < 2)
                & (df_hydrocron["wse"] > 0)
                & (df_hydrocron["partial_f"] != 1)
                & (df_hydrocron["partial_f"] != -999)
            ]
            df_hydrocron["date"] = pd.to_datetime(df_hydrocron["time_str"])
            df_hydrocron.attrs["wse_ref"] = df_hydrocron["wse"].iloc[0]
            df_hydrocron.attrs["area_ref"] = df_hydrocron["area_total"].iloc[0]
            df_hydrocron["relative_wse_hc"] = (
                df_hydrocron["wse"] - df_hydrocron["wse"].iloc[0]
            )

            # find T0 hydrocron
            find_first_date = None
            if self.vt_dict:
                for date in sorted(df_hydrocron["date"]):
                    date_ref = date.strftime("%d-%m-%Y")
                    index = self.data_vt["t"].tolist().index(date_ref)
                    vol = self.data_vt["vol_vt"].tolist()[index]
                    if not math.isnan(vol):
                        find_first_date = date
                        break

                if find_first_date:
                    self.find_first_date = find_first_date
                    df_first_date = df_hydrocron[
                        df_hydrocron["date"] == find_first_date
                    ]
                    df_hydrocron.attrs["wse_ref"] = df_first_date["wse"].iloc[0]
                    df_hydrocron.attrs["area_ref"] = df_first_date[
                        "area_total"
                    ].iloc[0]
                    df_hydrocron["relative_wse_hc"] = (
                        df_hydrocron["wse"] - df_hydrocron["wse"].iloc[0]
                    )
                    df_hydrocron["ds1_l"] = df_hydrocron["ds1_l"] * 1000
                    df_hydrocron["stockchange"] = (
                        df_hydrocron["ds1_l"] - df_hydrocron["ds1_l"].iloc[0]
                    )
                    find_first_date = pd.Timestamp(find_first_date)
                    find_first_date = find_first_date.strftime("%Y-%m-%d")
                    self.data_vt = self.data_vt[
                        self.data_vt["date"] >= find_first_date
                    ]
                    self.data_vt["relative_wse_vt"] = (
                        self.data_vt["wse_vt"] - self.data_vt["wse_vt"].iloc[0]
                    )
                    self.data_vt["delta_vol_vt"] = (
                        self.data_vt["vol_vt"] - self.data_vt["vol_vt"].iloc[0]
                    )
            else:
                self.find_first_date = start_time
                df_hydrocron["ds1_l"] = df_hydrocron["ds1_l"] * 1000
                df_hydrocron["stockchange"] = (
                    df_hydrocron["ds1_l"] - df_hydrocron["ds1_l"].iloc[0]
                )

            self.df_hydrocron = df_hydrocron.rename(
                columns={"wse": "wse_hc", "area_total": "area_total_hc"}
            )

    def get_swot_wse(self, t_values):
        """
        Caculate automatically swot_wse with phase_noise
        """
        dates = self.processor.dataset.X[:, 2]
        height = self.processor.dataset.Y
        classif = self.processor.dataset.classification
        phase_noise_level = self.processor.dataset.phase_noise_level
        dheight_dphase = self.processor.dataset.dheight_dphase
        mask = classif == 4
        filtered_height = height[mask]
        filtered_dates = dates[mask]
        filtered_phase_noise_level = phase_noise_level[mask]
        filtered_dheight_dphase = dheight_dphase[mask]
        self.swot_wse = []
        for date in t_values:
            date_mask = filtered_dates == date
            height_for_date = filtered_height[date_mask]
            phase_noise_for_date = filtered_phase_noise_level[date_mask]
            dheight_dphase_for_date = filtered_dheight_dphase[date_mask]
            wp = 1 / (phase_noise_for_date * dheight_dphase_for_date)
            heigh_wp = (wp.numpy()) * height_for_date.ravel().numpy()
            sum_height_wp = np.sum(heigh_wp)
            sum_wp = np.sum(wp.numpy())
            mean = sum_height_wp / sum_wp
            wse_span = self.dataset.max_wse - self.dataset.min_wse
            mean = mean * wse_span + self.dataset.min_wse
            self.swot_wse.append(float(mean))

    def get_pred_data_t(self):
        """
        Get height/surface/volume prediction
        """
        if self.vt_dict:
            self.vt_t = self.data_vt["t_normalized"].to_numpy()
            self.vt_real_date = pd.to_datetime(
                self.data_vt["t"], format="%d-%m-%Y"
            )
            # Generate input data
            # x and y to 0 - same time than the vt (= vérité terrain)
            generated_x = torch.zeros(self.vt_t.shape)
            generated_y = torch.zeros(self.vt_t.shape)
            generated_t = torch.Tensor(self.vt_t)
            generated_input = (
                torch.stack([generated_x, generated_y, generated_t], 0)
                .transpose(0, 1)
                .to(device)
            )
        else:
            self.vt_t = np.linspace(
                0.0, 1.0, self.processor.plot_params.plot_density
            )
            self.vt_real_date = pd.to_datetime(
                self.processor.plot_params.start_time
            ) + self.vt_t * (
                pd.to_datetime(self.processor.plot_params.end_time)
                - pd.to_datetime(self.processor.plot_params.start_time)
            )
            # Generate input data that covers all the times,
            # with a fixed x and y
            generated_t = torch.Tensor(self.vt_t)
            generated_x = torch.zeros(generated_t.shape)
            generated_y = torch.zeros(generated_t.shape)
            generated_input = (
                torch.stack([generated_x, generated_y, generated_t], 0)
                .transpose(0, 1)
                .to(device)
            )

        # Get predictions from the model
        out_dict = self.processor.make_prediction(generated_input)

        # Reshape and "de-normalize" predictions
        # Get wse predictions
        wse_span = self.dataset.max_wse - self.dataset.min_wse
        self.wse_time_serie = (
            wse_span * out_dict["hw"].cpu().detach().numpy()
            + self.dataset.min_wse
        )
        self.relative_wse_time_serie = (
            self.wse_time_serie - self.wse_time_serie[0]
        )
        self.sigma_w_time_serie = (
            wse_span * out_dict["vw"].cpu().detach().numpy()
        )

        if self.processor.plot_params.id_feature_swot:
            self.data_hc = self.get_data_hydrocron()
        # Get surface predictions
        roi_area = self.processor._compute_roi_bounding_box_surface()
        surfaces = [
            self.processor.compute_predicted_surface(t, roi_area)
            for t in self.vt_t
        ]
        surfaces = np.array(surfaces)
        sigma_surface = self.plot_surface_wse(surfaces)

        # Get volume predictions
        wse_span = self.dataset.max_wse - self.dataset.min_wse
        total_volume = roi_area * wse_span
        logging.info("Total volume = %s m3", {total_volume})

        # predict volume for all the times in vt
        volumes = [self._compute_predicted_volume(t) for t in self.vt_t]
        volumes = np.array(volumes)
        delta_volumes = volumes - volumes[0]
        sigma_volume = self.plot_volume_wse(delta_volumes)

        self.data_pred = pd.DataFrame(
            {
                "date_pred_vt": self.vt_real_date,
                "wse_pred": self.wse_time_serie,
                "relative_wse_pred": self.relative_wse_time_serie,
                "sigma_wse_pred": self.sigma_w_time_serie,
                "area_pred": surfaces,
                "sigma_area_inf_pred": sigma_surface[0][0],
                "sigma_area_sup_pred": sigma_surface[1][0],
                "vol_pred": volumes,
                "delta_vol_pred": delta_volumes,
                "sigma_delta_vol_inf_pred": sigma_volume[0][0],
                "sigma_delta_vol_sup_pred": sigma_volume[1][0],
            }
        )
        if self.processor.plot_params.id_feature_swot:
            self.data_pred["date_pred_vt"] = pd.to_datetime(
                self.data_pred["date_pred_vt"]
            )
            self.data_pred = self.data_pred[
                self.data_pred["date_pred_vt"]
                > pd.Timestamp(self.find_first_date).tz_localize(None)
            ]

    def plot_relative_wse_t(self):
        """
        Plot relative wse
        """
        # plot RELATIVE WSE / relative to the first value
        plt.figure()
        plt.scatter(
            self.data_pred.date_pred_vt,
            self.data_pred.relative_wse_pred,
            c="blue",
            label="Implicit Bathy prediction (ours)",
        )
        plt.fill_between(
            self.data_pred.date_pred_vt,
            self.data_pred.relative_wse_pred
            - 2 * self.data_pred.sigma_wse_pred,
            self.data_pred.relative_wse_pred
            + 2 * self.data_pred.sigma_wse_pred,
            alpha=0.2,
            color="black",
            label="+/- 2*sigma (95%)",
        )
        if len(self.swot_wse) > 0:
            min_swot = self.swot_wse[0]
            relative_swot_swe = [item - min_swot for item in self.swot_wse]
            plt.scatter(
                self.list_dates,
                relative_swot_swe,
                c="r",
                marker="x",
                zorder=10,
                label="Swot data",
            )
        if self.vt_dict:
            plt.scatter(
                self.data_vt.date,
                self.data_vt.relative_wse_vt,
                c="orange",
                label="In situ",
            )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        zmin, zmax = (
            self.data_pred.relative_wse_pred.min(),
            self.data_pred.relative_wse_pred.max(),
        )
        for t in self.list_dates:
            plt.plot([t, t], [zmin, zmax], ":", color="gray")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.ylabel("Water Surface Elevation relative to first value (m)")
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "relative_to_first_value_wse_time_serie"
        )
        plt.close()

    def plot_relative_wse_t_hydrocron(self):
        """
        Plot relative wse with hydrocron data
        """
        if self.processor.plot_params.id_feature_swot:
            plt.figure()
            plt.scatter(
                self.data_pred.date_pred_vt,
                self.data_pred.relative_wse_pred,
                c="blue",
                label="Implicit Bathy prediction (ours)",
            )
            plt.fill_between(
                self.data_pred.date_pred_vt,
                self.data_pred.relative_wse_pred
                - 2 * self.data_pred.sigma_wse_pred,
                self.data_pred.relative_wse_pred
                + 2 * self.data_pred.sigma_wse_pred,
                alpha=0.2,
                color="black",
                label="+/- 2*sigma (95%)",
            )
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron["quality_f"]
            ]
            plt.scatter(
                self.df_hydrocron.time_str,
                self.df_hydrocron.relative_wse_hc,
                c=colors,
                marker="x",
                zorder=10,
                label="LAKE SP \nqual=0 : green \nqual=1 : red",
            )
            if self.vt_dict:
                plt.scatter(
                    self.data_vt.date,
                    self.data_vt.relative_wse_vt,
                    c="orange",
                    label="In situ",
                )
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=3,
                fontsize=6,
            )
            # plot vertical time lines
            zmin, zmax = (
                min(self.data_pred.relative_wse_pred),
                max(self.data_pred.relative_wse_pred),
            )
            for t in self.list_dates:
                plt.plot([t, t], [zmin, zmax], ":", color="gray")
            plt.xlabel("Date")
            plt.ylabel(
                "Relative Water Surface Elevation to the first value (m)"
            )
            plt.xticks(rotation=45)
            plt.subplots_adjust(bottom=0.25)
            plt.savefig(
                self.processor.plot_params.save_path_for_plots
                / "relative_to_first_value_wse_time_serie_hydrocron.svg"
            )
            plt.savefig(
                self.processor.plot_params.save_path_for_plots
                / "relative_to_first_value_wse_time_serie_hydrocron.png"
            )
            plt.close()

    def plot_absolute_wse_t(self, t_values):
        """
        Plot absolute wse
        """
        # plot ABSOLUTE WSE
        swot_wse = self.get_swot_wse(t_values)
        plt.figure()
        plt.scatter(
            self.data_pred.date_pred_vt,
            self.data_pred.wse_pred,
            c="blue",
            label="Implicit Bathy prediction (ours)",
        )
        plt.fill_between(
            self.data_pred.date_pred_vt,
            self.data_pred.wse_pred - 2 * self.data_pred.sigma_wse_pred,
            self.data_pred.wse_pred + 2 * self.data_pred.sigma_wse_pred,
            alpha=0.2,
            color="black",
            label="+/- 2*sigma (95%)",
        )
        if swot_wse:
            plt.scatter(
                self.list_dates,
                swot_wse,
                c="r",
                marker="x",
                label="Swot data",
            )
        if self.vt_dict:
            plt.scatter(
                self.data_vt.date,
                self.data_vt.wse_vt,
                c="orange",
                label="In situ",
            )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        # plot vertical time lines
        zmin, zmax = (
            self.data_pred.wse_pred.min(),
            self.data_pred.wse_pred.max(),
        )
        for t in self.list_dates:
            plt.plot([t, t], [zmin, zmax], ":", color="gray")
        plt.xlabel("Date")
        plt.ylabel("Water Surface Elevation (m)")
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "absolute_wse_time_serie"
        )
        plt.close()

    def plot_absolute_wse_t_hydrocron(self):
        """
        PLOT ABSOLUTE WSE with hydrocron
        """
        if self.processor.plot_params.id_feature_swot:
            plt.figure()
            plt.scatter(
                self.data_pred.date_pred_vt,
                self.data_pred.wse_pred,
                c="blue",
                label="Implicit Bathy prediction (ours)",
            )
            plt.fill_between(
                self.data_pred.date_pred_vt,
                self.data_pred.wse_pred - 2 * self.data_pred.sigma_wse_pred,
                self.data_pred.wse_pred + 2 * self.data_pred.sigma_wse_pred,
                alpha=0.2,
                color="black",
                label="+/- 2*sigma (95%)",
            )
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron["quality_f"]
            ]
            plt.scatter(
                self.df_hydrocron.time_str,
                self.df_hydrocron.wse_hc,
                c=colors,
                marker="x",
                zorder=10,
                label="LAKE SP \nqual=0 : green \nqual=1 : red",
            )
            if self.vt_dict:
                plt.scatter(
                    self.data_vt.date,
                    self.data_vt.wse_vt,
                    c="orange",
                    label="In situ",
                )
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=3,
                fontsize=6,
            )
            # plot vertical time lines
            zmin, zmax = (
                self.data_pred.wse_pred.min(),
                self.data_pred.wse_pred.max(),
            )
            for t in self.list_dates:
                plt.plot([t, t], [zmin, zmax], ":", color="gray")
            plt.xlabel("Date")
            plt.xticks(rotation=45)
            plt.subplots_adjust(bottom=0.25)
            plt.ylabel("Water Surface Elevation (m)")
            plt.savefig(
                self.processor.plot_params.save_path_for_plots
                / "absolute_wse_time_serie_hydrocron"
            )
            plt.close()

    def plot_surface_t_hydrocron(self):
        """
        Plot ABSOLUTE AREA
        """
        plt.figure()
        plt.scatter(
            self.data_pred.date_pred_vt,
            self.data_pred.area_pred,
            label="Implicit Bathy prediction (ours)",
        )
        plt.fill_between(
            self.data_pred.date_pred_vt,
            self.data_pred.sigma_area_inf_pred,
            self.data_pred.sigma_area_sup_pred,
            alpha=0.2,
            color="black",
            label="∆S(T)",
        )
        if self.vt_dict:
            plt.scatter(
                self.data_vt.date,
                self.data_vt.area_vt,
                label="In situ",
                c="orange",
            )
        if self.processor.plot_params.id_feature_swot:
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron.quality_f
            ]
            plt.scatter(
                self.df_hydrocron.time_str,
                self.df_hydrocron.area_total_hc,
                c=colors,
                marker="x",
                zorder=10,
                label="LAKE SP \nqual=0 : green \nqual=1 : red",
            )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        smin, smax = (
            self.data_pred.area_pred.min(),
            self.data_pred.area_pred.max(),
        )
        # plot vertical time lines
        for t in self.list_dates:
            plt.plot([t, t], [smin, smax], ":", color="gray")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.ylabel("Water Area (km2)")
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "absolute_surface_time_serie.svg"
        )
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "absolute_surface_time_serie.png"
        )
        plt.close()

    def plot_volume_t(self):
        """
        Plot volume / t
        """
        plt.figure()
        plt.scatter(
            self.data_pred.date_pred_vt,
            self.data_pred.delta_vol_pred,
            c="blue",
            label="Implicit Bathy prediction (ours)",
        )
        plt.fill_between(
            self.data_pred.date_pred_vt,
            self.data_pred.sigma_delta_vol_inf_pred,
            self.data_pred.sigma_delta_vol_sup_pred,
            alpha=0.2,
            color="black",
            label="∆V(T)",
        )
        if self.vt_dict:
            plt.scatter(
                self.data_vt.date,
                self.data_vt.delta_vol_vt,
                c="orange",
                label="In situ",
            )
        if self.processor.plot_params.id_feature_swot:
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron.quality_f
            ]
            plt.scatter(
                self.df_hydrocron.time_str,
                self.df_hydrocron.stockchange,
                c=colors,
                marker="x",
                zorder=10,
                label="LAKE SP \nqual=0 : green \nqual=1 : red",
            )
        relative_vmin, relative_vmax = (
            self.data_pred.delta_vol_pred.min(),
            self.data_pred.delta_vol_pred.max(),
        )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        for t in self.list_dates:
            plt.plot([t, t], [relative_vmin, relative_vmax], ":", color="gray")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.ylabel("Water Volume Change relative to first date (hm3)")
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "relative_volume_time_serie.svg"
        )
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "relative_volume_time_serie.png"
        )
        plt.close()

    def extra_linear(self, h, y_smooth):
        """
        Extrapolation in plot wse/surface
        """
        h_min = np.min(h)
        h_max = np.max(h)
        delta = h_max - h_min

        low_zone = h <= (h_min + 0.2 * delta)
        high_zone = h >= (h_max - 0.2 * delta)

        coef_low = np.polyfit(h[low_zone], y_smooth[low_zone], 1)
        coef_high = np.polyfit(h[high_zone], y_smooth[high_zone], 1)
        interp_func = interp1d(h, y_smooth, kind="linear", bounds_error=False)

        def f(x):
            x = np.array(x, dtype=float)
            result = np.zeros_like(x)
            mask_low = x < h_min
            mask_mid = (x >= h_min) & (x <= h_max)
            mask_high = x > h_max
            # RELU
            result[mask_low] = np.maximum(
                0, coef_low[0] * x[mask_low] + coef_low[1]
            )
            result[mask_mid] = interp_func(x[mask_mid])
            result[mask_high] = np.maximum(
                0, coef_high[0] * x[mask_high] + coef_high[1]
            )
            return result

        return f, h_min, h_max

    def plot_surface_wse(self, surfaces):
        """
        Plot WSE / surface
        """
        wse_min = self.wse_time_serie - self.sigma_w_time_serie
        wse_max = self.wse_time_serie + self.sigma_w_time_serie
        unpacked_surfaces = np.array(surfaces)

        self.relative_wse = self.wse_time_serie - self.wse_time_serie[0]
        _, ax = plt.subplots()
        _ = ax.scatter(
            self.relative_wse,
            unpacked_surfaces,
            c="b",
            label="Implicit Bathy prediction (ours)",
        )
        h_relative = np.array(self.relative_wse).flatten()
        h = np.array(self.wse_time_serie).flatten()
        y = np.array(unpacked_surfaces).flatten()
        y_smooth = savgol_filter(y, window_length=7, polyorder=3)

        f, h_min, h_max = self.extra_linear(h, y_smooth)
        delta = h_max - h_min

        x_low_abs = np.linspace(h_min - 0.5 * delta, h_min, 50)
        x_high_abs = np.linspace(h_max, h_max + 0.5 * delta, 50)

        href = h[0] - h_relative[0]

        x_low_rel = x_low_abs - href
        x_high_rel = x_high_abs - href

        y_low = f(x_low_abs)
        y_high = f(x_high_abs)

        plt.plot(x_low_rel, y_low, "--", label="extrap basse")
        plt.plot(x_high_rel, y_high, "--", label="extrap haute")
        plt.plot(
            h_relative, y_smooth, color="red", linestyle="-", label="", zorder=1
        )
        sigma_surface = [[f(wse_min)], [f(wse_max)]]
        if self.vt_dict:
            plt.scatter(
                self.data_vt["relative_wse_vt"],
                self.data_vt["area_vt"],
                c="orange",
                label="ground truth",
            )
        if self.processor.plot_params.id_feature_swot:
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron["quality_f"]
            ]
            plt.scatter(
                self.df_hydrocron["relative_wse_hc"],
                self.df_hydrocron["area_total_hc"],
                c=colors,
                marker="x",
                label="Lake SP \nqual=0 : green \nqual=1 : red",
            )
        plt.xlabel("Relative Water Surface Elevation (m)")
        plt.ylabel("Water Area (km2)")
        plt.subplots_adjust(bottom=0.25)
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "relative_surface_wse"
        )
        plt.close()
        return sigma_surface

    def compute_dynamic_range_percentiles(self, zmin, zmax, percentile):
        """
        compute dynamic range percentiles
        """
        return zmin + percentile * (zmax - zmin)

    def plot_volume_wse(self, delta_vol):
        """
        Plot volume / wse
        """
        wse_min = (
            self.wse_time_serie - self.sigma_w_time_serie
        ) - self.wse_time_serie[0]
        wse_max = (
            self.wse_time_serie + self.sigma_w_time_serie
        ) - self.wse_time_serie[0]
        z_normalized = self.wse_time_serie - self.wse_time_serie[0]
        list_percentile = [0.25, 0.50, 0.75, 1]
        z_n = []
        for percentile in list_percentile:
            z = self.compute_dynamic_range_percentiles(
                z_normalized.min(), z_normalized.max(), percentile
            )
            z_n.append(z)

        _, ax = plt.subplots()
        _ = ax.scatter(
            z_normalized,
            delta_vol,
            c="b",
            label="Implicit Bathy prediction (ours)",
        )
        try:
            z = np.polyfit(
                z_normalized, delta_vol, self.processor.plot_params.deg_polyfit
            )
            p = np.poly1d(z)

        except LinAlgError:
            logging.error("Polyfit failed: SVD did not converge ")
            p = None
            return [[np.nan] * len(z_normalized), [np.nan] * len(z_normalized)]
        deg = len(p.c) - 1
        label = "y = " + " + ".join(
            (
                f"{c:.5f}x^{deg - i}"
                if deg - i > 1
                else f"{c:.5f}x"
                if deg - i == 1
                else f"{c:.5f}"
            )
            for i, c in enumerate(p.c)
        ).replace("+ -", "- ")
        pred_n = []
        for z in z_n:
            pred = p(z)
            pred_n.append(pred)
        (_,) = ax.plot(
            z_normalized,
            p(z_normalized),
            color="k",
            label=label,
        )
        sigma_volume = [[p(wse_min)], [p(wse_max)]]
        if self.vt_dict:
            h_norm = self.data_vt.relative_wse_vt
            gt = ax.scatter(
                h_norm,
                self.data_vt.delta_vol_vt,
                c="orange",
                label="ground truth",
            )
            try:
                mask = ~(np.isnan(h_norm) | np.isnan(self.data_vt.delta_vol_vt))
                z_vt = np.polyfit(
                    h_norm[mask],
                    self.data_vt.delta_vol_vt[mask],
                    self.processor.plot_params.deg_polyfit,
                )
                p_vt = np.poly1d(z_vt)
                deg = len(p_vt.c) - 1
                label_vt = "y = " + " + ".join(
                    (
                        f"{c:.5f}x^{deg - i}"
                        if deg - i > 1
                        else f"{c:.5f}x"
                        if deg - i == 1
                        else f"{c:.5f}"
                    )
                    for i, c in enumerate(p_vt.c)
                ).replace("+ -", "- ")
                true_n = []
                for z in z_n:
                    true = p_vt(z)
                    true_n.append(true)
                gt_fit = p_vt(h_norm[mask])
                (gt_fit,) = ax.plot(
                    h_norm[mask],
                    p_vt(h_norm[mask]),
                    color="k",
                    label=label_vt,
                )
                _ = [gt, gt_fit]
                ecarts = [round((x - y), 4) for x, y in zip(pred_n, true_n)]
                results = {
                    "E_25%": ecarts[0],
                    "E_50%": ecarts[1],
                    "E_75%": ecarts[2],
                    "E_100%": ecarts[3],
                }
                self.processor.insert_indicators_report(results)
            except np.linalg.LinAlgError as e:
                logging.info("Polyfit a échoué : %s", {e})
                z_vt = None
        if self.processor.plot_params.id_feature_swot:
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron["quality_f"]
            ]
            _ = ax.scatter(
                self.df_hydrocron["relative_wse_hc"],
                self.df_hydrocron["stockchange"],
                c=colors,
                marker="x",
                label="Lake SP \nqual=0 : green \nqual=1 : red",
                zorder=2,
            )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        plt.xlabel("Water Surface Elevation relative to the first date (m)")
        plt.ylabel("Water Volume Change relative to the first date (hm3)")
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "relative_volume_wse"
        )
        plt.close()
        return sigma_volume

    def plot_volume_area(self):
        """
        Plot volume / area
        """

        _, ax = plt.subplots()
        _ = ax.scatter(
            self.data_pred.area_pred,
            self.data_pred.delta_vol_pred,
            c="b",
            label="Implicit Bathy prediction (ours)",
        )
        if self.vt_dict:
            plt.scatter(
                self.data_vt.area_vt,
                self.data_vt.delta_vol_vt,
                c="orange",
                label="ground truth",
            )
        if self.processor.plot_params.id_feature_swot:
            colors = [
                "green" if q == 0 else "red" if q == 1 else "pink"
                for q in self.df_hydrocron["quality_f"]
            ]
            ax.scatter(
                self.df_hydrocron["area_total_hc"],
                self.df_hydrocron["stockchange"],
                c=colors,
                marker="x",
                label="Lake SP \nqual=0 : green \nqual=1 : red",
                zorder=2,
            )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6
        )
        plt.xlabel("Absolute Water Surface (km2)")
        plt.ylabel("Water Volume Change relative to the first date (hm3)")
        plt.savefig(
            self.processor.plot_params.save_path_for_plots
            / "relative_volume_area"
        )
        plt.close()

    def export_csv(self):
        """
        Export CSV
        """
        s2_tif_path = self.processor.plot_params.s2_tif_path
        data_path = Path(s2_tif_path).parent
        df_id_swot = pd.read_csv(
            (Path(data_path) / "dataset_id_swot.txt"),
            header=None,
            names=["id_swot"],
        )
        df_id_swot["date"] = pd.to_datetime(
            df_id_swot["id_swot"].str.split("_").str[0], format="%Y%m%d"
        ).dt.date
        self.path_data_csv = (
            self.processor.plot_params.save_path_for_plots / "data.csv"
        )
        self.data_pred["date"] = pd.to_datetime(
            self.data_pred["date_pred_vt"]
        ).dt.date
        self.data_csv = self.data_pred
        if self.vt_dict:
            self.data_vt["date"] = pd.to_datetime(self.data_vt["date"]).dt.date
            self.data_csv = pd.merge(
                self.data_vt, self.data_csv, on="date", how="inner"
            )
        if self.processor.plot_params.id_feature_swot:
            self.df_hydrocron["date"] = pd.to_datetime(
                self.df_hydrocron["time_str"]
            ).dt.date
            self.data_csv = pd.merge(
                self.df_hydrocron, self.data_csv, on="date", how="inner"
            )
        self.data_csv = pd.merge(
            self.data_csv, df_id_swot, on="date", how="inner"
        )
        self.data_csv.to_csv(self.path_data_csv)

    def compute_metrics(self, df, col_true, col_pred):
        """
        Compute metrics
        """
        if col_true in df.columns and col_pred in df.columns:
            mare, mae, mre, me = self.processor.mean_absolute_rel_error(
                df[col_true], df[col_pred]
            )
            r = {}
            if col_true == "delta_vol_vt":
                mae, maern, maetv = self.processor.mean_error_delta_vol(
                    df[col_true], df[col_pred], df[col_true].max()
                )
                r.update({"mae": mae, "maern": maern, "maetv": maetv})
            q68, _ = self.processor.quantile_68(df[col_true], df[col_pred])
            r.update(
                {"mare": mare, "mae": mae, "mre": mre, "me": me, "q68": q68}
            )
            return r
        return {"mare": np.nan, "mae": np.nan, "me": np.nan, "q68": np.nan}

    def compute_indicators(self):
        """
        Compute all indicator values
        """

        configs = {
            "height": (
                "relative_wse_vt",
                "relative_wse_hc",
                "relative_wse_pred",
            ),
            "area": ("area_vt", "area_total_hc", "area_pred"),
            "delta_volume": ("delta_vol_vt", "stockchange", "delta_vol_pred"),
        }

        results = {}

        results = {
            f"{metric}_{key}_{suffix}": value
            for key, (truth, hc, pred) in configs.items()
            for suffix, col in [("hc", hc), ("pred", pred)]
            if col in self.data_csv.columns
            for metric, value in self.compute_metrics(
                self.data_csv, truth, col
            ).items()
        }
        self.processor.insert_indicators_report(results)
