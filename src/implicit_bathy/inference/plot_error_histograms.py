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
Plot error histograms class
"""

from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.cauchy import Cauchy
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal

from implicit_bathy.inference.inference_classes import Plotter


class PlotterErrorHistograms(Plotter):
    """
    Plot error histograms class
    """

    def _compute_model_error_and_sigma_prediction(
        self,
        target_date,
    ):
        """
        Compute model error and sigma prediction

        Args:
            target_date (float): date normalized.
        """
        min_wse = self.dataset.min_wse
        max_wse = self.dataset.max_wse
        data_this_time, data_wse_this_time, _, _, _ = (
            self.dataset.select_input_from_dataset(target_date)
        )

        out_dict = self.processor.make_prediction(data_this_time)
        h = out_dict["h"].cpu().detach().numpy()
        pw = out_dict["pw"].cpu().detach().numpy()
        sigma_g = out_dict["vg"].cpu().detach().numpy()
        sigma_w = out_dict["vw"].cpu().detach().numpy()
        wse_span = max_wse - min_wse

        ### Get sigma predictions from the model -------------------------------------
        ground_sigma = sigma_g * wse_span
        water_sigma = sigma_w * wse_span

        ### Compute error between model prediction and ground truth  -----------------------
        # data_err = (difference true h - h predicted)
        data_wse_this_time = (
            data_wse_this_time.cpu().detach().numpy()
        )  # still normalized [0 - 1]
        data_err = (data_wse_this_time[:, 0] - h) * wse_span

        # split errors into water and ground :
        params = self.processor.plot_params

        # Where pw > 0.999 (water_thresh), keep water_err = (difference true h - h predicted)
        water_indices = np.where(pw >= params.water_threshold_proba)[
            0
        ]  # water_thresh = 0.99
        water_error = data_err[water_indices]
        # water_data_for_hist = data_this_time[water_indices]

        # Where pw < 0.1 (ground_thresh), keep ground_err = (difference true h - h predicted)
        ground_indices = np.where(pw <= params.ground_threshold_proba)[0]
        ground_error = data_err[ground_indices]

        return (
            water_error,
            ground_error,
            water_sigma,
            ground_sigma,
        )

    def _plot_distrib(self, mean, var, vmin, vmax, color, water):
        """
        Plot distribution

        Args:
            mean (float): mean of data_err
            var (float): std of data_err
            vmin, vmax (float): min/max for x axis
            color (str): color in histogramm
            water (bool): true for water distribution

        """
        x = torch.linspace(vmin, vmax, 500)

        # probability models as defined in params
        distribution_map = {
            "gaussian": Normal,
            "cauchy": Cauchy,
            "laplace": Laplace,
        }
        mean_t = torch.tensor([mean])
        if (var_t := torch.tensor([var])) > 0:
            if water:
                distribution_name = "gaussian"
                mx = distribution_map[distribution_name](mean_t, var_t)
            else:
                distribution_name = (
                    self.processor.training_params.ground_prob.lower()
                )
                mx = distribution_map[distribution_name](mean_t, var_t)

            phi_x = np.exp(mx.log_prob(x))

            # Plotting the function f_z
            ax = plt.gca()
            ax.plot(x, phi_x, c=color)
            ax.legend([distribution_name])
        else:
            print("Distribution Variance is 0, cannot plot curve")

    def _plot_error_histogram(self, data, sigma, title, vmin, vmax, water=True):
        """
        Plot error histogram
        data_mse = (difference true h - h predicted)
        sigma = the variance estimated by the Bayesian model.

        Args:
            data (float): mean of data_err
            sigma (float): std of data_err
            title (str) : histo title
            vmin, vmax (float): min/max for x axis
            water (bool): true for water distribution
        """
        ax = plt.gca()

        if data.shape[0] > 0:
            # HISTOGRAM = data_err = (difference true h - h predicted)
            # ex : How many points have an error of 0? Of -0.2?
            ax.hist(data, bins=np.linspace(vmin, vmax, 100), density=True)
            ax.set_title(title)
            # METRICS FROM DATA_ERR = (difference true h - h predicted)
            mean = data.mean()  # mean of data_err = Mean Absolute deviation ?
            aad = (np.abs(data)).mean()  # average absolute deviation ?
            variance = ((data - mean) ** 2).mean()

            std = np.sqrt(variance)
            subtitle = (
                f"mean(wse-h) -> MAD={100 * mean:2.2f}cm "
                + f"\n mean(|wse-h|) -> AAD={100 * aad:2.2f}cm"
                + "\n Red : Distrib with std = STD des erreurs ={:2.2f}cm".format(
                    100 * std
                )
            )

            # METRICS FROM SIGMA computed by the network
            av_sigma = np.mean(sigma)
            if self.processor.training_params.bayesian_model:
                subtitle += "\n Green : Distrib with std = Average Sigma predicted ={:2.2f}cm".format(
                    100 * av_sigma
                )
                self._plot_distrib(mean, std, vmin, vmax, "red", water)
                self._plot_distrib(mean, av_sigma, vmin, vmax, "green", water)
            ax.set_xlabel(subtitle, loc="left")
            ax.set_ylabel("density")

    def plot_all_error_histograms(self, t_values):
        """
        Plot error histogramm
        Args:
            t_values (list): list of time values.
        """
        ref = self.dataset.reference_date
        data = torch.load(self.dataset.dataset_path)
        list_nb_j = sorted(data["list_dates"])
        list_dates = [ref + timedelta(days=d) for d in list_nb_j]
        for _, (target_date, date) in enumerate(
            sorted(zip(t_values, list_dates))
        ):
            (
                water_error,
                ground_error,
                water_sigma,
                ground_sigma,
            ) = self._compute_model_error_and_sigma_prediction(target_date)

            # plot
            fig = plt.figure(figsize=(5, 8))
            fig.add_subplot(4, 1, 1)
            self._plot_error_histogram(
                water_error,
                water_sigma,
                title="WATER ERROR = GT swot wse - model prediction h",
                vmin=-0.5,
                vmax=0.5,
            )

            fig.add_subplot(4, 1, 3)
            self._plot_error_histogram(
                ground_error,
                ground_sigma,
                title="GROUND ERROR : GT swot wse - model prediction h",
                vmin=-4,
                vmax=4,
                water=False,
            )
            plt.savefig(
                self.processor.plot_params.save_path_for_plots
                / f"error_histograms_time_{target_date:.3f}_{date}.png"
            )
            plt.close()
