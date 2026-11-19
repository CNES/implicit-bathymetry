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
Plot cross section class
"""

from datetime import timedelta

import matplotlib.pyplot as plt
import torch

from implicit_bathy.inference.inference_classes import Plotter


class PlotterCrossSection(Plotter):
    """
    Plot cross section class
    """

    def _generate_cross_section_data(self, t_values, x, dim):
        """
        Generates input data and predictions for a single cross-section.

        Args:
            t_values (list): List of times for the cross-section.
            x (float): Position of the cross-section in the dimension.
            dim (str): Dimension being plotted ('x' or 'y').

        Returns:
            x0: Positions in the cross-section.
            hg, hw, out: Predictions for ground, water, and final outputs.
        """

        # Generate input data
        x0, generated_input = self.simulation.simulate_input_for_2D_cut(
            t_values, x, self.plot_density_x, self.plot_density_y, dim
        )

        # Get predictions from the model
        out_dict = self.processor.make_prediction(generated_input)

        # Reshape predictions for each time
        hg = out_dict["hg"].reshape(len(t_values), -1).cpu().detach().numpy()
        hw = out_dict["hw"].reshape(len(t_values), -1).cpu().detach().numpy()
        out = out_dict["h"].reshape(len(t_values), -1).cpu().detach().numpy()

        # "De-normalize" predictions
        wse_span = self.dataset.max_wse - self.dataset.min_wse
        hg = hg * wse_span + self.dataset.min_wse
        hw = hw * wse_span + self.dataset.min_wse
        out = out * wse_span + self.dataset.min_wse

        if self.processor.training_params.bayesian_model:
            sigma_g = (
                out_dict["vg"].reshape(len(t_values), -1).cpu().detach().numpy()
            )
            sigma_w = (
                out_dict["vw"].reshape(len(t_values), -1).cpu().detach().numpy()
            )
            sigma_g = sigma_g * wse_span
            sigma_w = sigma_w * wse_span
            return x0, hg, hw, out, sigma_g, sigma_w
        return x0, hg, hw, out, [], []

    def _plot_cross_section(
        self, ax, x0, x, dim, hg, hw, out, sigma_g=[], sigma_w=[]
    ):
        """
        Plots a single cross-section.

        Args:
            ax: Matplotlib axis object.
            x0: Positions in the cross-section.
            x: Position of the cross-section in the dimension.
            dim: Dimension being plotted ('x' or 'y').
            hg, hw, out: Predictions for ground, water, and final outputs.
        """
        # Plot model predictions
        # multiply hg, hw and out with the good row in self.mask_nan_values

        if dim == "x":
            x_value_nan_mask = int(x * self.extend_mask.shape[1])
            this_nan_mask = self.extend_mask[::-1, x_value_nan_mask]
        elif dim == "y":
            x_value_nan_mask = int((1 - x) * self.extend_mask.shape[0])
            this_nan_mask = self.extend_mask[x_value_nan_mask, :]

        for i in range(hg.shape[0]):
            ax.plot(
                x0,
                hg[i, :] * this_nan_mask,
                label="hg",
                color="black",
                linewidth=1,
            )  # shape (1, 128) if only one time
            ax.plot(
                x0,
                hw[i, :] * this_nan_mask,
                label="hw",
                color="blue",
                linewidth=1,
            )
            ax.plot(
                x0,
                out[i, :] * this_nan_mask,
                label="out",
                color="red",
                linewidth=3,
                linestyle="--",
            )
        if self.processor.plot_params.dsm_path is not None:
            # Plot ground truth DSM
            if dim == "y":
                data_array = self.gt_dsm
                ax.plot(
                    x0,
                    data_array[(int(self.plot_density_x * (1 - x))), :],
                    c="green",
                    linestyle="dotted",
                )
            elif dim == "x":
                data_array = self.gt_dsm[::-1, :]
                ax.plot(
                    x0,
                    data_array[:, int(self.plot_density_y * x)],
                    c="green",
                    linestyle="dotted",
                )

        if self.processor.training_params.bayesian_model:
            ax.fill_between(
                x0,
                hg[i, :] * this_nan_mask - sigma_g[i, :],
                hg[i, :] * this_nan_mask + sigma_g[i, :],
                alpha=0.2,
                color="black",
            )
            ax.fill_between(
                x0,
                hw[i, :] * this_nan_mask - sigma_w[i, :],
                hw[i, :] * this_nan_mask + sigma_w[i, :],
                alpha=0.2,
                color="blue",
            )

        # Configure plot labels and legend
        ax.set_title(f"{dim} = {x:.2f}")
        ax.set_xlabel("x" if dim == "y" else "y" + ", normalized")
        ax.set_ylabel("wse, normalized")
        ax.legend(
            ["hg", "hw", "h", "DSM"], loc="upper right", bbox_to_anchor=(1, 1)
        )
        ax.set_ylim(
            self.dataset.min_wse,
            self.processor.plot_params.max_wse_for_X_plot,
        )
        ax.grid(False)

    def plot_many_cross_sections(self, t_values, dim="x"):
        """
        Plots many cross-sections.

        Args:
            t_values (list): list of time values.
            dim: Dimension being plotted ('x' or 'y').
        """
        ref = self.dataset.reference_date
        data = torch.load(self.dataset.dataset_path)
        list_nb_j = sorted(data["list_dates"])
        list_dates = [ref + timedelta(days=d) for d in list_nb_j]
        for _, (t, date) in enumerate(sorted(zip(t_values, list_dates))):
            # for now, let's fix "many" to 4
            params = self.processor.plot_params

            ### plot 4 subplots
            _, axs = plt.subplots(
                2, 2, figsize=(3 * params.fig_size, 3 * params.fig_size)
            )

            # cut-cross in (x,y) are hard-coded at [0.25, 0.45, 0.55, 0.65] OR [0.2, 0.4, 0.6, 0.8]
            for ax, x in zip(axs.flatten(), [0.2, 0.4, 0.6, 0.8]):
                ### Generate input and results for one X-cut
                # for all times asked
                # (in case of a standard network: sigma_g=[], sigma_w=[])
                x0, hg, hw, out, sigma_g, sigma_w = (
                    self._generate_cross_section_data([t], x, dim)
                )

                ### plot
                self._plot_cross_section(
                    ax, x0, x, dim, hg, hw, out, sigma_g, sigma_w
                )

                ## add data points underneath if asked
                self.dataset.plot_data_slice(
                    ax, x - 0.01, x + 0.01, dim, [t], True
                )

                ax.set_ylim(
                    self.dataset.min_wse,
                    self.processor.plot_params.max_wse_for_X_plot,
                )

            plt.savefig(
                params.save_path_for_plots
                / f"plot_cross_section_width_0_02_{dim}_time_{t:.3f}_{date}.png"
            )
            plt.close()
