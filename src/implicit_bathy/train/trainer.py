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
Class trainer
"""

import logging
import os
import time

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from implicit_bathy.config.config import DatasetConfig
from implicit_bathy.preprocess.datasetSWOT import MultiDateLakeDataset
from implicit_bathy.train.losses import (
    label_loss,
    nll_max,
    uncertainty_loss,
    water_loss,
)
from implicit_bathy.train.model import (
    BayesianTwoBranchModel,
    StandardTwoBranchModel,
)

logging.basicConfig(level=logging.INFO)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
MODE_TEST = os.getenv("MODE_TEST") or False


class StandardTrainer:
    """
    This class implement the training of "standard" (non-bayesian) model.
    All methods are shared with BayesianTrainer, which is a child class,
    except for **instantiate_model** and **train**.

    This class instantiate the StandardTwoBranchModel (model class).
    """

    def __init__(self, config_params):
        """
        Set parameters, load dataset and instantiate model.
        """
        # Select GPU usage -----------------------------------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        logging.info(self.device)

        # Will the simulation be random, or seeded?------------------------------
        if config_params.seed:
            torch.manual_seed(config_params.seed)
            torch.cuda.manual_seed_all(config_params.seed)
            torch.use_deterministic_algorithms(True)
            logging.info("Seed %s used.", config_params.seed)

        # Load config info --------------------------------------------------------
        self.n_epochs = config_params.n_epochs

        # Load the dataset -----------------------------------------------------------
        dataset_params = DatasetConfig(dataset_path=config_params.dataset_path)
        self.dataset = MultiDateLakeDataset(dataset_params)
        self.dataset.load_dataset()

        # config_params.xmin =
        # config_params.xmax =
        self.dataset.X = self.dataset.X.to(self.device)
        self.dataset.Y = self.dataset.Y.to(self.device)
        self.dataset.L = self.dataset.L.to(self.device)
        logging.info("Loading dataset :")
        logging.info(self.dataset.X.shape)  # to check if it's not empty
        if self.dataset.X.shape[0] == 0:
            logging.info("!!! WARNING ! THIS DATASET IS EMPTY !")

        # instantiate model -----------------------------------------------------------
        self.model = self.instantiate_model(config_params)
        self.weighted_by_phase_noise = config_params.weighted_by_phase_noise
        self.weighted_function = config_params.weighted_function

    def instantiate_model(self, config_params):
        """."""
        return StandardTwoBranchModel(config_params)

    def set_model_env(self, config_params):
        """
        init model
        """
        self.model.ground_net.to(self.device)
        self.model.water_net.to(self.device)
        logging.info(f"Init model {self.model.description}")

        # define optimizer and scheduler -----------------------------------------
        # the optimizer will update all the weights of the model
        # the name and learning rate are defined by the user in the config file
        optimizer_name = config_params.optimizer_name
        lr = config_params.lr
        self.optimizer = self.load_optimizer(
            self.model.parameters(), optimizer_name, lr
        )
        self.scheduler = ReduceLROnPlateau(
            self.load_optimizer(self.model.parameters(), optimizer_name, lr),
            factor=0.5,
            patience=3,
            min_lr=1e-8,
        )

        # reterieve some parameters infos ----------------------------------------
        model_params = self.model.parameters()
        self.num_params = sum(
            p.numel() for p in model_params if p.requires_grad
        )
        logging.info("Loaded model %s parameters", self.num_params)

        self.saving_path = config_params.save_path / self.model.description
        self.saving_path.mkdir(parents=True, exist_ok=True)

    def load_optimizer(self, params, optimizer_name, lr):
        """
        load optimizer
        """
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(params, lr)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(params, lr)
        elif optimizer_name == "RAdam":
            optimizer = torch.optim.RAdam(params, lr)
        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(params, lr)
        if MODE_TEST:
            isinstance(
                optimizer,
                (
                    torch.optim.Adam,
                    torch.optim.SGD,
                    torch.optim.RAdam,
                    torch.optim.AdamW,
                ),
            )
        return optimizer

    def save_model(self, path, epoch, config_params):
        """
        save model
        """
        parameters = {
            "model_state_dict": self.model.state_dict(),
            "water_embedding": self.model.water_net.weight,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_descript": self.model.description,
            "bayesian_model": config_params.bayesian_model,
            "ground_model": config_params.ground_model,
            "ground_model_params": config_params.ground_model_params,
            "water_model": config_params.water_model,
            "water_model_params": config_params.water_model_params,
            "optimizer_name": config_params.optimizer_name,
            "lr": config_params.lr,
            "n_epochs": self.n_epochs,
            "total_trainable_params": self.num_params,
            "model_name": self.saving_path
            / (self.model.description + f"_epoch{epoch}.pth"),
            "weighted_by_phase_noise": self.weighted_by_phase_noise,
            "weighted_function": self.weighted_function,
            "ground_model_mean_init": config_params.ground_model_mean_init,
            "ground_model_var_init": config_params.ground_model_var_init,
            "water_model_mean_init": config_params.water_model_mean_init,
            "water_model_var_init": config_params.water_model_var_init,
        }
        # ground_model_params :
        # IF MLP : 1) width, 2) depth, 3) x_freq_enc .
        # IF GARF : 1) width, 2) depth, 3) gaussian_act_sigma
        # if it's a bayesian model, in additon, save bayesian parameters :
        if config_params.bayesian_model:
            parameters.update(
                {
                    "ground_prob": config_params.ground_prob,
                    "lambda_label": config_params.lambda_label,
                    "n_epochs_wloss": config_params.n_epochs_wloss,
                    "lambda_wloss": config_params.lambda_wloss,
                    "lambda_uncert": config_params.lambda_uncert,
                    "wl_optimizer_name": config_params.wl_optimizer_name,
                    "wl_lr": config_params.wl_lr,
                }
            )
        torch.save(
            parameters,
            path,
        )

    def train(self, config_params):
        """
        Method that orchestrates the training for a Standard (non-bayesian) model.
        The model must predict the WSE of each tripplet (x, y, t).
        The Loss used is the MSE (Mean-Squared Error).
        """
        start_time = time.time()

        # init model, set optimizer and scheduler
        self.set_model_env(config_params)

        # Initializing lists that will be used to graph progress.
        train_loss_history = []
        water_ratio_history = []

        for epoch in range(self.n_epochs):
            self.model.train()

            # Separate the inputs and the labels.
            inputs = self.dataset.X  # torch.Size([nb_points, 3]) = x, y, t
            targets = self.dataset.Y  # WSE

            # Zeros the gradients.
            self.optimizer.zero_grad()

            # Make predictions
            outputs = self.model(inputs, config_params)

            # compute loss
            if self.weighted_by_phase_noise:
                if self.weighted_function == "inverse":
                    weights = (
                        torch.pow(self.dataset.phase_noise_level, -1)
                        .unsqueeze(1)
                        .cuda()
                    )
                if self.weighted_function == "normalized":
                    min_val = self.dataset.phase_noise_level.min()
                    max_val = self.dataset.phase_noise_level.max()
                    normalized_tensor = (
                        self.dataset.phase_noise_level - min_val
                    ) / (max_val - min_val)
                    weights = (1 - normalized_tensor).unsqueeze(1).cuda()
                loss = F.mse_loss(
                    outputs["h"], targets, weight=weights
                )  # Weighted Mean Squared error
            else:
                loss = F.mse_loss(outputs["h"], targets)  # Mean Squared Error

            # keep value of Loss for plot
            train_loss_history.append(loss.item())

            # compute an indicator to plot during training :
            # max_label outputs 1 is the points is classified as ground,
            # 0 if it's classified as water
            # the mean gives an indication of the ratio ground/water found in the dataset
            intermediate = (
                outputs["max_label"][:, 0] > outputs["max_label"][:, 1]
            ).type(torch.float16)
            water_ratio_history.append(torch.mean(intermediate, dim=0).item())

            # propagate backward
            loss.backward()

            # Adjust weights
            self.optimizer.step()

            # plot training evolution
            if (epoch != 0 and epoch % config_params.plot_every_epoch == 0) or (
                epoch == config_params.n_epochs - 1
            ):
                logging.info("Epoch n°" + str(epoch) + ":")
                plt.plot(train_loss_history, "k")
                plt.plot(water_ratio_history, "b")
                plt.legend(["MSE loss", "water_ratio"])
                plt.xlabel("epochs")
                plt.title("Model progress")
                plt.savefig(
                    self.saving_path
                    / (self.model.description + f"_epoch{epoch}.png")
                )
                logging.info("Current Loss %s", train_loss_history[epoch])

                # if asked by user, save model at intermediary steps
                if config_params.save_model_epoch:
                    self.save_model(
                        self.saving_path
                        / (self.model.description + f"_epoch{epoch}.pth"),
                        epoch,
                        config_params,
                    )

        # save model at the end of training
        self.save_model(
            self.saving_path / (self.model.description + f"_epoch{epoch}.pth"),
            epoch,
            config_params,
        )

        training_time = (time.time() - start_time) / 60
        logging.info("Training time took: %s minutes ", training_time)


class BayesianTrainer(StandardTrainer):
    """
    This class implement the training of Bayesian model.
    All methods are shared with StandardTrainer, which is the parent class,
    except for **instantiate_model** and **train**.

    This class instantiate the BayesianTwoBranchModel (model class).
    The training is also doifferent : the loss(es) used are not the same.
    """

    def instantiate_model(self, config_params):
        print(config_params)
        return BayesianTwoBranchModel(config_params)

    def train(self, config_params):
        """
        Method that orchestrates the training for a Bayesian model.
        The model must predict the WSE of each tripplet (x, y, t)
        AND THE VARIANCE OF THE WSE DISTRIBUTION.
        (the kind of distribution : Gaussian, Cauchy, etc) is chosen by the user.
        3 Losses are used sequentially :
        - first the Negative-Log-Likelihood
        - then a custom "label loss"
        - finally, a custom KL-loss
        (see file losses.py)
        The epoch of apparition of each Loss is chosen by the user in the parameters.
        """
        start_time = time.time()

        # init model
        self.set_model_env(config_params)

        # Initializing lists that will be used to graph progress.
        # List will contain tuples (epoch, value)
        nll_loss_history = []
        water_ratio_history = []
        ratio_points_pw_099_history = []
        ratio_points_pw_095_history = []
        ratio_points_pw_010_history = []
        uncert_loss_history = []
        label_loss_history = []
        water_loss_history = []

        # Estimation of weights (once for all the training process)
        if self.weighted_by_phase_noise:
            if self.weighted_function == "inverse":
                weights = (
                    torch.pow(self.dataset.phase_noise_level, -1)
                    .unsqueeze(1)
                    .cuda()
                )
            if self.weighted_function == "normalized":
                min_val = self.dataset.phase_noise_level.min()
                max_val = self.dataset.phase_noise_level.max()
                normalized_tensor = (
                    self.dataset.phase_noise_level - min_val
                ) / (max_val - min_val)
                weights = (1 - normalized_tensor).unsqueeze(1).cuda()
        # -------------------------------------------------------------------
        # PLOT SEPARATE INDICATORS PER TIME :
        # compute once the time masks
        # t_values = np.unique(self.dataset.X[:, 2].cpu().detach().numpy())
        # time_masks = []
        # for target_date in t_values:
        #     mask_this_time = self.dataset.X[:, 2].to(dtype=torch.float32) == np.float32(
        #         target_date
        #     )
        #     time_masks.append(mask_this_time)
        # -------------------------------------------------------------------

        for epoch in range(self.n_epochs):
            self.model.train()

            # Separate the inputs and the labels.
            inputs = self.dataset.X  # torch.Size([nb_points, 3]) = x, y, t
            targets = self.dataset.Y  # WSE
            target_label = self.dataset.L

            # Zeros the gradients.
            self.optimizer.zero_grad()

            # Make predictions
            outputs = self.model(inputs, config_params)

            if epoch % config_params.plot_every_epoch == 0:
                # Indicator functions -------------------------------------------------------
                # max_label outputs 1 is the points is classified as ground,
                # 0 if it's classified as water
                # the mean gives an indication of the ratio ground/water found in the dataset
                water_ratio = torch.mean(
                    outputs["max_label"].type(torch.float16), dim=0
                ).item()

                water_ratio_history.append((epoch, water_ratio))

                # -------------------------------------------------------------------
                # PLOT SEPARATE INDICATORS PER TIME :
                # all_ratios_this_time = []
                # for mask_time in time_masks:
                #     labels_this_time = outputs["max_label"][mask_time].type(torch.float16)
                #     ratio_this_time = torch.mean(labels_this_time, dim=0).item()
                #     all_ratios_this_time.append(ratio_this_time)
                # water_ratio_history.append(all_ratios_this_time)
                # -------------------------------------------------------------------

                # probabilty for each point to be water
                proba_water = outputs["pw"][..., None]
                nb_points_pw_099 = torch.sum(proba_water > 0.99)
                nb_points_pw_095 = torch.sum(proba_water > 0.95)
                nb_points_pw_010 = torch.sum(proba_water < 0.10)
                ratio_points_pw_099_history.append(
                    (epoch, nb_points_pw_099 / len(proba_water))
                )
                ratio_points_pw_095_history.append(
                    (epoch, nb_points_pw_095 / len(proba_water))
                )
                ratio_points_pw_010_history.append(
                    (epoch, nb_points_pw_010 / len(proba_water))
                )

            # FIRST LOSS -----------------------------------------------------------------
            # we start with the NLL Loss to create the most suited
            # probability distributions for ground and water
            nll_loss = nll_max(outputs, targets, config_params)

            # if weighted, the loss will be ponderated by phase noise level
            if self.weighted_by_phase_noise:
                nll_loss = torch.mul(nll_loss, weights)
            # loss = nll_loss
            loss = torch.mean(nll_loss, dim=0)

            # -------------------------------------------------------------------
            # PLOT SEPARATE INDICATORS PER TIME :
            # compute average NLL per time
            # average_nlls_this_time = []
            # for mask_time in time_masks:
            #     nll_this_time = nll_loss[mask_time]
            #     average_this_time = torch.mean(nll_this_time, dim=0)
            #     average_nlls_this_time.append(average_this_time.item())
            # nll_loss_history.append(average_nlls_this_time)
            # -------------------------------------------------------------------
            if epoch % config_params.plot_every_epoch == 0:
                nll_loss_history.append((epoch, loss.item()))

            # SECOND LOSS -----------------------------------------------------------------
            # after some time
            # (when the network is roughly able to discriminate water and ground points),
            # we ask the network to be more certain of its decisions
            if config_params.lambda_label:
                if config_params.lambda_label > 0:
                    weight = config_params.lambda_label
                    ponderated_label_loss = weight * label_loss(
                        outputs["pw"], target_label
                    )
                    loss += ponderated_label_loss
                    ponderated_label_loss_value = ponderated_label_loss.item()
                else:
                    ponderated_label_loss_value = 0.0
                if epoch % config_params.plot_every_epoch == 0:
                    label_loss_history.append(
                        (epoch, ponderated_label_loss_value)
                    )

            if config_params.lambda_uncert:
                if (
                    epoch > config_params.n_epochs_uncert
                    and config_params.lambda_uncert > 0
                ):
                    ponderated_uncert_loss = (
                        config_params.lambda_uncert * uncertainty_loss(outputs)
                    )
                    loss += ponderated_uncert_loss
                    ponderated_uncert_loss_value = ponderated_uncert_loss.item()
                    if epoch % config_params.plot_every_epoch == 0:
                        uncert_loss_history.append(
                            (epoch, ponderated_uncert_loss_value)
                        )
                else:
                    ponderated_uncert_loss_value = 0.0
                    if epoch % config_params.plot_every_epoch == 0:
                        uncert_loss_history.append(
                            (epoch, ponderated_uncert_loss_value)
                        )

            # THIRD LOSS -------------------------------------------------------
            # at this step, we're sure - at 99% - that some points are water and some are ground.
            # We separate them and build their distributions of wse for each time/date.
            # Then, we use a KL-Loss to make the predicted distribution as close as
            # possible to the "experimental" distribution.
            # we have as many KL Losses as we have time/date

            if epoch == config_params.n_epochs_wloss:
                # for this Loss, we change the optimizer and learning rate
                # from now on, this will update only the weights of the WATER MODEL
                # since the values are smaller, we have to take a smaller lr
                # the name and learning rate are defined by the user in the config file
                wl_optimizer_name = config_params.wl_optimizer_name
                wl_lr = float(config_params.wl_lr)
                self.optimizer = self.load_optimizer(
                    self.model.water_net.parameters(), wl_optimizer_name, wl_lr
                )

            if (
                epoch > config_params.n_epochs_wloss
                and config_params.lambda_wloss > 0
            ):
                thresh = config_params.confidence_threshold
                this_water_loss = water_loss(
                    outputs, targets, self.dataset.t_sizes, thresh
                )
                ponderated_water_loss = (
                    config_params.lambda_wloss * this_water_loss
                )
                loss += torch.sum(ponderated_water_loss, dim=0)
                if epoch % config_params.plot_every_epoch == 0:
                    water_loss_value = (
                        torch.log(ponderated_water_loss)
                        .flatten()
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    water_loss_history.append((epoch, water_loss_value))
            else:
                if epoch % config_params.plot_every_epoch == 0:
                    water_loss_value = np.zeros(len(self.dataset.t_sizes))
                    water_loss_history.append(
                        (epoch, water_loss_value)
                    )  # la Loss est calculée sur chacun des temps

            loss.backward()

            # Adjust weights
            self.optimizer.step()

            # plot training evolution
            if (epoch != 0 and epoch % config_params.plot_every_epoch == 0) or (
                epoch == config_params.n_epochs - 1
            ):
                logging.info(f"Epoch n°{epoch}: - - - - - - - - - - - - - ")
                ratio_for_print = [round(water_ratio, 2)]
                logging.info("Water Ratio %s ", ratio_for_print)

                # print(f"Optimizer {self.optimizer}")
                # print("LUT values", self.model.water_net.weight)

                if type(self.model.water_net.weight) is not int:  # lut
                    mu_water = (
                        torch.sigmoid(self.model.water_net.weight[:, 0])
                        .cpu()
                        .detach()
                    )
                    mu_water_m = (
                        mu_water * (self.dataset.max_wse - self.dataset.min_wse)
                        + self.dataset.min_wse
                    )

                    sigma_water = (
                        torch.exp(self.model.water_net.weight[:, 1])
                        .cpu()
                        .detach()
                    )
                    sigma_water_m = sigma_water * (
                        self.dataset.max_wse - self.dataset.min_wse
                    )

                    mu_water_for_print = [
                        round(elem, 2)
                        for elem in mu_water_m.cpu().detach().numpy()
                    ]
                    sigma_water_for_print = [
                        round(elem, 2)
                        for elem in sigma_water_m.cpu().detach().numpy()
                    ]
                    logging.info("Predicted mu_water : %s", mu_water_for_print)
                    logging.info(
                        "Predicted sigma_water : %s", sigma_water_for_print
                    )

                # print Losses
                nll_losses_for_print = [round(loss.item(), 2)]
                logging.info("NLL Loss %s", nll_losses_for_print)
                if config_params.lambda_label:
                    logging.info(
                        "Label Loss %s", round(ponderated_label_loss_value, 2)
                    )
                if config_params.lambda_uncert:
                    logging.info(
                        "Uncertainty Loss %s",
                        round(ponderated_uncert_loss_value, 2),
                    )
                logging.info("%s points with p_w>0.99", nb_points_pw_099)
                logging.info("Water Loss %s", water_loss_value)
                # Water Loss is a vector of 7 elements : one loss per date

                # # plot losses
                # # METHOD 1 : all losses on the same plots
                # (a,) = plt.plot(water_ratio_history, "b")
                # (b,) = plt.plot(nll_loss_history, "k")
                # # i have to it it this way, otherwise there is a bug of color in the legend...
                # if epoch <= config_params.n_epochs_uncert:
                #     plt.legend([a, b], ["water ratio", "NLL loss"])

                # if epoch > config_params.n_epochs_uncert:
                #     (c,) = plt.plot(uncert_loss_history, "r")
                #     plt.legend([a, b, c], ["water ratio", "NLL loss", "uncert loss"])

                # if epoch > config_params.n_epochs_wloss:
                #     # there a as many lines as number of times
                #     # we need to create a list to unpack
                #     list_to_unpack = [0] * config_params.water_model_params[0]
                #     list_to_unpack = plt.plot(water_loss_history, "g")
                #     plt.legend(
                #         [a, b, c, list_to_unpack[0]],
                #         ["water ratio", "NLL loss", "uncert loss", "water loss"],
                #     )
                # plt.xlabel("epochs")
                # plt.ylim([-10, 2])
                # plt.title("Model progress")

                # METHOD 2 : each indicator has its own plot
                # unpack history
                ratio_points_pw_099_history_np = [
                    (i, v.cpu().item()) for i, v in ratio_points_pw_099_history
                ]
                ratio_points_pw_095_history_np = [
                    (i, v.cpu().item()) for i, v in ratio_points_pw_095_history
                ]
                ratio_points_pw_010_history_np = [
                    (i, v.cpu().item()) for i, v in ratio_points_pw_010_history
                ]

                epochs_water_ratios, water_ratios = zip(*water_ratio_history)
                epochs_ratio_points_pw_099, ratio_points_pw_099 = zip(
                    *ratio_points_pw_099_history_np
                )
                epochs_ratio_points_pw_095, ratio_points_pw_095 = zip(
                    *ratio_points_pw_095_history_np
                )
                epochs_ratio_points_pw_010, ratio_points_pw_010 = zip(
                    *ratio_points_pw_010_history_np
                )
                epochs_nll_losses, nll_losses = zip(*nll_loss_history)
                if config_params.lambda_label:
                    epochs_label_losses, label_losses = zip(*label_loss_history)
                if config_params.lambda_uncert:
                    epochs_uncert_losses, uncert_losses = zip(
                        *uncert_loss_history
                    )
                epochs_water_losses, water_losses = zip(*water_loss_history)

                fig, axs = plt.subplots(5, 1, figsize=(10, 10))

                axs[0].plot(epochs_water_ratios, water_ratios, "b", linewidth=1)
                axs[0].set_title(
                    f"Water ratio (=0 only water, =1 only ground) = "
                    f"{round(np.mean(water_ratio), 2)} for epoch {epoch}"
                )
                axs[0].spines["top"].set_visible(False)
                axs[0].spines["right"].set_visible(False)
                axs[0].spines["bottom"].set_visible(False)
                axs[0].set_xticks([])
                axs[0].set_ylim(0, 1)
                axs[0].set_yticks([0, 0.5, 1])

                axs[1].plot(
                    epochs_ratio_points_pw_099,
                    ratio_points_pw_099,
                    "r",
                    linewidth=1,
                )
                axs[1].plot(
                    epochs_ratio_points_pw_095,
                    ratio_points_pw_095,
                    "g",
                    linewidth=1,
                )
                axs[1].plot(
                    epochs_ratio_points_pw_010,
                    ratio_points_pw_010,
                    "b",
                    linewidth=1,
                )
                axs[1].set_title(
                    f"Classification - {nb_points_pw_010} points with p_w<0.10 (blue) \n "
                    f"{nb_points_pw_099} points >0.99 (red) - {nb_points_pw_095} points >0.95 (green)"
                )
                axs[1].spines["top"].set_visible(False)
                axs[1].spines["right"].set_visible(False)
                axs[1].spines["bottom"].set_visible(False)
                axs[1].set_xticks([])
                axs[1].set_ylim(0, 1)
                axs[1].set_yticks([0, 0.5, 1])

                axs[2].plot(epochs_nll_losses, nll_losses, "k", linewidth=1)
                axs[2].set_title("log(NLL Loss)")
                axs[2].spines["top"].set_visible(False)
                axs[2].spines["right"].set_visible(False)
                axs[2].spines["bottom"].set_visible(False)
                axs[2].set_xticks([])
                # axs[2].set_ylim(-4, 2)

                if config_params.lambda_label:
                    axs[3].plot(
                        epochs_label_losses, label_losses, "r", linewidth=1
                    )
                    axs[3].set_title("Label Loss")
                    axs[3].spines["top"].set_visible(False)
                    axs[3].spines["right"].set_visible(False)
                    axs[3].spines["bottom"].set_visible(False)
                    axs[3].set_xticks([])
                    # axs[3].set_ylim(-0.25, 1.25)
                    # axs[3].set_yticks([0, 0.5, 1])
                if config_params.lambda_uncert:
                    axs[3].plot(
                        epochs_uncert_losses, uncert_losses, "r", linewidth=1
                    )
                    axs[3].set_title("Uncertainty Loss")
                    axs[3].spines["top"].set_visible(False)
                    axs[3].spines["right"].set_visible(False)
                    axs[3].spines["bottom"].set_visible(False)
                    axs[3].set_xticks([])

                axs[4].plot(epochs_water_losses, water_losses, "g", linewidth=1)
                axs[4].set_title("log(Water Losses)")
                axs[4].spines["top"].set_visible(False)
                axs[4].spines["right"].set_visible(False)
                axs[4].set_ylim(-20, 5)

                axs[4].set_xlabel("Epoch")
                if config_params.lambda_uncert:
                    if epoch > config_params.n_epochs_uncert:
                        axs[0].axvline(
                            x=config_params.n_epochs_uncert, color="r", ls=":"
                        )
                        axs[1].axvline(
                            x=config_params.n_epochs_uncert, color="r", ls=":"
                        )
                        axs[2].axvline(
                            x=config_params.n_epochs_uncert, color="r", ls=":"
                        )
                        axs[3].axvline(
                            x=config_params.n_epochs_uncert, color="r", ls=":"
                        )
                        axs[4].axvline(
                            x=config_params.n_epochs_uncert, color="r", ls=":"
                        )

                if epoch > config_params.n_epochs_wloss:
                    axs[0].axvline(
                        x=config_params.n_epochs_wloss, color="g", ls=":"
                    )
                    axs[1].axvline(
                        x=config_params.n_epochs_wloss, color="g", ls=":"
                    )
                    axs[2].axvline(
                        x=config_params.n_epochs_wloss, color="g", ls=":"
                    )
                    axs[3].axvline(
                        x=config_params.n_epochs_wloss, color="g", ls=":"
                    )
                    axs[4].axvline(
                        x=config_params.n_epochs_wloss, color="g", ls=":"
                    )

                fig.tight_layout(pad=1.0)

                plt.savefig(
                    self.saving_path
                    / (self.model.description + f"_epoch{epoch}.png")
                )
                plt.close()

                # TO MONITOR TRAINING -------------------------------------------
                # distribution of self-entropy
                # self_entropy = -proba_water * torch.log(proba_water + 10e-18)
                # self_entropy_np = self_entropy.cpu().detach().numpy()

                # # plot distribution of p_w
                # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                # axs[0].hist(
                #     proba_water.cpu().detach().numpy(),
                #     bins=np.linspace(0, 1, 50),
                #     density=True,
                # )
                # axs[0].set_title(f"Distribution of P_w - epoch {epoch}")

                # axs[1].hist(
                #     self_entropy_np,
                #     bins=np.linspace(0, 1, 50),
                #     density=True,
                # )
                # axs[1].set_title(
                #     f"Distribution of self-entropy -pw.log(pw) - epoch {epoch}"
                # )

                # plt.savefig(
                #     self.saving_path
                #     / (self.model.description + f"_epoch{epoch}_P_W.png")
                # )
                # plt.close()
                # --------------------------------------------------------------
                if epoch == (config_params.n_epochs - 1):
                    final_epochs = config_params.n_epochs - 1
                    logging.info("Save number points + values losses")
                    results = {
                        f"water_ratio_epoch_{final_epochs}": water_ratio,
                        "nb_points_pw_010": nb_points_pw_010.item(),
                        "nb_points_pw_099": nb_points_pw_099.item(),
                        "nb_points_pw_095": nb_points_pw_095.item(),
                        "nll_losses": nll_losses[-1],
                    }

                    if config_params.lambda_label:
                        results.update({"label_losses": label_losses[-1]})
                    if config_params.lambda_uncert:
                        results.update({"uncert_losses": uncert_losses[-1]})

                    results.update({"water_losses": str(water_losses[-1])})
                    out = Path(self.saving_path).resolve().parent.parent
                    path_result = Path(out) / "report.csv"
                    items = list(results.items())
                    df = pd.DataFrame(items, columns=["key", "value"])
                    df.to_csv(path_result, mode="w", header=False)

                if epoch in config_params.save_model_epoch:
                    self.save_model(
                        self.saving_path
                        / (self.model.description + f"_epoch{epoch}.pth"),
                        epoch,
                        config_params,
                    )

        self.save_model(
            self.saving_path / (self.model.description + f"_epoch{epoch}.pth"),
            epoch,
            config_params,
        )

        training_time = (time.time() - start_time) / 60
        logging.info("Training time took: %s minutes", training_time)
