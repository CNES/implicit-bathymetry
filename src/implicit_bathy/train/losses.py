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
All losses used in train section
"""

import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.cauchy import Cauchy
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

MODE_TEST = os.getenv("MODE_TEST") or False


# For Standard (non bayesian) network -------------------------------------------
def mse_loss(outputs, targets):
    """."""
    return F.mse_loss(outputs["h"], targets)


# For Bayesian network
# First Loss --------------------------------------------------------------------
def nll_max(outputs, targets, config_params, weights=None):
    """
    Negative Log Likelihood loss function, the main loss for training Bayes models.

    outputs is the dictionary produced by the forward() of a bayes model
    it contains the results of an inference on Batch_size points

    targets is the z value as measured in the dataset, size is [Batch_size, 1]
    """
    # Ground mean
    mu_g = outputs["hg"][..., None]
    # Ground variance
    sigma_g = outputs["vg"][..., None]

    # Water mean
    mu_w = outputs["hw"][..., None]
    # Water variance
    sigma_w = outputs["vw"][..., None]

    # we set the water distribution as gaussian
    Nw = Normal(mu_w, sigma_w)

    # ground distribution can be chosen by the user (defined in params):
    distribution_map = {
        "gaussian": Normal,
        "cauchy": Cauchy,
        "laplace": Laplace,
    }
    ground_distribution_name = config_params.ground_prob.lower()
    if ground_distribution_name in distribution_map:
        Ng = distribution_map[ground_distribution_name](mu_g, sigma_g)
    else:
        logging.info(
            "This probability distribution for ground is not implemented"
        )

    # PDF of the model at z values given by targets.
    phi_g = torch.exp(Ng.log_prob(targets))
    phi_w = torch.exp(Nw.log_prob(targets))

    # CDF of the model at z values given by targets.
    Phi_g = Ng.cdf(targets)
    Phi_w = Nw.cdf(targets)

    # -log of the PDF of the max of two distributions
    # This value is lower when the z value is likely according to the max distribution
    nll = -torch.log(Phi_w * phi_g + Phi_g * phi_w + 10e-12)
    # nll -> torch.Size([494221, 1]) = one value per data point
    # nll = torch.mean(nll, dim=0)
    if MODE_TEST:
        assert isinstance(nll, torch.Tensor)
    return nll


# Second Loss --------------------------------------------------------------------
def uncertainty_loss(outputs, weights=None):
    """
    Uncertainty function for Bayes methods.
    The goal is to make the model more certain of its predictions.

    outputs is the dictionary produced by the forward() of a bayes model
    it contains the results of an inference on Batch_size points
    """
    p_w = outputs["pw"][..., None]
    # We minimize the self entropy aka -P log P
    # The goal is to encourage pw to be close to 0 or 1
    uncert = torch.mean(-p_w * torch.log(p_w + 10e-18))
    if MODE_TEST:
        assert isinstance(uncert, torch.Tensor)
    return uncert


# Third Loss --------------------------------------------------------------------
# We define the distance between two distrubitions as the KL loss.
# Note : this formula is only valid for gaussian distributions
# which is the case here as we are interested only in the water model for this loss
def kl_loss(h_w, v_w, mu_t, v_t):
    """KL loss"""
    kl = 0.5 * (
        (v_w / (v_t + 10e-18)) ** 2
        + ((mu_t - h_w) ** 2) / (v_w**2 + 10e-18)
        - 1
        + torch.log((v_t**2 + 10e-18) / (v_w**2 + 10e-18))
    )
    if MODE_TEST:
        assert isinstance(kl, torch.Tensor)
    return kl


def water_loss_bayesian(outputs, targets, t_sizes, thresh):
    """
    Water loss function for Bayes methods.
    The goal here is to align the probability distribution of the water model
    with the empirical measured mean
    and variance of the points detected with a high degree of certainty as water.

    outputs is the dictionary produced by the forward() of a bayes model
    it contains the results of an inference on Batch_size points
    """
    # separate outputs into tuples of len 7
    p_w = torch.split(outputs["pw"], t_sizes, dim=0)
    h_w = torch.split(outputs["hw"], t_sizes, dim=0)
    v_w = torch.split(outputs["vw"], t_sizes, dim=0)
    targets_t = torch.split(targets, t_sizes, dim=0)

    loss_list = [
        water_loss_monodate(p_w[i], h_w[i], v_w[i], targets_t[i], thresh)
        for i in range(len(t_sizes))
    ]
    w_loss = torch.stack(loss_list, dim=0)
    # print("h loss", [loss.item() for loss in loss_list])
    if MODE_TEST:
        assert isinstance(w_loss, torch.Tensor)
    return w_loss


def water_loss_monodate(p_w, h_w, v_w, targets, thresh):
    """Water loss"""
    mask = torch.where(p_w[..., None] >= thresh)

    if sum(p_w[..., None].cpu().detach().numpy() >= thresh)[0] == 0:
        logging.info("WARNING : NO POINTS WITH P_W > %s", thresh)

    tw = torch.index_select(targets, 0, mask[0])
    if tw.shape[0] > 0:
        mu_t = torch.mean(tw, dim=0)
        v_t = torch.std(tw, dim=0)
        # return torch.abs(h_w[0] - mu_t) + torch.abs(v_w[0] - v_t)
        this_kl_loss = kl_loss(h_w[0], v_w[0], mu_t, v_t)
        if np.isnan(this_kl_loss.cpu().detach().numpy())[0]:
            logging.info("WARNING : NaN in KL Loss.")
            logging.info(this_kl_loss)
        return this_kl_loss
    return torch.Tensor([0.0]).to(device)


def label_loss(p_w, target_labels):
    """
    Label loss is cross entropy
    """
    lloss = torch.log(F.binary_cross_entropy(p_w, target_labels) + 10e-6)
    if MODE_TEST:
        assert isinstance(lloss, torch.Tensor)
    return lloss


def chunk_meanstd(
    data: torch.Tensor,
    chunk_sizes: list[int],
    mask: torch.Tensor | None = None,
    expand: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and std per chunk in a 1D tensor

    Args:
        data (torch.Tensor): tensor of shape (N, 1)
        chunk_sizes (list[int]): list of chunk sizes
        mask (torch.Tensor | None, optional): optional boolean mask. Defaults to None.
        expand (bool, optional): expand the result to the shape of data. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: chunk_mean and chunk_std for each chunk
    """
    output_shape = (len(chunk_sizes), 1)

    # Map chunk ids
    chunk_ids = torch.repeat_interleave(
        torch.tensor(chunk_sizes)
    )  # eg. chunk_sizes = (3,2,4) -> torch.Tensor([0,0,0,1,1,2,2,2,2])
    chunk_ids = chunk_ids.unsqueeze(-1).to(device=data.device)

    # Mask of valid entries
    is_valid = (
        mask if mask is not None else torch.ones_like(data).to(dtype=bool)
    )

    # Sum of valid values per chunk
    sum_per_chunk = torch.zeros(output_shape, device=data.device)
    sum_per_chunk = sum_per_chunk.scatter_add(
        0, chunk_ids, torch.where(is_valid, data, 0.0)
    )

    # Count of valid values per chunk
    count_per_chunk = torch.zeros(output_shape, device=data.device)
    count_per_chunk = count_per_chunk.scatter_add(
        0, chunk_ids, is_valid.float()
    )

    # Compute mean per chunk
    chunk_mean = sum_per_chunk / count_per_chunk.clamp(
        min=1
    )  # If a chunk has zero valid points, chunk_mean = 0

    # Sum of squared difference per chunk
    squared_diff = torch.square(data - chunk_mean[chunk_ids.squeeze()])
    sq_sum_per_chunk = torch.zeros(output_shape, device=data.device)
    sq_sum_per_chunk = sq_sum_per_chunk.scatter_add(
        0, chunk_ids, torch.where(is_valid, squared_diff, 0.0)
    )

    # Compute std per chunk
    chunk_std = torch.sqrt(
        sq_sum_per_chunk / (count_per_chunk - 1).clamp(min=1)
    )  # If a chunk has zero valid points, nanstd = 0

    # Chunks with invalid values only return nans
    chunk_mean = torch.where(count_per_chunk > 0, chunk_mean, torch.nan)
    chunk_std = torch.where(count_per_chunk > 1, chunk_std, torch.nan)

    if expand:
        return chunk_mean[chunk_ids.squeeze()], chunk_std[chunk_ids.squeeze()]
    if MODE_TEST:
        assert isinstance(chunk_mean, torch.Tensor)
        assert isinstance(chunk_std, torch.Tensor)
    return chunk_mean, chunk_std


def water_loss(outputs, targets, t_sizes, thresh):
    """
    Water loss function for Bayes methods.
    The goal here is to align the probability distribution
    of the water model with the empirical measured mean and
    variance of the points detected with a high degree of certainty as water.

    outputs is the dictionary produced by the forward() of a bayes model
    it contains the results of an inference on Batch_size points
    """
    # target water level where the water probability is above threshold
    water_mask = outputs["pw"].unsqueeze(-1) >= thresh

    # Compute mean and std of targets for each date for water points only
    mu_t, v_t = chunk_meanstd(targets, t_sizes, water_mask)

    # Get the predicted water levels (mean and std) for each date
    indices = torch.zeros(len(t_sizes), dtype=int)
    indices[1:] = torch.cumsum(torch.tensor(t_sizes), 0)[:-1]
    h_w = outputs["hw"][indices].unsqueeze(-1)
    v_w = outputs["vw"][indices].unsqueeze(-1)

    loss = kl_loss(h_w, v_w, torch.nan_to_num(mu_t), torch.nan_to_num(v_t))

    is_invalid = torch.logical_or(torch.isnan(mu_t), torch.isnan(v_t))
    if torch.sum(is_invalid) > 0:
        loss = torch.where(is_invalid, 0.0, loss)
    if MODE_TEST:
        assert isinstance(loss, torch.Tensor)
    return loss
