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
Model class
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.cauchy import Cauchy
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal
from torch.nn import Embedding

MODE_TEST = os.getenv("MODE_TEST") or False


def freq_enc(x, freq_enc_dim):
    """."""
    if freq_enc_dim > 0:
        cos = torch.stack(
            [torch.cos((2**N) * np.pi * x) for N in range(freq_enc_dim)], 0
        )
        sin = torch.stack(
            [torch.sin((2**N) * np.pi * x) for N in range(freq_enc_dim)], 0
        )
        x = torch.cat((x[None, ...], cos, sin), 0)
        x = torch.reshape(x.transpose(0, 1), (x.shape[1], -1))
    if MODE_TEST:
        isinstance(x, torch.Tensor)
    return x


class gaussian_net(nn.Module):
    """
    Gaussian network
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,  # hidden dimension
        dim_out,  # output dimension, Mu and Beta
        num_layers,  # number of layers
        final_activation,  # activation of final layer (nn.Identity() for direct output)
        sigma,  # different signals may require different sigma - this is a hyperparameter
        const_init=None,# None for random init, else torch.Tensor[dim_out] initial values
    ):
        super().__init__()
        self.init_layer = nn.Linear(dim_in, dim_hidden)
        self.layers = nn.ModuleList([self.init_layer])
        for _ in range(num_layers):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.layers.append(nn.Linear(dim_hidden, dim_out))
        self.final_activation = final_activation
        self.sigma = sigma
        if const_init != None:
            with torch.no_grad():
                self.layers[num_layers+1].weight.mul_(0.01)
                self.layers[num_layers+1].bias[:] = const_init

    def forward(self, x):
        """."""
        for i in range(len(self.layers) - 1):
            x = self.gaussian(((self.layers)[i])(x))
        x = self.final_activation(((self.layers)[-1])(x))
        if MODE_TEST:
            assert isinstance(x, torch.Tensor)
        return x

    def gaussian_init(self, x):
        """."""
        x_ = self.init_layer(x)
        mu = torch.mean(x_, axis=0).unsqueeze(0)
        out = (-0.5 * (mu - x_) ** 2 / self.sigma**2).exp()
        if MODE_TEST:
            assert isinstance(out, torch.Tensor)
        return out

    def gaussian(self, x):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        out = (-0.5 * (x) ** 2 / self.sigma**2).exp()
        if MODE_TEST:
            assert isinstance(out, torch.Tensor)
        return out


class diff_lut_model:
    def __init__(self, lut_size, output_dim):
        """."""
        self.lut_size = lut_size
        self.lut = Embedding(embedding_dim=output_dim, num_embeddings=lut_size)
        self.init_value = -10
        self.lut.weight = torch.nn.Parameter(
            self.init_value
            * torch.ones((lut_size, output_dim), dtype=torch.float)
        )

    def forward(self, x):
        """."""
        i = x * (self.lut_size - 1)  # x is torch.Size([494455])
        i_low = i.int()
        offset = torch.where(x == 1, 0, 1)
        fact = i - i_low
        fact = fact.unsqueeze(-1)
        out = self.lut(i_low) * (1 - fact) + self.lut(i_low + offset) * (fact)
        if MODE_TEST:
            assert isinstance(out, torch.Tensor)
        return out  # out is torch.Size([494455, 2])

    def parameters(self):
        """."""
        return [self.lut.weight]


class LUT_adapter:
    """."""

    def __init__(self, dim_in, dim_out):  # diff_lut_model(params.n_dates, 1)
        self.diff_lut_model = diff_lut_model(dim_in, dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        """."""
        out = self.diff_lut_model.forward(x)
        return out

    def parameters(self):
        """."""
        return [self.diff_lut_model.lut.weight]

    def to(self, device):
        """."""
        return self.diff_lut_model.lut.to(device)

    @property
    def weight(self):
        """."""
        return self.diff_lut_model.lut.weight

    @weight.setter
    def weight(self, value):
        """."""
        self.diff_lut_model.lut.weight = value  # Set the weight properly


class MLP(nn.Module):
    """
    Basic MLP
    self.dim_in
    self.dim_out

    self.layers
    self.activation
    self.final_activation
    """

    def __init__(
        self,
        dim_in=2,  # input dimension
        dim_hidden=32,  # hidden dimension
        dim_out=1,  # output dimension
        num_layers=4,  # number of layers
        activation=F.relu, # activation of hidden layers
        final_activation=torch.sigmoid,# activation of final layer (nn.Identity() for direct output)
        const_init=None,# None for random init, else torch.Tensor[dim_out] initial values
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = 0  # to be defined

        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_hidden)])
        for i in range(num_layers):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.layers.append(nn.Linear(dim_hidden, dim_out))
        self.activation = activation
        self.final_activation = final_activation
        if const_init != None:
            with torch.no_grad():
                self.layers[num_layers+1].weight.mul_(0.01)
                self.layers[num_layers+1].bias[:] = const_init

    def forward(self, x):
        """."""
        if self.dim_in > 2:
            freq_enc_dim = int((self.dim_in - 2) / 4)
            x = freq_enc(x, freq_enc_dim)

        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        for i in range(len(self.layers) - 1):
            x = self.activation(((self.layers)[i])(x))
        x = self.final_activation(((self.layers)[-1])(x))
        if MODE_TEST:
            assert isinstance(x, torch.Tensor)
        return x


def init_two_branch_model(config_params, output_dim):
    """
    initialise GROUND branch based on config parameters
    """
    # ground_model_params: Optional[list] = []  # FIRST width, SECOND depth, THIRD freq_enc_dim
    w1 = config_params.ground_model_params[0]  # width = hidden dimension
    d1 = config_params.ground_model_params[1]  # depth = number of layers

    ground_description = ""
    # Initialization of the ground network
    if (config_params.ground_model_mean_init != None):
        #We take the logits because its the inverse of the sigmoid activation function used later
        ground_model_mean_init = torch.logit(torch.Tensor([config_params.ground_model_mean_init]))
        if output_dim == 1: #Non Bayesian case
            ground_init = ground_model_mean_init
        elif output_dim == 2: #Bayesian : Initialize both mean and variance
            if config_params.ground_model_var_init is None: 
                raise ValueError("ground_model_var_init not defined")
            else: 
                ground_model_var_init = torch.log(torch.Tensor([config_params.ground_model_var_init]))
                ground_init = torch.cat((ground_model_mean_init, ground_model_var_init)) 
    
    if config_params.ground_model.lower() == "mlp":
        # input dimension = X, Y + 4 * freq_enc_dim
        inDim = 2 + 4 * config_params.ground_model_params[2]
        ground_net = MLP(
            dim_in=inDim,
            dim_hidden=w1,
            dim_out=output_dim,
            num_layers=d1,
            activation=F.relu,
            final_activation=nn.Identity(),
            const_init=ground_init,
        )
        ground_description = f"GROUND-mlp-w-{w1}-d-{d1}-inDim-{inDim}"

    elif config_params.ground_model.lower() == "garf":
        if len(config_params.ground_model_params) == 3:
            gauss_act_sigma = config_params.ground_model_params[2]
        else:
            raise ValueError("gauss_act_sigma not defined")

        ground_net = gaussian_net(
            dim_in=2,  # 2 input dimensions (x and y)
            dim_hidden=w1,
            dim_out=output_dim,  # output dimension, Mu and Sigma
            num_layers=d1,
            final_activation=nn.Identity(),
            sigma=gauss_act_sigma,
            const_init=ground_init,
        )
        str_gauss_act_sigma = str(gauss_act_sigma).replace(".", "-")
        ground_description = (
            f"GROUND-garf-w-{w1}-d-{d1}-sigma-" + str_gauss_act_sigma
        )
    else:
        raise ValueError("Ground model not defined")

    # initialise WATER branch based on config parameters  -----------------
    # Initialization of the water network
    if (config_params.water_model_mean_init != None):
        #We take the logits because its the inverse of the sigmoid activation function used later
        water_model_mean_init = torch.logit(torch.Tensor([config_params.water_model_mean_init]))
        if output_dim == 1: #Non Bayesian case
            water_init = water_model_mean_init
        elif output_dim == 2: #Bayesian : Initialize both mean and variance
            if config_params.water_model_var_init is None:
                raise ValueError("water_model_var_init not defined")
            else:
                water_model_var_init = torch.log(torch.Tensor([config_params.water_model_var_init]))
                water_init = torch.cat((water_model_mean_init, water_model_var_init))     
    if config_params.water_model.lower() == "mlp":
        w2 = config_params.water_model_params[0]  # hidden dimension
        d2 = config_params.water_model_params[1]  # number of layers
        water_net = MLP(
            dim_in=1,  # input dimension = t
            dim_hidden=w2,
            dim_out=output_dim,
            num_layers=d2,
            activation=F.relu,
            final_activation=nn.Identity(),
            const_init=water_init,
        )
        water_description = f"WATER-mlp-w-{w2}-d-{d2}"
    elif config_params.water_model.lower() == "lut":
        n_dates = config_params.water_model_params[0]
        water_net = LUT_adapter(n_dates, output_dim)
        water_description = f"WATER-lut-ndates-{n_dates}"
    else:
        raise ValueError("Water model not defined")

    description = ground_description + "_" + water_description
    if MODE_TEST:
        assert isinstance(ground_net, gaussian_net)
        assert isinstance(water_net, MLP)
        assert isinstance(description, str)
    return ground_net, water_net, description


class StandardTwoBranchModel(nn.Module):
    """
    This is a modified MLP with two separate branches. One branch takes spatial
    coordinates and generates the ground height at that point (this is either
    the bottom of the lake or local land depending on water height). The other
    branch takes the time coordinate for the acquisition, and returns the
    corresponding water height. After the branches generate a h_g (ground height)
    and h_w (water height), the maximum of the two is selected and becomes the
    final output for an input of (x, y, t).
    """

    def __init__(self, config_params):
        """
        The constructor for the aforementioned model.
        self.ground_net
        self.water_net

        output_dim :
        # classic model : 1 output dimension = h
        # bayesian model : 2 output dimension = mu, sigma
        """
        super().__init__()
        output_dim = 1
        self.ground_net, self.water_net, self.description = (
            init_two_branch_model(config_params, output_dim)
        )
        self.description = "standard_" + self.description

    def parameters(self):
        all_params = [p for p in self.ground_net.parameters()]
        all_params.extend(self.water_net.parameters())
        if MODE_TEST:
            assert isinstance(all_params, list)
        return all_params

    def forward(self, coords: torch.Tensor, config_params):
        """
        coords : Tensor of size [batch_size, 3]
        """

        ######### Forward loop for ground height.
        x = coords[:, :2]  # torch.Size([494455, 2]) float32
        ground_feat = self.ground_net.forward(x)
        hg = torch.sigmoid(ground_feat)  # torch.Size([494455, 1])

        ########## Forward loop for water height
        t = coords[:, 2]
        water_feat = self.water_net.forward(t)
        hw = torch.sigmoid(water_feat)  # torch.Size([494455, 1])

        ########## Combining hg and hw outputs
        if self.training:
            combined = torch.cat((hg, hw), 1)
            max_label = F.softmax(
                config_params.softmax_factor * combined, dim=1
            )
            mult = torch.mul(combined, max_label)
            out = (torch.sum(mult, 1)).reshape((x.shape)[0], 1)
        else:
            max_label = torch.where(hg > hw, 1, 0)
            out = torch.max(hg, hw)
        if MODE_TEST:
            assert isinstance(out, torch.Tensor)
            assert isinstance(max_label, torch.Tensor)
            assert isinstance(hg, torch.Tensor)
            assert isinstance(hw, torch.Tensor)
        return {"h": out, "max_label": max_label, "hg": hg, "hw": hw}


class BayesianTwoBranchModel(nn.Module):
    """
    This is a modified MLP with two separate branches. One branch takes spatial
    coordinates and generates the ground height AND VARIANCE at that point (this is either
    the bottom of the lake or local land depending on water height). The other
    branch takes the time coordinate for the acquisition, and returns the
    corresponding water height AND VARIANCE. After the branches generate a h_g (ground height)
    and h_w (water height) AND THEIR VARIANCES, the maximum of the two is selected and becomes the
    final output for an input of (x, y, t).
    """

    def __init__(self, config_params):
        """
        The constructor for the aforementioned model.
        self.ground_net
        self.water_net

        """
        # bayesian model :
        super().__init__()
        output_dim = 2
        self.ground_net, self.water_net, self.description = (
            init_two_branch_model(config_params, output_dim)
        )
        self.description = "bayesian_" + self.description

    def parameters(self):
        all_params = [p for p in self.ground_net.parameters()]
        all_params.extend(self.water_net.parameters())
        if MODE_TEST:
            isinstance(all_params, list)
        return all_params

    # overwite forward
    def forward(self, coords: torch.Tensor, config_params):
        """
        coords : Tensor of size [batch_size, 3]
        """

        ######### Forward loop for ground.
        x = coords[:, :2]
        ground_feat = self.ground_net.forward(x)

        # we want the mean to be between 0 and 1 so we put a sigmoid as final activation
        # We want the variance to be between 0 and inf so we use an exponential
        mu_g = torch.sigmoid(ground_feat[:, 0])
        sigma_g = torch.exp(ground_feat[:, 1]) + 0.0001

        ########## Forward loop for water
        t = coords[:, 2]
        water_feat = self.water_net.forward(t)

        # we want the mean to be between 0 and 1 so we put a sigmoid as final activation
        # We want the variance to be between 0 and inf so we use an exponential
        mu_w = torch.sigmoid(water_feat[:, 0])
        sigma_w = torch.exp(water_feat[:, 1]) + 0.0001

        ########## Combining hg and hw outputs
        # Max label indicates where the model thinks ground is above water :
        # the basic binary water mask
        max_label = torch.where(mu_g > mu_w, 1, 0)

        # We also compute the probability of water.
        # This is the cumulated probability of the predicted water height under the local ground distribution
        if config_params.ground_prob.lower() == "gaussian":
            Ng = Normal(mu_g, sigma_g)
        elif config_params.ground_prob.lower() == "cauchy":
            Ng = Cauchy(mu_g, sigma_g)
        elif config_params.ground_prob.lower() == "laplace":
            Ng = Laplace(mu_g, sigma_g)
        else:
            print("This probability distribution is not implemented")
        p_w = Ng.cdf(mu_w)

        # The final inference value is the max of the two means
        out_z = torch.max(mu_g, mu_w)
        if MODE_TEST:
            assert isinstance(out_z, torch.Tensor)
            assert isinstance(max_label, torch.Tensor)
            assert isinstance(mu_g, torch.Tensor)
            assert isinstance(mu_w, torch.Tensor)
            assert isinstance(sigma_g, torch.Tensor)
            assert isinstance(sigma_w, torch.Tensor)
            assert isinstance(p_w, torch.Tensor)
        return {
            "h": out_z,
            "max_label": max_label,
            "hg": mu_g,
            "hw": mu_w,
            "vg": sigma_g,
            "vw": sigma_w,
            "pw": p_w,
        }
