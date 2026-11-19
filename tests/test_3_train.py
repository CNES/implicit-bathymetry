#!/usr/bin/env python3

"""Train unit tests."""

from pathlib import Path
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from implicit_bathy.train.trainer import StandardTrainer, BayesianTrainer
from implicit_bathy.train.model import (
    freq_enc,
    gaussian_net,
    diff_lut_model,
    LUT_adapter,
    MLP,
    init_two_branch_model,
    StandardTwoBranchModel,
    BayesianTwoBranchModel,
)
from implicit_bathy.train.losses import (
    mse_loss,
    nll_max,
    uncertainty_loss,
    kl_loss,
    water_loss_bayesian,
    water_loss_monodate,
    label_loss,
    chunk_meanstd,
    water_loss,
)

from implicit_bathy.config.config import (
    BayesianTrainingConfig,
    StandardTrainingConfig,
)


# Train
@pytest.fixture
def config_params_standard(tmp_path):
    dataset_params = StandardTrainingConfig(
        seed=3,
        monitor_every_epoch=10,
        dataset_path=Path(
            (tmp_path) / "output" / "data" / "dataset_classes_1_2_3_4.pt"
        ),
        save_path=Path((tmp_path) / "output" / "train"),
        bayesian_model=False,
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
        softmax_factor=0.2,
    )
    return dataset_params


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


def test_load_optimizer(config_params_standard):
    params = [
        torch.nn.Parameter(torch.randn(10, 10)),
        torch.nn.Parameter(torch.randn(10)),
    ]
    t = StandardTrainer(config_params_standard)
    opt_adam = t.load_optimizer(params, "Adam", 0.005)
    opt_sgd = t.load_optimizer(params, "SGD", 0.005)
    opt_radam = t.load_optimizer(params, "RAdam", 0.005)
    opt_adamw = t.load_optimizer(params, "AdamW", 0.005)

    assert isinstance(opt_adam, torch.optim.Adam)
    assert isinstance(opt_sgd, torch.optim.SGD)
    assert isinstance(opt_radam, torch.optim.RAdam)
    assert isinstance(opt_adamw, torch.optim.AdamW)


def test_train_standard(config_params_standard, tmp_path):
    t = StandardTrainer(config_params_standard)
    t.train(config_params_standard)
    train_pth = Path(
        (tmp_path)
        / "output"
        / "train"
        / "standard_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4"
        / "standard_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4_epoch9999.pth"
    )
    assert train_pth.exists()


def test_train_bayesian(config_params_bayesian, tmp_path):
    t = BayesianTrainer(config_params_bayesian)
    t.train(config_params_bayesian)
    train_pth = Path(
        (tmp_path)
        / "output"
        / "train"
        / "bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4"
        / "bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4_epoch9999.pth"
    )
    assert train_pth.exists()


# Model.py
def test_freq_enc():
    x = torch.tensor([1, 0, 1, 0, 1])
    freq_enc_dim = 0
    r = freq_enc(x, freq_enc_dim)
    assert isinstance(r, torch.Tensor)


@pytest.fixture
def gaussian_net_class():
    ground_net = gaussian_net(
        dim_in=2,
        dim_hidden=16,
        dim_out=1,
        num_layers=4,
        final_activation=nn.Identity(),
        sigma=0.1,
    )
    return ground_net


def test_foward_gaussian_net(gaussian_net_class):
    x = torch.tensor([1.0, 0.0])
    r = gaussian_net_class.forward(x)
    assert isinstance(r, torch.Tensor)


def test_gaussian_init(gaussian_net_class):
    x = torch.tensor([1.0, 0.0])
    r = gaussian_net_class.gaussian_init(x)
    assert isinstance(r, torch.Tensor)


def test_gaussian(gaussian_net_class):
    x = torch.tensor([1.0, 0.0])
    r = gaussian_net_class.gaussian(x)
    assert isinstance(r, torch.Tensor)


@pytest.fixture
def diff_lut_model_class():
    difflut_model = diff_lut_model(lut_size=2, output_dim=2)
    return difflut_model


def test_foward_diff_lut_model(diff_lut_model_class):
    x = torch.tensor([1.0, 0.0])
    r = diff_lut_model_class.forward(x)
    assert isinstance(r, torch.Tensor)


def test_parameters_diff_lut_model(diff_lut_model_class):
    r = diff_lut_model_class.parameters()
    assert isinstance(r, list)


@pytest.fixture
def lut_adapter_class():
    lut_adapter_model = LUT_adapter(dim_in=2, dim_out=1)
    return lut_adapter_model


def test_foward_lut_adapter(lut_adapter_class):
    x = torch.tensor([1.0, 0.0])
    r = lut_adapter_class.forward(x)
    assert isinstance(r, torch.Tensor)


def test_parameters_lut_adapter(lut_adapter_class):
    r = lut_adapter_class.parameters()
    assert isinstance(r, list)


@pytest.fixture
def mlp_class():
    mlp_model = MLP(
        dim_in=2,  # input dimension
        dim_hidden=2,  # hidden dimension
        dim_out=1,  # output dimension
        num_layers=4,  # number of layers
        activation=F.relu,
        final_activation=torch.sigmoid,
    )
    return mlp_model


def test_foward_mlp(mlp_class):
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    r = mlp_class.forward(x)
    assert isinstance(r, torch.Tensor)


def test_init_two_branch_model(config_params_standard):
    ground_net, water_net, description = init_two_branch_model(
        config_params_standard, 1
    )
    assert isinstance(ground_net, gaussian_net)
    assert isinstance(water_net, MLP)
    assert isinstance(description, str)


@pytest.fixture
def standard_two_branch_model_class(config_params_standard):
    standard_two_branch_class = StandardTwoBranchModel(config_params_standard)
    return standard_two_branch_class


def test_parameters_standard_two_branch_model(standard_two_branch_model_class):
    all_param = standard_two_branch_model_class.parameters()
    assert isinstance(all_param, list)


def test_forward_standard_two_branch_model(
    standard_two_branch_model_class, config_params_standard
):
    coords = torch.tensor([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    r = standard_two_branch_model_class.forward(coords, config_params_standard)
    assert isinstance(r, dict)


@pytest.fixture
def bayesian_two_branch_model_class(config_params_bayesian):
    bayesian_two_branch_class = BayesianTwoBranchModel(config_params_bayesian)
    return bayesian_two_branch_class


def test_parameters_bayesian_two_branch_model(bayesian_two_branch_model_class):
    all_param = bayesian_two_branch_model_class.parameters()
    assert isinstance(all_param, list)


def test_forward_bayesian_two_branch_model(
    bayesian_two_branch_model_class, config_params_bayesian
):
    coords = torch.tensor([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    r = bayesian_two_branch_model_class.forward(coords, config_params_bayesian)
    assert isinstance(r, dict)


# losses.py
@pytest.fixture
def output_standard():
    output_standard = {
        "h": torch.tensor([[0.4828], [0.5635]]),
        "max_label": torch.tensor([[0.4967, 0.5033], [0.5043, 0.4957]]),
        "hg": torch.tensor([[0.4501], [0.6061]]),
        "hw": torch.tensor([[0.5151], [0.5202]]),
    }
    return output_standard


@pytest.fixture
def output_bayesian():
    output_bayesian = {
        "h": torch.tensor([0.5885, 0.5301]),
        "max_label": torch.tensor([1, 1]),
        "hg": torch.tensor([0.5885, 0.5301]),
        "hw": torch.tensor([0.4875, 0.4907]),
        "vg": torch.tensor([0.9941, 0.8576]),
        "vw": torch.tensor([1.0527, 1.1551]),
        "pw": torch.tensor([0.4678, 0.4854]),
    }
    return output_bayesian


def test_mse_loss(output_standard):
    targets = torch.tensor([[1.0], [0.0]])
    r = mse_loss(output_standard, targets)
    assert isinstance(r, torch.Tensor)


def test_nll_max(output_bayesian, config_params_bayesian):
    targets = torch.tensor([[1.0], [0.0]])
    r = nll_max(output_bayesian, targets, config_params_bayesian)
    assert isinstance(r, torch.Tensor)


def test_uncertainty_loss(output_bayesian):
    r = uncertainty_loss(output_bayesian)
    assert isinstance(r, torch.Tensor)


def test_kl_loss():
    h_w = 0.4875
    v_w = 1.0527
    mu_t = torch.tensor([1.0, 0.0])
    v_t = torch.tensor([1.0, 0.0])
    r = kl_loss(h_w, v_w, mu_t, v_t)
    assert isinstance(r, torch.Tensor)


def test_water_loss_bayesian(output_bayesian):

    targets = torch.tensor([[1.0], [0.0]])
    t_sizes = [1, 1]
    thresh = 0.5
    w_loss = water_loss_bayesian(output_bayesian, targets, t_sizes, thresh)
    assert isinstance(w_loss, torch.Tensor)


def test_water_loss_monodate():
    p_w = torch.tensor([[1.0], [0.0]])
    h_w = torch.tensor([[1.0], [0.0]])
    v_w = torch.tensor([[1.0], [0.0]])
    targets = torch.tensor([[1.0], [0.0]])
    thresh = 0.5
    r = water_loss_monodate(p_w, h_w, v_w, targets, thresh)
    assert isinstance(r, torch.Tensor)


def test_label_loss():
    p_w = torch.tensor([[1.0], [0.0]])
    target_labels = torch.tensor([[1.0], [0.0]])
    r = label_loss(p_w, target_labels)
    assert isinstance(r, torch.Tensor)


def test_chunk_meanstd():
    data = torch.tensor([[1.0], [0.0]])
    chunk_sizes = [1, 1]
    mask = torch.tensor([[True], [False]])
    chunk_mean, chunk_std = chunk_meanstd(data, chunk_sizes, mask)
    assert isinstance(chunk_mean, torch.Tensor)
    assert isinstance(chunk_std, torch.Tensor)


def test_water_loss(output_bayesian):
    targets = torch.tensor([[1.0], [0.0]])
    t_sizes = [1, 1]
    thresh = 0.5
    loss = water_loss(output_bayesian, targets, t_sizes, thresh)
    assert isinstance(loss, torch.Tensor)
