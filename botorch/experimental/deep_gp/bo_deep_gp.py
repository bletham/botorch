#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.utils import draw_sobol_normal_samples, draw_sobol_samples
from botorch.experimental.deep_gp.deep_gp import DeepGPModel, fit_deep_gp, pred_deep_gp
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from torch.distributions import Normal


def bo_deepgp(
    f: Callable,
    x_bounds: torch.Tensor,
    num_init: int = 1000,
    num_iter: int = 9000,
    device: torch.device = None,
    exploit_ratio: float = 0.5,
    exploit_sd: float = 0.01,
    num_cand: int = 50000,
    num_best: int = 10,
    dgp_layers: Optional[List[int]] = None,
    num_samples: int = 10,
    noise_level: Optional[float] = 0.01,
    refit_freq: Optional[List[int]] = None,
    num_epochs: Optional[List[int]] = None,
    reweight: bool = False,
    reweight_hyper: float = 0.001,
    print_freq: int = 100,
) -> Dict[str, Union[str, Optional[torch.Tensor]]]:
    """
    Implements deep Gaussian processes model for Bayesian optimization with
    single objective.

    f must be a minimization problem.
    """
    # hyper-parameter settings
    x_bounds, dim, dgp_layers, refit_freq, num_epochs, device = get_hyper_parameters(
        x_bounds, dgp_layers, refit_freq, num_epochs, device
    )
    print(f"Using device: {device}")

    # draw initial samples
    x = draw_sobol_samples(x_bounds, num_init, 1).squeeze(1).to(device)
    y = f(x).to(device)

    # calculate how many candidates are randomly sampled in the whole feature space
    # and nearby current "optimal" designs
    num_cand_explore, num_cand_exploit = (
        math.ceil(num_cand * (1 - exploit_ratio)),
        math.ceil(num_cand * exploit_ratio),
    )
    num_cand_exploit_each = math.ceil(num_cand_exploit / num_best)
    num_cand_exploit = num_cand_exploit_each * num_best

    best_so_far = torch.empty(num_iter + 1).to(device)
    best_so_far[0] = min(y)
    running_time = []
    skip_connect = [False] * len(dgp_layers)
    weights = None
    noise_fixed = False if noise_level is None else True

    for i in range(num_iter):
        start_time = time.time()

        # normalize objectives
        y_mean = torch.mean(y)
        y_sd = torch.std(y)
        y_norm = (y - y_mean) / y_sd

        # find the current optimal designs
        _, best_ix = torch.topk(y, num_best, largest=False)
        x_centroid = x[best_ix, :]

        if i % refit_freq[0] == 0:
            # full refresh of the model
            if reweight:
                # calculate weights of samples
                weights = get_sample_weights(y, reweight_hyper, device)

            # deep GP model specification
            model = DeepGPModel(
                x.shape[-1], dgp_layers, skip_connect, inducing_bounds=x_bounds
            )

            # likelihood specification
            if noise_level is None:
                likelihood = GaussianLikelihood()
            else:
                likelihood = GaussianLikelihood(noise_constraint=GreaterThan(0))
                likelihood.initialize(noise=noise_level)

            # fit deep GP model
            fit_deep_gp(
                model,
                likelihood,
                x,
                y_norm,
                weights=weights,
                noise_fixed=noise_fixed,
                num_epochs=num_epochs[0],
                num_samples=num_samples,
                device=device,
            )

        elif i % refit_freq[1] == 0:
            # Update current model

            if reweight:
                weights = get_sample_weights(y, reweight_hyper, device)

            fit_deep_gp(
                model,
                likelihood,
                x,
                y_norm,
                weights=weights,
                noise_fixed=noise_fixed,
                num_epochs=num_epochs[1],
                num_samples=num_samples,
                device=device,
            )

        # generate candidates that are randomly drawn from the whole feature space
        # (exploration)
        x_cand_explore = (
            draw_sobol_samples(x_bounds, num_cand_explore, 1).squeeze(1).to(device)
        )

        # generate candidates that are randomly drawn nearby current optimal designs
        # (exploitation)
        x_cand_exploit_raw = (
            draw_sobol_normal_samples(dim, num_cand_exploit) * exploit_sd
        )
        x_cand_exploit_raw = x_cand_exploit_raw.to(device=device)
        x_cand_exploit = (
            x_centroid.repeat(num_cand_exploit_each, 1) + x_cand_exploit_raw
        )

        # guarantee all the candidates are in the bounds of features
        x_cand = back_to_bound(
            torch.cat([x_cand_explore, x_cand_exploit], 0), bounds=x_bounds, dim=dim
        )

        # get predictive means and covariances of candidates from the deep GP model
        x_cand_mean_raw, x_cand_var_raw, _ = pred_deep_gp(
            model, likelihood, x_cand, device=device
        )

        # get means and standard deviations of candidates
        x_cand_mean, x_cand_sd = get_mean_and_sd(
            x_cand_mean_raw,
            x_cand_var_raw,
            raw=True,
            sample_mean=y_mean,
            sample_sd=y_sd,
        )

        # evaluate EI of samples
        x_cand_ei = EI(x_cand_mean, x_cand_sd, best_so_far[i])

        # get the best candidate
        _, x_new_ix = torch.topk(x_cand_ei, 1)
        x_new = x_cand[x_new_ix, :]
        # evaluate its function value
        y_new = f(x_new).to(device)

        x = torch.cat((x, x_new))
        y = torch.cat((y, y_new))

        best_so_far[i + 1] = min(best_so_far[i], y_new)

        running_time.append(time.time() - start_time)

        if i % print_freq == 0:
            print(i, best_so_far[i + 1])

    output = {
        "x_history": x.cpu(),
        "y_history": y.cpu(),
        "running_time": running_time,
        "best_so_far": best_so_far.cpu(),
    }

    return output


def get_hyper_parameters(
    x_bounds: torch.Tensor,
    dgp_layers: List[int],
    refit_freq: List[int],
    num_epochs: List[int],
    device: str,
) -> Tuple[torch.Tensor, List[int], List[int], List[int], torch.device]:
    if dgp_layers is None:
        dgp_layers = [4, 4]
    if refit_freq is None:
        refit_freq = [500, 1000000]
    if num_epochs is None:
        num_epochs = [2000, 100]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = x_bounds.shape[1]
    return (
        x_bounds.to(device),
        dim,
        dgp_layers,
        refit_freq,
        num_epochs,
        torch.device(device),
    )


def back_to_bound(x: torch.Tensor, bounds: torch.Tensor, dim: int) -> torch.Tensor:
    for i in range(dim):
        x[x[:, i] < bounds[0, i], i] = bounds[0, i]
        x[x[:, i] > bounds[1, i], i] = bounds[1, i]
    return x


def get_mean_and_sd(
    raw_mean: torch.Tensor,
    raw_variance: torch.Tensor,
    raw: bool = True,
    sample_mean: torch.Tensor = None,
    sample_sd: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        calculate mean and standard deviations of candidates under linear combinations
        given the predictive means and covariances provided by deep GP model

        adopt moment matching method to approximate means and standard deviations
    """
    if raw:
        # if means and standard deviations of candidates in the raw scale are needed
        raw_mean = raw_mean * sample_sd + sample_mean
        raw_variance = raw_variance * sample_sd * sample_sd

    mean = raw_mean.mean(0)
    sd = torch.sqrt(raw_variance.mean(0) + (raw_mean ** 2).mean(0) - mean ** 2)

    return mean, sd


def EI(
    x_cand_mean: torch.Tensor, x_cand_sd: torch.Tensor, best_f: torch.Tensor
) -> torch.Tensor:
    u = -(x_cand_mean - best_f) / x_cand_sd
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = x_cand_sd * (updf + u * ucdf)

    return ei


def get_sample_weights(
    y: torch.Tensor, reweight_hyper: float, device: torch.device
) -> torch.Tensor:
    """
        calculate samples weights based on the distances between samples and the best
    """
    # get ranks of samples based on objectives
    num_sample = len(y)
    y_ranks = np.argsort(np.argsort(y.cpu().numpy()))
    y_weights_unnorm = torch.from_numpy(1 / (reweight_hyper * num_sample + y_ranks)).to(
        device
    )

    # normalize the weights
    y_weights = y_weights_unnorm / sum(y_weights_unnorm)

    return y_weights
