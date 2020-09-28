#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import gpytorch
import torch
import torch.nn as nn
import tqdm
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    MeanFieldVariationalDistribution,
    VariationalStrategy,
)
from scipy.cluster.vq import kmeans2
from torch.utils.data import DataLoader, TensorDataset


"""
Implement deep Gaussian processes (GP) regression

Inference is performed using stochastic variational inference following

Hugh Salimbeni and Marc Deisenroth. "Doubly stochastic variational
inference for deep Gaussian processes." Advances in Neural Information
Processing Systems. 2017.

Implementation is adapted from
https://gpytorch.readthedocs.io/en/latest/examples/05_Deep_Gaussian_Processes/
Deep_Gaussian_Processes.html
"""


class GPLayer(DeepGPLayer):
    """
        Define class for GP layers. As suggest by Salimbeni and Deisenroth (2017), a
        "skip connection" is used as in ResNet. This modification is done by
        overwritting __call__().
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_samples: int = 10,
        num_inducing: int = 100,
        variational_dist: str = "MeanField",
        mean_type: str = "linear",
        initialize_method: Optional[str] = None,
        inducing_bounds: Optional[torch.Tensor] = None,
        train_x: Optional[torch.Tensor] = None,
    ) -> None:
        """
            Args:
                input_dims:         input dimension of a layer
                output_dims:        output dimsension of a layer; equivalent to the number of
                                    GPs in this layer. If set to None, the output dimension is squashed.
                num_samples:        number of samples used to approximate expected log-lik in ELBO
                num_inducing:       number of inducing points used for variational distribution
                variational_dist:   variational distribution family; support diagonal
                                    covariance matrix (MeanField) and full covariance matrix (Cholesky)
                mean_type:          specify type of mean function; support constant and linear
                                    mean function
                initialize_method:  support using kmeans to initialize inducing points locations
                inducing_bounds:    specify bounds of initial inducing locations; typically
                                    set to be bounds of inputs when known
                train_x:            features in the training set, used to initialize inducing points
                                    locations with kmeans
        """
        if output_dims is None:
            # this is true only for the final layer with single output
            batch_shape = torch.Size([])
            inducing_points = torch.randn(num_inducing, input_dims)
        else:
            batch_shape = torch.Size([output_dims])
            if inducing_bounds is None:
                # this is true under 3 scenarios:
                # (1) all hidden layers
                # (2) initial layer when the bounds of features are unknown
                # (3) final layer with multiple outputs
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            else:
                # this is true only for the first layer when bounds of features are specified
                inducing_points = (
                    draw_sobol_samples(inducing_bounds, num_inducing, 1, [output_dims])
                    .squeeze(2)
                    .transpose(0, 1)
                )

        if initialize_method == "kmeans":
            # this is true only for the first layer when initialization method is kmeans
            assert train_x is not None
            if num_inducing < train_x.shape[0]:
                # this is true only when number of samples is larger than number of inducing points
                inducing_points = torch.from_numpy(
                    kmeans2(
                        train_x.detach().cpu().numpy(), num_inducing, minit="points"
                    )[0]
                ).repeat(output_dims, 1, 1)
            else:
                # this is true only when number of samples is smaller than number of inducing points
                # inducing points: [training samples, randomly generated inducing points]
                inducing_points = torch.cat(
                    [
                        train_x.repeat(output_dims, 1, 1),
                        inducing_points[:, train_x.shape[0] :, :],
                    ],
                    1,
                )

        if variational_dist == "MeanField":
            variational_distribution = MeanFieldVariationalDistribution(
                num_inducing_points=num_inducing, batch_shape=batch_shape
            )
        else:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing, batch_shape=batch_shape
            )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        self.num_samples = num_samples

        super(GPLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == "linear":
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        else:
            self.mean_module = ConstantMean(batch_shape=batch_shape)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(
        self, x: torch.Tensor, *other_inputs: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
            Overriding __call__ isn't strictly necessary, but it lets us add
            concatenation based skip connections easily. For example,
            hidden_layer2(hidden_layer1_outputs, inputs) will pass the
            concatenation of the first hidden layer's outputs and the input
            data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGPModel(DeepGP):
    """
        Define deep GP architecture including hidden layers and output layer.
        Currently the code only supports editing number of hidden layers here
        instead of using an argument to allow users specifying it. Users are
        able to specify the number of GPs in each layer.
    """

    def __init__(
        self,
        x_dim: int,
        num_GPs: List[int],
        skip_connections: List[bool] = None,
        num_tasks: int = 1,
        num_samples: int = 10,
        num_inducing: int = 100,
        variational_dist: str = "MeanField",
        mean_type: str = "linear",
        initialize_method: Optional[str] = None,
        inducing_bounds: torch.Tensor = None,
        train_x: torch.Tensor = None,
    ) -> None:
        """
            Args:
                x_dim:              dimension of original feature
                num_GPs:            its length specifies number of hidden layers; each element
                                    specifies the number of units in the layer
                skip_connections:   whether to include skip connection; its length should
                                    match num_GPs
                num_tasks:          specify dimension of output
                num_samples:        number of samples used to approximate expected log-lik in ELBO
                num_inducing:       number of inducing points used for variational
                                    distribution
                variational_dist:   variational distribution family; support diagonal
                                    covariance matrix (MeanField) and full covariance matrix (Cholesky)
                mean_type:          specify type of mean function; support constant and linear
                                    mean function
                initialize_method:  support using kmeans to initialize inducing points locations
                inducing_bounds:    specify bounds of initial inducing locations; typically
                                    set to be bounds of inputs when known
                train_x:            features in the training set, used to initialize inducing points
                                    locations with kmeans
        """
        # define a subclass to simplify code
        class GPLayerSub(GPLayer):
            def __init__(
                self,
                input_dims,
                output_dims,
                num_samples=num_samples,
                num_inducing=num_inducing,
                variational_dist=variational_dist,
                mean_type=mean_type,
                initialize_method=None,
                inducing_bounds=None,
                train_x=None,
            ):

                super().__init__(
                    input_dims,
                    output_dims,
                    num_samples,
                    num_inducing,
                    variational_dist,
                    mean_type,
                    initialize_method,
                    inducing_bounds,
                    train_x,
                )

        super().__init__()

        if skip_connections is None:
            skip_connections = [False] * len(num_GPs)

        self.num_hidden_layers = len(num_GPs)
        self.layers = nn.ModuleList()
        self.skip_connections = skip_connections

        # initial layer
        self.layers.append(
            GPLayerSub(
                x_dim,
                num_GPs[0],
                initialize_method=initialize_method,
                inducing_bounds=inducing_bounds,
                train_x=train_x,
            )
        )

        # hidden layers
        for i in range(self.num_hidden_layers - 1):
            self.layers.append(
                GPLayerSub(num_GPs[i] + x_dim * skip_connections[i], num_GPs[i + 1])
            )

        # final layer
        if num_tasks == 1:
            self.layers.append(
                GPLayerSub(num_GPs[-1] + x_dim * skip_connections[-1], None)
            )
        else:
            self.layers.append(
                GPLayerSub(num_GPs[-1] + x_dim * skip_connections[-1], num_tasks)
            )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        inputs = x
        x = self.layers[0](x)

        for i in range(self.num_hidden_layers):
            if self.skip_connections[i]:
                x = self.layers[i + 1](x, inputs)
            else:
                x = self.layers[i + 1](x)

        return x


class DeepGPInterLayer(DeepGPModel):
    """
        Check intermediate layers of deep GP model
    """

    def __init__(
        self,
        x_dim: int,
        num_GPs: List[int],
        model: DeepGPModel,
        layer: int,
        num_hidden_layers: int,
    ) -> None:
        super(DeepGPInterLayer, self).__init__(x_dim, num_GPs)
        self.features = nn.Sequential(
            *list(model.children())[: -(num_hidden_layers + 1 - layer)]
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        x = self.features(x)
        return x


def fit_deep_gp(
    model: DeepGPModel,
    likelihood: gpytorch.likelihoods,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    weights: torch.Tensor = None,
    device: str = "cpu",
    batch_size: int = 1024,
    num_epochs: int = 500,
    num_samples: int = 10,
    noise_fixed: bool = False,
    lr: float = 0.01,
) -> None:
    """
        Deep GP model fitting.

        Args:
            model:      a deep GP model
            likelihood: likelihood
            train_x:    [n, d] tensor; n training samples with d features
            train_y:    either a [n] tensor: 1-d response or a [n,t] tensor: t tasks
            weights:    weights of samples; if None then no weighting will be applied
            device:     specify the device where data and model should be stored on
            batch_size: sample size of each minibatch
            num_epochs: number of training epochs
            num_samples: number of samples used to approximate expected log-lik in ELBO
            noise_fixed: specify whether the noise level is fixed; support noise-free
                         deep GP
            lr:          learning rate of optimizer

        Returns:
            None
    """

    train_x, train_y = train_x.to(device), train_y.to(device)
    model, likelihood = model.to(device), likelihood.to(device)

    if weights is not None:
        weights = weights.to(device)
        train_dataset = TensorDataset(train_x, train_y, weights)
    else:
        train_dataset = TensorDataset(train_x, train_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    likelihood.train()

    # using adam optimizer
    if not noise_fixed:
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
        )
    else:
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)

    # variational ELBO under the deep GP setting
    mll = DeepApproximateMLL(VariationalELBO(likelihood, model, train_x.shape[-2]))

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        for data_batch in train_loader:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(data_batch[0])
                loss = -mll(output, data_batch[1])
                if weights is not None:
                    loss = torch.sum(loss * data_batch[2])
                loss.backward()
                optimizer.step()

                if not (i + 1) % 50:
                    print("Iter %d/%d - Loss: %.3f" % (i + 1, num_epochs, loss.item()))

    model.eval()
    likelihood.eval()


def pred_deep_gp(
    model: DeepGPModel,
    likelihood: gpytorch.likelihoods,
    test_x: torch.Tensor,
    test_y: torch.Tensor = None,
    device: str = "cpu",
    batch_size: int = 1024,
    num_samples: int = 10,
    num_tasks: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    """
        Prediction using trained deep GP model.

        Args:
            model:       a trained model
            likelihood:  likelihood with trained parameters
            test_x:      [n, d] tensor; n testing samples with d features
            test_y:      (1) None: test_y is unknown (log-likelihood will be
                         unavailable)
                         (2) a [n] tensor: 1-d response
                         (3) a [n,t] tensor: response with t tasks
            device:      specify the device where data and model should be stored on
            batch_size:  sample size of each minibatch; default to 1024
            num_samples: number of samples used to approximate expected log-lik in ELBO
            num_tasks:   number of tasks

        Returns:
            A tuple with two/three elements:
                predictive means
                predictive variances
                log-likelihoods (unavailable if test_y is None)
    """

    test_x = test_x.to(device=device)
    if test_y is not None:
        test_y = test_y.to(device)
        test_dataset = TensorDataset(test_x, test_y)
    else:
        test_dataset = TensorDataset(test_x)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    with torch.no_grad():
        means = []
        variances = []
        logliks = []
        for data_batch in test_loader:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                predictions = likelihood(model(data_batch[0]))
                means.append(predictions.mean)
                if num_tasks is None:
                    variances.append(predictions.variance)
                else:
                    variances.append(
                        get_covariance(
                            predictions.covariance_matrix,
                            num_samples,
                            num_tasks,
                            len(data_batch[0]),
                            device,
                        )
                    )
                if test_y is not None:
                    logliks.append(
                        likelihood.log_marginal(data_batch[1], model(data_batch[0]))
                    )
                    return (
                        torch.cat(means, dim=1),
                        torch.cat(variances, dim=1),
                        torch.cat(logliks, dim=1),
                    )

    return torch.cat(means, dim=1), torch.cat(variances, dim=1), None


def get_covariance(
    raw_covariance: torch.Tensor,
    num_samples: int,
    num_tasks: int,
    num_test_samples: int,
    device: str,
) -> torch.Tensor:
    """
        get covariance matrices from a large blocked diagonal matrix; used inside
        pred_deep_gp
    """

    covariance = torch.empty(num_samples, num_test_samples, num_tasks, num_tasks).to(
        device
    )
    for i in range(num_test_samples):
        covariance[:, i, :, :] = raw_covariance[
            :, num_tasks * i : num_tasks * (i + 1), num_tasks * i : num_tasks * (i + 1)
        ]
    return covariance
