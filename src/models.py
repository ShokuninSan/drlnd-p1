from typing import Optional, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation_fn: Callable = F.relu,
        seed: Optional[int] = None,
    ):
        """
        Creates a QNetwork instance.

        :param layer_dims: dimensions of NN layers.
        :param activation_fn: activation function (default: ReLU).
        :param seed: random seed.
        """
        super(QNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.layers = nn.ModuleList([])
        for layer in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[layer], layer_dims[layer + 1]))

        self.activation_fn = activation_fn

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        :param state: state input.
        :return: action value output.
        """
        x = state
        for h in self.layers[:-1]:
            x = self.activation_fn(h(x))
        return self.layers[-1](x)
