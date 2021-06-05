import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        """
        Creates a QNetwork instance.

        :param state_size: state space dimension.
        :param action_size: action space dimension.
        :param seed: random seed.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_sizes = [512, 256]
        self.h1 = nn.Linear(state_size, hidden_sizes[0])
        self.h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.act = F.relu
        self.output = nn.Linear(hidden_sizes[1], action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass.

        :param state: state input.
        :return: action value output.
        """
        return self.output(self.act(self.h2(self.act(self.h1(state)))))
