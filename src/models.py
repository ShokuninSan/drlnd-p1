import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_sizes = [512, 256]
        self.h1 = nn.Linear(state_size, hidden_sizes[0])
        self.h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.act = F.relu
        self.output = nn.Linear(hidden_sizes[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.output(self.act(self.h2(self.act(self.h1(state)))))
