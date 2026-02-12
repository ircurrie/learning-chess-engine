import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Simple value network mapping board tensor -> scalar value in [-1, 1]."""

    def __init__(self, input_size: int = 768, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PolicyNet(nn.Module):
    """Policy model that scores board states (higher = more desirable).

    For action selection we score the resulting board for each legal move and form
    a softmax over those scores to obtain a distribution over moves.
    """

    def __init__(self, input_size: int = 768, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 768) -> returns (batch,)
        out = self.net(x).squeeze(-1)
        return out
