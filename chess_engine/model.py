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
    """Policy model that scores each square on the board (higher = more desirable destination).

    Input: board state (batch, 768)
    Output: (batch, 768) scores, one per square. For move selection we apply softmax
    over the scores of legal move destinations.
    """

    def __init__(self, input_size: int = 768, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),  # output 768 scores (one per square)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 768) -> returns (batch, 768)
        return self.net(x)
