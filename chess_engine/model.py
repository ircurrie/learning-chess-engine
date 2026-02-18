import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Simple value network mapping board tensor -> scalar value in [-1, 1]."""

    def __init__(self, input_size: int = 769, hidden: int = 256):
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

    Input: board state (batch, 769) where the final scalar is `turn`.
    Output: (batch, 4096) scores, one per possible (from,to) square pair
    (from_index * 64 + to_index). We mask to legal moves and softmax over them.
    """

    def __init__(self, input_size: int = 769, output_size: int=4096, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 769) -> returns (batch, 4096)
        return self.net(x)
