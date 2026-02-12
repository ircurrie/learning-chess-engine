from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from .utils import board_to_tensor


class ChessDataset(Dataset):
    """A lightweight dataset that holds (FEN, value) pairs for supervised training.

    `items` can be a list of (fen_str, float_value) tuples.
    """

    def __init__(self, items: Optional[List[Tuple[str, float]]] = None):
        self.items = items or []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fen, value = self.items[idx]
        x = board_to_tensor(fen)
        y = torch.tensor(value, dtype=torch.float32)
        return x, y

    @classmethod
    def from_pgn(cls, pgn_path: str, max_games: int = 1000):
        """Parse a PGN file and create simple training pairs.

        For now this is a stub that extracts final result for each position,
        producing one sample per game: the starting position with game result.
        Replace or extend with richer extraction logic later.
        """
        import chess.pgn

        items = []
        with open(pgn_path, "r", encoding="utf-8") as f:
            game_count = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                else:
                    value = 0.0
                # use starting position (FEN of initial board)
                items.append((chess.Board().fen(), value))
                game_count += 1
                if game_count >= max_games:
                    break
        return cls(items)
