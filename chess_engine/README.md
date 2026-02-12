chess_engine — simple training scaffolding

This package provides a minimal starting point for building and training a chess engine.

Install dependencies (recommended into a venv):

```bash
pip install -r requirements.txt
```

Quick usage example (supervised value training):

```python
from chess_engine import ChessDataset, Trainer

# create a tiny dataset: starting position -> draw (0.0)
dataset = ChessDataset([("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0)])
trainer = Trainer()
trainer.fit(dataset, epochs=2, batch_size=1)
```

Notes:
- `utils.board_to_tensor` encodes a board as a 12x8x8 flattened tensor.
- `model.ValueNet` is a small MLP predicting a scalar value in [-1, 1].
- `dataset.from_pgn` is a basic stub; replace with richer sample extraction for self-play or supervised training.
 - `dataset.from_pgn` is a basic stub; replace with richer sample extraction for self-play or supervised training.

Self-play (unsupervised) quickstart:

```python
from chess_engine import Trainer

trainer = Trainer()
# run 50 iterations collecting 8 games per iteration
trainer.train_self_play(iterations=50, games_per_iter=8)
trainer.save("models/run1")
```

Notes:
- `PolicyNet` scores resulting boards and a softmax over move scores is used to sample moves.
- `Trainer.train_self_play` uses REINFORCE with a value baseline for policy updates.
