# Simple CLI Chess

This is a minimal human vs simple-engine CLI chess game using `python-chess`.

Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Run

```bash
python -m chess_game.cli
```

Controls
- Enter moves in UCI format (`e2e4`) or SAN (`Nf3`).
- Ctrl-C to quit.

Notes
- The engine uses a tiny minimax with material evaluation (depth=2).
- This is intended as a small starter project you can extend (GUI, stronger AI, network play).
