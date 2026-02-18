import chess
import torch
from typing import Optional

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board_or_fen):
    """Convert a `chess.Board` or FEN string into a tensor containing piece planes
    plus a `turn` feature. Returns a torch.FloatTensor shape (769,).

    Plane order: [white pawn, white knight, ..., white king, black pawn, ..., black king]
    followed by a single scalar: +1.0 if white to move, -1.0 if black to move.
    """
    if isinstance(board_or_fen, str):
        board = chess.Board(board_or_fen)
    elif isinstance(board_or_fen, chess.Board):
        board = board_or_fen
    else:
        raise ValueError(f"Expected str or chess.Board, got {type(board_or_fen)}")

    planes = []
    for color in [chess.WHITE, chess.BLACK]:
        for ptype in PIECE_TYPES:
            plane = [0.0] * 64
            for sq in chess.SQUARES:
                piece = board.piece_type_at(sq)
                piece_color = board.color_at(sq)
                if piece == ptype and piece_color == color:
                    plane[sq] = 1.0
            planes.append(plane)
    # flatten planes in square order and append turn feature
    flat = [v for plane in planes for v in plane]
    turn_val = 1.0 if board.turn == chess.WHITE else -1.0
    flat.append(turn_val)
    return torch.tensor(flat, dtype=torch.float32)


def select_move_and_logprob(policy, board, temperature: float = 1.0, device: Optional[str] = None):
    """Given a `policy` (PolicyNet) and a `chess.Board`, score each legal move
    by the destination square score, sample one move from the softmax distribution,
    and return (chosen_move, log_prob, moves, probs_tensor).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    moves = list(board.legal_moves)
    if len(moves) == 0:
        return None, None, moves, None

    # Score the current board state: policy now returns per-move logits of size 4096
    board_tensor = board_to_tensor(board).to(device).unsqueeze(0)  # (1, 769)
    move_logits = policy(board_tensor).squeeze(0)  # (4096,)

    # Score each legal move by the (from,to) index = from*64 + to
    move_scores = []
    for mv in moves:
        idx = mv.from_square * 64 + mv.to_square
        move_scores.append(move_logits[idx])
    move_scores = torch.stack(move_scores)

    # Softmax over legal moves and sample
    probs = torch.softmax(move_scores / max(1e-8, temperature), dim=0)
    dist = torch.distributions.Categorical(probs)
    idx = dist.sample()
    chosen = moves[int(idx.item())]
    logp = dist.log_prob(idx)
    return chosen, logp, moves, probs
