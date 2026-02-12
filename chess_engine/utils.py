import chess
import torch

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board_or_fen):
    """Convert a `chess.Board` or FEN string into a 768-dim float tensor (12x8x8).

    Planes order: [white pawn, white knight, ..., white king, black pawn, ..., black king].
    Returns a torch.FloatTensor shape (768,).
    """
    if isinstance(board_or_fen, str):
        board = chess.Board(board_or_fen)
    else:
        board = board_or_fen

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
    # flatten planes in square order
    flat = [v for plane in planes for v in plane]
    return torch.tensor(flat, dtype=torch.float32)


def select_move_and_logprob(policy, board, temperature: float = 1.0, device: Optional[str] = None):
    """Given a `policy` (PolicyNet) and a `chess.Board`, score resulting boards
    for each legal move, sample one move from the softmax distribution, and
    return (chosen_move, log_prob, moves, probs_tensor).
    """
    from typing import Optional as _Optional

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    moves = list(board.legal_moves)
    if len(moves) == 0:
        return None, None, moves, None

    # build batch of next-state tensors
    next_tensors = []
    for mv in moves:
        b2 = board.copy()
        b2.push(mv)
        next_tensors.append(board_to_tensor(b2))

    xb = torch.stack(next_tensors).to(device)
    scores = policy(xb)  # (n_moves,)
    probs = torch.softmax(scores / max(1e-8, temperature), dim=0)
    dist = torch.distributions.Categorical(probs)
    idx = dist.sample()
    chosen = moves[int(idx.item())]
    logp = dist.log_prob(idx)
    return chosen, logp, moves, probs
