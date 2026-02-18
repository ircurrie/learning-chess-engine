import chess
import random
from typing import Optional, Tuple

import torch
from chess_engine.model import PolicyNet, ValueNet
from chess_engine.utils import select_move_and_logprob, board_to_tensor

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def evaluate_board(board: chess.Board) -> int:
    """Simple material evaluation from White's perspective."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def minimax(board: chess.Board, depth: int, alpha: int = -999999, beta: int = 999999, maximizing: bool = True):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None
    best_move = None
    if maximizing:
        max_eval = -999999
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = 999999
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move


def load_models(prefix: str, device: Optional[str] = None) -> Tuple[PolicyNet, ValueNet]:
    """Load policy and value models from files with given prefix.

    Expects files: `{prefix}_policy.pt` and `{prefix}_value.pt`.
    Returns `(policy_net, value_net)` on success.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet()
    value = ValueNet()
    policy_path = prefix + "_policy.pt"
    value_path = prefix + "_value.pt"
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    value.load_state_dict(torch.load(value_path, map_location=device))
    policy.to(device)
    value.to(device)
    policy.eval()
    value.eval()
    return policy, value


def get_model_move(board: chess.Board, policy: PolicyNet, temperature: float = 1.0, device: Optional[str] = None) -> chess.Move | None:
    """Use a loaded `policy` network to select and return a move for `board`.

    Falls back to `None` when no legal moves are available.
    """
    chosen, _, moves, _ = select_move_and_logprob(policy, board, temperature=temperature, device=device)
    # Safety: ensure chosen is legal in current board. If not, try to recover.
    legal = list(board.legal_moves)
    if chosen in legal:
        return chosen
    # Log unexpected illegal selection for debugging
    try:
        print("[policy-warning] chosen move not legal for board:\n", board.fen())
        print("[policy-warning] chosen:", getattr(chosen, 'uci', lambda: str(chosen))())
        print("[policy-warning] legal moves:", [m.uci() for m in legal])
    except Exception:
        pass
    # If chosen is not in legal moves (unexpected), try to find a move with same UCI
    if chosen is not None:
        try:
            uc = chosen.uci()
            for m in legal:
                if m.uci() == uc:
                    return m
        except Exception:
            pass
    # Fall back to a random legal move (or None if no legal moves)
    return (random.choice(legal) if legal else None)


def get_value_estimate(board: chess.Board, value_net: ValueNet, device: Optional[str] = None) -> float:
    """Return the value network's prediction for `board` (scalar in [-1,1])."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = board_to_tensor(board).to(device)
    value_net.eval()
    with torch.no_grad():
        v = value_net(x.unsqueeze(0)).item()
    return float(v)


def get_top_moves(board: chess.Board, policy: PolicyNet, device: Optional[str] = None, top_n: int = 10) -> list:
    """Return the top N moves from the policy network for a given board.
    
    Returns list of tuples: (move, score, san).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = board_to_tensor(board).to(device).unsqueeze(0)
    policy.eval()
    with torch.no_grad():
        square_scores = policy(x).squeeze(0)  # (768,)
    
    moves = list(board.legal_moves)
    move_scores = []
    for mv in moves:
        dest_sq = mv.to_square
        score = square_scores[dest_sq].item()
        try:
            san = board.san(mv)
        except Exception:
            san = mv.uci()
        move_scores.append((mv, score, san))
    
    # Sort by score descending
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return move_scores[:top_n]


def get_engine_move(board: chess.Board, depth: int = 2, policy: Optional[PolicyNet] = None, temperature: float = 1.0, device: Optional[str] = None) -> chess.Move | None:
    """Return a move chosen by a provided `policy` (preferred) or fallback to minimax.

    - If `policy` is provided, sample a move using the policy.
    - Otherwise use `minimax` (existing behavior).
    """
    if policy is not None:
        mv = get_model_move(board, policy, temperature=temperature, device=device)
        if mv is not None:
            return mv
    # fallback to classical minimax engine
    _, move = minimax(board, depth, maximizing=board.turn == chess.WHITE)
    if move is None:
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None
    return move
