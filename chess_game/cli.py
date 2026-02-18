import argparse
import chess
from chess_game.game import load_models, get_model_move, get_value_estimate, get_engine_move


def input_move(board: chess.Board) -> chess.Move | None:
    s = input("Your move (UCI like e2e4 or SAN like Nf3): ").strip()
    if not s:
        return None
    # Try UCI first
    try:
        move = chess.Move.from_uci(s)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    # Try SAN
    try:
        move = board.parse_san(s)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Play human vs engine (optionally using a trained model)")
    parser.add_argument("--model-prefix", help="prefix for model files (saved as <prefix>_policy.pt and <prefix>_value.pt)")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for policy")
    parser.add_argument("--depth", type=int, default=2, help="minimax depth for fallback engine")
    args = parser.parse_args()

    board = chess.Board()
    policy = None
    value_net = None
    if args.model_prefix:
        try:
            policy, value_net = load_models(args.model_prefix)
            print(f"Loaded models from {args.model_prefix}")
        except Exception as e:
            print(f"Failed to load models from {args.model_prefix}: {e}")
            policy = None

    print("Simple CLI chess — human vs simple engine")
    print("Enter moves in UCI (e2e4) or SAN (Nf3). Type Ctrl-C to quit.")
    try:
        while not board.is_game_over():
            print(board)
            if board.turn == chess.WHITE:
                print("White to move")
            else:
                print("Black to move")

            move = input_move(board)
            if move is None:
                print("Invalid or empty move. Try again.")
                continue
            board.push(move)
            if board.is_game_over():
                break

            # engine move: if policy loaded, try policy first and fallback to minimax
            engine_move = None
            used = "minimax"
            if policy is not None:
                try:
                    engine_move = get_model_move(board, policy, temperature=args.temperature)
                    if engine_move is not None:
                        used = "policy"
                except Exception as e:
                    print("Policy selection failed, falling back to minimax:", e)
                    engine_move = None

            if engine_move is None:
                engine_move = get_engine_move(board, depth=args.depth, policy=None)
                used = "minimax"

            if engine_move is None:
                break

            # log where move came from and optional value estimate
            # Print SAN/description safely (compute SAN before push may raise if move illegal)
            try:
                san_str = board.san(engine_move)
            except Exception:
                try:
                    san_str = engine_move.uci()
                except Exception:
                    san_str = str(engine_move)

            if value_net is not None:
                try:
                    val = get_value_estimate(board, value_net)
                    print(f"Engine ({used}) plays: {san_str}  |  value={val:.3f}")
                except Exception:
                    print(f"Engine ({used}) plays: {san_str}")
            else:
                print(f"Engine ({used}) plays: {san_str}")

            board.push(engine_move)

        print(board)
        print("Game over:", board.result())
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == '__main__':
    main()
