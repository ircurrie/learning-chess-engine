import chess
from chess_game.game import get_engine_move


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
    board = chess.Board()
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

            # engine move
            engine_move = get_engine_move(board, depth=2)
            if engine_move is None:
                break
            print("Engine plays:", board.san(engine_move))
            board.push(engine_move)

        print(board)
        print("Game over:", board.result())
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == '__main__':
    main()
