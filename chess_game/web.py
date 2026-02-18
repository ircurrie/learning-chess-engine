from flask import Flask, render_template, request, redirect, url_for, flash
import chess
import chess.svg

from chess_game.game import load_models, get_engine_move, get_value_estimate, get_top_moves

app = Flask(__name__)
app.secret_key = "dev-secret"

# Global board and optional models (single-user demo)
board = chess.Board()
policy = None
value_net = None


@app.route("/", methods=["GET"])
def index():
    svg = chess.svg.board(board=board, size=480)
    value = None
    top_moves = []
    if value_net is not None:
        try:
            value = get_value_estimate(board, value_net)
        except Exception:
            value = None
    if policy is not None:
        try:
            top_moves = get_top_moves(board, policy, top_n=10)
        except Exception:
            top_moves = []
    # Debug log the evaluation to server console
    try:
        print(f"[web] FEN={board.fen()} value={value} top_moves={[ (m.uci(),round(s,3)) for m,s,sa in top_moves ]}")
    except Exception:
        pass
    return render_template("index.html", board_svg=svg, value=value, fen=board.fen(), top_moves=top_moves)


@app.route("/move", methods=["POST"])
def move():
    s = request.form.get("move", "").strip()
    if not s:
        flash("Empty move")
        return redirect(url_for("index"))
    # Try UCI then SAN
    try:
        mv = chess.Move.from_uci(s)
        if mv in board.legal_moves:
            board.push(mv)
            return redirect(url_for("index"))
    except Exception:
        pass
    try:
        mv = board.parse_san(s)
        if mv in board.legal_moves:
            board.push(mv)
            return redirect(url_for("index"))
    except Exception:
        pass
    flash("Invalid move")
    return redirect(url_for("index"))


@app.route("/engine", methods=["POST"])
def engine_move():
    global policy, value_net
    depth = int(request.form.get("depth", 2))
    temperature = float(request.form.get("temperature", 1.0))
    # use policy if available
    mv = None
    used = "minimax"
    if policy is not None:
        try:
            mv = get_engine_move(board, depth=depth, policy=policy, temperature=temperature)
            used = "policy"
        except Exception:
            mv = None
            used = "minimax"
    if mv is None:
        mv = get_engine_move(board, depth=depth, policy=None)
        used = "minimax"
    if mv is None:
        flash("No legal moves")
        return redirect(url_for("index"))
    # Compute SAN in the current position, then push move. Compute value after push.
    try:
        san_str = board.san(mv)
    except Exception:
        san_str = mv.uci()
    board.push(mv)
    info = f"Engine ({used}) played: {san_str}"
    if value_net is not None:
        try:
            val = get_value_estimate(board, value_net)
            info += f" | value={val:.3f}"
        except Exception:
            pass
    flash(info)
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"]) 
def reset():
    global board
    board = chess.Board()
    flash("Board reset")
    return redirect(url_for("index"))


@app.route("/load", methods=["POST"]) 
def load():
    global policy, value_net
    prefix = request.form.get("prefix", "").strip()
    if not prefix:
        flash("Model prefix required")
        return redirect(url_for("index"))
    try:
        policy, value_net = load_models(prefix)
        flash(f"Loaded models from {prefix}")
    except Exception as e:
        flash(f"Failed to load models: {e}")
    return redirect(url_for("index"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Run chess web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-prefix", help="Optional model prefix to load at startup")
    args = parser.parse_args()

    if args.model_prefix:
        try:
            policy, value_net = load_models(args.model_prefix)
            print(f"Loaded models from {args.model_prefix}")
        except Exception as e:
            print("Failed to load models:", e)

    app.run(host=args.host, port=args.port, debug=True)
