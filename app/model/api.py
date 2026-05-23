"""JSON API for Connect 4 model inference (weights stay server-side)."""

from __future__ import annotations

import json

import numpy as np
from flask import Blueprint, request

from model.game_logic import check_winner, next_player_to_move, rebuild_board
from model.inference import MODEL_REGISTRY, best_move_and_scores

model = Blueprint("model", __name__)


def _board_from_payload(data: dict):
    """Return (board, error_message) from JSON body."""
    moves = data.get("moves")
    if moves is not None:
        if not isinstance(moves, list):
            return None, "moves must be a list of column indices (0–6)."
        try:
            moves = [int(m) for m in moves]
        except (TypeError, ValueError):
            return None, "moves must contain integers."
        board, err = rebuild_board(moves)
        if err:
            return None, err
        return board, None

    board = data.get("board")
    if board is not None:
        if not isinstance(board, list) or len(board) != 6:
            return None, "board must be a 6×7 grid."
        try:
            grid = [[int(cell) for cell in row] for row in board]
        except (TypeError, ValueError):
            return None, "board cells must be integers."
        if any(len(row) != 7 for row in grid):
            return None, "board must be a 6×7 grid."
        return grid, None

    return None, "Provide either moves (column list) or board (6×7 grid)."


def _scores_json(masked: np.ndarray) -> list:
    rows = []
    for c in range(7):
        v = masked[c]
        if np.isneginf(v):
            rows.append({"col": c, "value": None})
        else:
            rows.append({"col": c, "value": round(float(v), 4)})
    return rows


@model.route("/best_move", methods=["POST"])
def best_move():
    data = request.get_json(silent=True) or {}
    model_key = data.get("model", "cnn")
    if model_key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        return json.dumps({"error": f"model must be one of: {valid}"}), 400, {
            "Content-Type": "application/json"
        }

    board, err = _board_from_payload(data)
    if err:
        return json.dumps({"error": err}), 400, {"Content-Type": "application/json"}

    outcome = check_winner(board)
    if outcome == 2:
        return json.dumps({"error": "Board is full (draw)."}), 400, {
            "Content-Type": "application/json"
        }
    if outcome != 0:
        return json.dumps({"error": "Position already has a winner."}), 400, {
            "Content-Type": "application/json"
        }

    result = best_move_and_scores(board, model_key)
    if result is None:
        return json.dumps(
            {
                "error": (
                    f"Model '{model_key}' is not loaded. "
                    "Place weights in app/models/ or set MODEL_DIR."
                )
            }
        ), 503, {"Content-Type": "application/json"}

    best_col, masked = result
    moves = data.get("moves")
    num_discs = len(moves) if moves is not None else sum(
        1 for r in board for c in r if c != 0
    )
    side = next_player_to_move(num_discs)

    out = {
        "input": data,
        "model": model_key,
        "side_to_move": side,
        "side_label": "Red (+1)" if side == 1 else "Yellow (−1)",
        "best_col": best_col,
        "scores": _scores_json(masked),
    }
    return json.dumps(out, indent=2), 200, {"Content-Type": "application/json"}
