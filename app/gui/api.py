"""Web UI: start screen, then play Connect 4 against the model."""

from __future__ import annotations

from typing import Optional

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from config import title
from model.game_logic import (
    COLS,
    board_to_numpy_list,
    check_winner,
    drop_piece,
    empty_board,
    next_player_to_move,
    rebuild_board,
    last_move_position,
)
from model.inference import ai_move, is_model_loaded, model_availability, model_choices, model_label

gui = Blueprint("gui", __name__)

MODEL_CHOICES = model_choices()
VALID_MODEL_KEYS = frozenset(key for key, _ in MODEL_CHOICES)


def _play_redirect():
    """Return to the board anchor so the page does not jump to the top after each move."""
    return redirect(url_for("gui.play") + "#board")


def _moves() -> list:
    if "moves" not in session:
        session["moves"] = []
    return session["moves"]


def _clear_game() -> None:
    for key in ("model_key", "human_side", "moves", "first_player"):
        session.pop(key, None)
    session.modified = True


def _game_active() -> bool:
    return session.get("model_key") in VALID_MODEL_KEYS


def _human_side() -> int:
    return int(session.get("human_side", 1))


def _model_side() -> int:
    return -_human_side()


def _side_name(side: int) -> str:
    return "Red" if side == 1 else "Yellow"


def _outcome_message(winner: int, human_side: int) -> str:
    if winner == 2:
        return "Draw — the board is full."
    if winner == human_side:
        return "You win!"
    return "The model wins."


def _apply_model_turn(model_key: str) -> Optional[str]:
    """Let the model play once if it is their turn. Returns an error message or None."""
    moves = _moves()
    board, err = rebuild_board(moves)
    if err:
        return err

    w = check_winner(board)
    if w != 0:
        return None

    if next_player_to_move(len(moves)) != _model_side():
        return None

    col = ai_move(board, model_key)
    if col is None:
        return (
            f"The {model_label(model_key)} model could not choose a move "
            "(weights missing or inference failed)."
        )

    if drop_piece(board, col, _model_side()) is None:
        return "The model tried an illegal column."

    moves.append(col)
    session.modified = True
    return None


@gui.route("/", methods=["GET"])
def index():
    availability = model_availability()
    return render_template(
        "start.html",
        title=title,
        models=MODEL_CHOICES,
        model_availability=availability,
        any_model_loaded=any(availability.values()),
    )


@gui.post("/start")
def start_game():
    model_key = request.form.get("model", "")
    first = request.form.get("first_player", "")

    if model_key not in VALID_MODEL_KEYS:
        flash("Choose a model.")
        return redirect(url_for("gui.index"))

    if not is_model_loaded(model_key):
        flash(f"{model_label(model_key)} is not loaded on this server.")
        return redirect(url_for("gui.index"))

    if first not in ("human", "model"):
        flash("Choose who goes first.")
        return redirect(url_for("gui.index"))

    err = _begin_match(model_key, first)
    if err:
        _clear_game()
        flash(err)
        return redirect(url_for("gui.index"))

    return _play_redirect()


def _begin_match(model_key: str, first: str) -> Optional[str]:
    """Reset the board for a match; keep model and first-player settings."""
    session["model_key"] = model_key
    session["first_player"] = first
    session["human_side"] = -1 if first == "model" else 1
    session["moves"] = []
    session.modified = True
    if first == "model":
        return _apply_model_turn(model_key)
    return None


@gui.route("/play", methods=["GET"])
def play():
    if not _game_active():
        return redirect(url_for("gui.index"))

    model_key = session["model_key"]
    moves = _moves()
    board, _ = rebuild_board(moves)
    if (
        board is not None
        and check_winner(board) == 0
        and next_player_to_move(len(moves)) == _model_side()
    ):
        err = _apply_model_turn(model_key)
        if err:
            flash(err)

    return render_template("game.html", title=title, **_play_context())


@gui.post("/play/move")
def play_move():
    if not _game_active():
        return redirect(url_for("gui.index"))

    model_key = session["model_key"]
    raw = request.form.get("column")
    try:
        col = int(raw)
    except (TypeError, ValueError):
        flash("Invalid column.")
        return _play_redirect()

    if col < 0 or col >= COLS:
        flash("Column out of range.")
        return _play_redirect()

    moves = _moves()
    board, err = rebuild_board(moves)
    if err:
        _clear_game()
        flash(err)
        return redirect(url_for("gui.index"))

    w = check_winner(board)
    if w != 0:
        flash("This game is already over.")
        return _play_redirect()

    human = _human_side()
    if next_player_to_move(len(moves)) != human:
        flash("Wait for the model to move.")
        return _play_redirect()

    if drop_piece(board, col, human) is None:
        flash("That column is full.")
        return _play_redirect()

    moves.append(col)
    session.modified = True

    w = check_winner(board)
    if w == 0:
        err = _apply_model_turn(model_key)
        if err:
            flash(err)

    return _play_redirect()


@gui.post("/play/change-model")
def change_model():
    _clear_game()
    return redirect(url_for("gui.index"))


@gui.post("/play/replay")
def replay():
    if not _game_active():
        return redirect(url_for("gui.index"))

    model_key = session["model_key"]
    first = session.get("first_player", "human")
    if first not in ("human", "model"):
        first = "human"

    err = _begin_match(model_key, first)
    if err:
        flash(err)
    return _play_redirect()


def _play_context():
    model_key = session["model_key"]
    human = _human_side()
    model_side = _model_side()
    moves = _moves()
    board, err = rebuild_board(moves)
    if err:
        session["moves"] = []
        session.modified = True
        moves = []
        board = empty_board()
        flash(err)

    w = check_winner(board)
    can_play = w == 0 and next_player_to_move(len(moves)) == human
    your_color = _side_name(human)
    ai_color = _side_name(model_side)

    status_kind = "play"
    status_message = ""
    if w == 2:
        status_kind = "over"
        status_message = _outcome_message(2, human)
    elif w != 0:
        status_kind = "over"
        status_message = _outcome_message(w, human)

    turn_hint = ""
    if status_kind == "play":
        if can_play:
            turn_hint = f"Your turn ({your_color})."
        else:
            turn_hint = f"Model is thinking… ({ai_color})."

    last_move = last_move_position(moves)
    last_move_row = last_move[0] if last_move else None
    last_move_col = last_move[1] if last_move else None
    last_move_by_model = (
        last_move is not None
        and (1 if (len(moves) - 1) % 2 == 0 else -1) == model_side
    )

    return {
        "board": board_to_numpy_list(board),
        "move_count": len(moves),
        "model_label": model_label(model_key),
        "model_key": model_key,
        "your_color": your_color,
        "ai_color": ai_color,
        "human_is_red": human == 1,
        "can_play": can_play,
        "columns": list(range(COLS)),
        "status_kind": status_kind,
        "status_message": status_message,
        "turn_hint": turn_hint,
        "went_first": session.get("first_player", "human"),
        "last_move_row": last_move_row,
        "last_move_col": last_move_col,
        "last_move_by_model": last_move_by_model,
    }
