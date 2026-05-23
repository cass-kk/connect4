"""Connect4 rules: 6 rows x 7 columns, row 0 is top, row 5 is bottom."""

from __future__ import annotations

from typing import List, Optional, Tuple

ROWS, COLS = 6, 7


def empty_board() -> List[List[int]]:
    return [[0] * COLS for _ in range(ROWS)]


def column_top_empty(board: List[List[int]], col: int) -> bool:
    return 0 <= col < COLS and board[0][col] == 0


def drop_piece(board: List[List[int]], col: int, player: int) -> Optional[int]:
    """Place player in column; return row index or None if illegal."""
    if not column_top_empty(board, col):
        return None
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            board[r][col] = player
            return r
    return None


def check_winner(board: List[List[int]]) -> int:
    """Return 1 / -1 if that player won, 0 if no winner yet, 2 if draw."""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                streak = 1
                for k in range(1, 4):
                    nr, nc = r + dr * k, c + dc * k
                    if (
                        0 <= nr < ROWS
                        and 0 <= nc < COLS
                        and board[nr][nc] == p
                    ):
                        streak += 1
                    else:
                        break
                if streak >= 4:
                    return p

    if all(board[0][c] != 0 for c in range(COLS)):
        return 2
    return 0


def board_to_numpy_list(board: List[List[int]]) -> list:
    """JSON-serializable copy."""
    return [row[:] for row in board]


def rebuild_board(moves: List[int]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """
    Replay a list of column indices (alternating P1, P2, P1, …).
    Returns (board, None) or (None, error message).
    """
    b = empty_board()
    turn = 1
    for col in moves:
        if not (0 <= col < COLS):
            return None, "Column out of range in move list."
        if drop_piece(b, col, turn) is None:
            return None, "A column was full when replaying the position."
        turn = -turn
    return b, None


def next_player_to_move(num_discs_placed: int) -> int:
    """Empty board → P1 (+1) to move; after one disc, P2; etc."""
    return 1 if num_discs_placed % 2 == 0 else -1


def last_move_position(moves: List[int]) -> Optional[Tuple[int, int]]:
    """Return (row, col) of the most recently played disc, or None if no moves."""
    if not moves:
        return None
    col = moves[-1]
    row = ROWS - sum(1 for m in moves if m == col)
    return row, col
