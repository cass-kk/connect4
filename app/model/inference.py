"""Load models from disk and predict moves. Runs only on the server.

TensorFlow is imported only when loading Keras models, so a broken or missing
TensorFlow install does not block the PyTorch CNN from running.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_tf: Any = None
_tf_unavailable_reason: str | None = None

# kind: pytorch_cnn | keras_transformer | keras_board
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "cnn": {
        "label": "Convolutional Neural Network (CNN)",
        "filename": "cnn_connect4.pt",
        "kind": "pytorch_cnn",
    },
    "transformer": {
        "label": "Transformer",
        "filename": "transformer_connect4.keras",
        "kind": "keras_transformer",
    },
    "dqn": {
        "label": "Deep Q-Network (DQN)",
        "filename": "dqn.keras",
        "kind": "keras_board",
    },
    "pg": {
        "label": "Policy Gradient (PG)",
        "filename": "pg.keras",
        "kind": "keras_board",
    },
    "ensemble_dqn": {
        "label": "Ensemble DQN",
        "filename": "ensemble_dqn_component.keras",
        "kind": "keras_board",
    },
    "ensemble_pg": {
        "label": "Ensemble PG",
        "filename": "ensemble_pg_component.keras",
        "kind": "keras_board",
    },
}

_loaded: Dict[str, Any] = {}


def _get_tf():
    """Return the tensorflow module, or None if import failed (cached)."""
    global _tf, _tf_unavailable_reason
    if _tf is False:
        return None
    if _tf is not None:
        return _tf
    try:
        import tensorflow as tf

        _tf = tf
        return tf
    except Exception as e:
        _tf = False
        _tf_unavailable_reason = str(e)
        print(
            "TensorFlow could not be imported; Keras models are disabled. "
            f"Reason: {e}"
        )
        return None


def _keras_custom_objects(tf):
    """Build custom layer classes (must match names in saved .keras)."""

    @tf.keras.utils.register_keras_serializable()
    class PositionalIndex(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            seq_length = tf.shape(inputs)[1]
            return tf.range(seq_length, dtype=tf.int32)

        def get_config(self):
            return super().get_config()

    @tf.keras.utils.register_keras_serializable()
    class ClassTokenIndex(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            return tf.zeros((batch_size, 1), dtype=tf.int32)

        def get_config(self):
            return super().get_config()

    @tf.keras.utils.register_keras_serializable()
    class GetItem(tf.keras.layers.Layer):
        def __init__(self, index=0, **kwargs):
            super().__init__(**kwargs)
            self.index = index

        def __call__(self, *args, **kwargs):
            def clean(obj):
                if tf.is_tensor(obj) or (
                    hasattr(obj, "__class__")
                    and "KerasTensor" in obj.__class__.__name__
                ):
                    return obj
                if isinstance(obj, (list, tuple)):
                    cleaned = [clean(x) for x in obj]
                    return type(obj)([c for c in cleaned if c is not None])
                return None

            return super().__call__(*clean(list(args)), **kwargs)

        def call(self, inputs, *args, **kwargs):
            if isinstance(inputs, list):
                return inputs[self.index]
            return inputs[:, self.index]

        def get_config(self):
            config = super().get_config()
            config.update({"index": self.index})
            return config

    return {
        "PositionalIndex": PositionalIndex,
        "ClassTokenIndex": ClassTokenIndex,
        "GetItem": GetItem,
    }


class Connect4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.mean([2, 3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _model_dir() -> str:
    env = os.environ.get("MODEL_DIR")
    if env:
        return env
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def model_choices() -> Tuple[Tuple[str, str], ...]:
    return tuple((key, spec["label"]) for key, spec in MODEL_REGISTRY.items())


def model_availability() -> Dict[str, bool]:
    return {key: is_model_loaded(key) for key in MODEL_REGISTRY}


def _load_pytorch_cnn(path: str) -> Optional[Connect4CNN]:
    model = Connect4CNN()
    state = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


def _load_keras_model(path: str, *, needs_custom_objects: bool) -> Optional[Any]:
    tf = _get_tf()
    if tf is None:
        return None
    if needs_custom_objects:
        custom_objects = _keras_custom_objects(tf)
        with tf.keras.utils.custom_object_scope(custom_objects):
            return tf.keras.models.load_model(path)
    return tf.keras.models.load_model(path)


def load_models() -> None:
    global _loaded
    _loaded = {}
    model_dir = _model_dir()

    for key, spec in MODEL_REGISTRY.items():
        path = os.path.join(model_dir, spec["filename"])
        kind = spec["kind"]

        if not os.path.isfile(path):
            print(f"{spec['label']} weights not found at {path}")
            continue

        try:
            if kind == "pytorch_cnn":
                _loaded[key] = _load_pytorch_cnn(path)
            elif kind == "keras_transformer":
                _loaded[key] = _load_keras_model(path, needs_custom_objects=True)
            elif kind == "keras_board":
                _loaded[key] = _load_keras_model(path, needs_custom_objects=False)
            else:
                print(f"Unknown model kind '{kind}' for {key}")
                continue
            print(f"{spec['label']} model loaded.")
        except Exception as e:
            print(f"Failed to load {spec['label']}: {e}")


def _valid_moves(board_state: List[List[int]]) -> List[int]:
    return [c for c in range(7) if board_state[0][c] == 0]


def _mask_logits(logits: np.ndarray, board_state: List[List[int]]) -> Tuple[np.ndarray, int]:
    masked = np.full(7, -np.inf)
    for c in _valid_moves(board_state):
        masked[c] = logits[c]
    return masked, int(np.argmax(masked))


def _board_to_pytorch_input(board_state: List[List[int]]) -> np.ndarray:
    board_array = np.array(board_state)
    input_board = np.zeros((1, 2, 6, 7), dtype=np.float32)
    input_board[0, 0, :, :] = (board_array == 1).astype(np.float32)
    input_board[0, 1, :, :] = (board_array == -1).astype(np.float32)
    return input_board


def _board_to_keras_board_input(board_state: List[List[int]]) -> np.ndarray:
    board_array = np.array(board_state, dtype=np.float32)
    input_board = np.zeros((1, 6, 7, 2), dtype=np.float32)
    input_board[0, :, :, 0] = (board_array == 1).astype(np.float32)
    input_board[0, :, :, 1] = (board_array == -1).astype(np.float32)
    return input_board


def _board_to_keras_transformer_input(board_state: List[List[int]]) -> np.ndarray:
    board_array = np.array(board_state, dtype=np.float32)
    input_board = np.zeros((6, 7, 2), dtype=np.float32)
    input_board[:, :, 0] = (board_array == 1).astype(np.float32)
    input_board[:, :, 1] = (board_array == -1).astype(np.float32)
    return input_board.reshape(1, 42, 2)


def _masked_pytorch_cnn_logits(
    board_state: List[List[int]],
) -> Optional[Tuple[np.ndarray, int]]:
    model = _loaded.get("cnn")
    if model is None:
        return None
    tensor = torch.tensor(_board_to_pytorch_input(board_state))
    with torch.no_grad():
        logits = model(tensor).numpy()[0]
    return _mask_logits(logits, board_state)


def _masked_keras_transformer_logits(
    board_state: List[List[int]],
) -> Optional[Tuple[np.ndarray, int]]:
    model = _loaded.get("transformer")
    if model is None:
        return None
    preds = model.predict(_board_to_keras_transformer_input(board_state), verbose=0)[0]
    return _mask_logits(preds, board_state)


def _masked_keras_board_logits(
    model_key: str, board_state: List[List[int]]
) -> Optional[Tuple[np.ndarray, int]]:
    model = _loaded.get(model_key)
    if model is None:
        return None
    preds = model.predict(_board_to_keras_board_input(board_state), verbose=0)[0]
    return _mask_logits(preds, board_state)


def is_model_loaded(model_key: str) -> bool:
    return model_key in MODEL_REGISTRY and _loaded.get(model_key) is not None


def model_label(model_key: str) -> str:
    spec = MODEL_REGISTRY.get(model_key)
    return spec["label"] if spec else model_key


def ai_move(board_state: List[List[int]], model_key: str) -> Optional[int]:
    out = best_move_and_scores(board_state, model_key)
    if out is None:
        return None
    return out[0]


def best_move_and_scores(
    board_state: List[List[int]], model_key: str
) -> Optional[Tuple[int, np.ndarray]]:
    """
    Returns (best_column, masked_logits_length_7) with -inf for illegal columns.
    None if the requested model is not loaded.
    """
    spec = MODEL_REGISTRY.get(model_key)
    if spec is None:
        return None

    kind = spec["kind"]
    if kind == "pytorch_cnn":
        out = _masked_pytorch_cnn_logits(board_state)
    elif kind == "keras_transformer":
        out = _masked_keras_transformer_logits(board_state)
    elif kind == "keras_board":
        out = _masked_keras_board_logits(model_key, board_state)
    else:
        return None

    if out is None:
        return None
    masked, best = out
    return best, masked
