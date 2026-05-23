# Connect 4

Play Connect Four in the browser against trained neural networks, or query move predictions through a JSON API.

## Project overview: two-part learning pipeline

This project compares Connect Four agents trained in **two stages**. The work originally shipped on **AWS Lightsail with Docker** and an **Anvil** backend. Now it runs locally as a **Flask** web app.

### Part 1 — Supervised learning (`supervised_learning_with_AWS_Docker/`)

Board–move pairs are generated from expert or simulated play, then used to train models that predict the best column from a position.

1. **Data generation** — `C4_data_generation.ipynb` builds `.npy` datasets saved under `data/`.
2. **Model training** — `create_connect4_models.ipynb` trains the **CNN** (PyTorch) and **Transformer** (Keras) checkpoints.
3. **Cloud deployment (original)** — The trained weights were served through `backend.py`, an **Anvil uplink** that loaded CNN and Transformer models inside a **Docker** container on **AWS Lightsail**. Setup notes and screenshots live in this folder (`README-Docker.txt`, `Dockerfile`, `docker-compose.yml`, `aws_output_screenshots.pdf`).

Supervised models learn directly from labeled examples: given a board, imitate the recorded move.

### Part 2 — Reinforcement learning (`unsupervised_learning_comparisons/`)

Agents learn by **playing the game**, receiving reward for wins and penalties for losses, without a fixed move label for every position.

Notebooks in this folder train and compare:

- **DQN** (Deep Q-Network) — `train_DQN.ipynb`
- **Policy Gradient (PG)** — `train_PG.ipynb`
- **Combined experiments and comparisons** — `reinforcement_learning_combined_code.ipynb`, `pg_vs_dqn.ipynb`

These produce the Keras weight files (`dqn.keras`, `pg.keras`, and ensemble components) used in the demo app alongside the supervised models.

Written reports: `Supervised_Learning_Notes.pdf` and `Unsupervised_Learning_Report.pdf` (project root).

### From AWS + Docker → Flask app


| Stage                       | How models were served                                      |
| --------------------------- | ----------------------------------------------------------- |
| **Original (course / AWS)** | Docker image on Lightsail → `backend.py` → Anvil frontend   |
| **Current (local demo)**    | `app/` Flask server → browser UI at `http://127.0.0.1:5000` |


The Flask app reuses the same weight files but replaces Anvil with:

- A **web UI** (`app/gui/`) — pick a model, play on the board
- A **JSON API** (`app/model/api.py`) — `POST /model/best_move`
- **Local inference** (`app/model/inference.py`) — loads all six model types at startup

Docker/AWS artifacts are kept in `supervised_learning_with_AWS_Docker/` for reference; day-to-day play uses the Flask app only.

---

## How to run

Place model weight files in `app/models/` (see [Available models](#available-models)), then from a terminal:

```powershell
cd connect4\app
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe api.py
```

Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

On startup, the terminal prints which models loaded successfully. Restart the server after changing config or adding new weights.

**After first-time setup**, you usually only need:

```powershell
cd connect4\app
.\.venv\Scripts\python.exe api.py
```

### JSON API

With the server running:

```http
POST http://127.0.0.1:5000/model/best_move
Content-Type: application/json

{
  "model": "cnn",
  "moves": [3, 3, 4, 4, 5]
}
```

Send either a `moves` list (column indices played so far) or a full 6×7 `board`. The response includes the recommended column and per-column scores.

---

## File structure

```
connect4
├── app/                                    # Flask demo — run from here
│   ├── api.py                              # Entry point; starts the server
│   ├── config.py                           # Title, base URL, allow_lan_access
│   ├── requirements.txt
│   ├── gui/api.py                          # Web UI (/, /play)
│   ├── model/
│   │   ├── api.py                          # JSON API (/model/best_move)
│   │   ├── game_logic.py                   # Board rules, win detection
│   │   └── inference.py                    # Model registry, loading, prediction
│   ├── templates/                          # HTML (start screen, game board)
│   ├── static/css/                         # Stylesheets
│   └── models/                             # Weights the server loads (gitignored)
│
├── supervised_learning_with_AWS_Docker/    # Part 1: supervised pipeline + AWS/Docker
│   ├── data/                               # Training .npy datasets
│   ├── C4_data_generation.ipynb            # Generate supervised training data
│   ├── create_connect4_models.ipynb        # Train CNN & Transformer
│   ├── backend.py                          # Legacy Anvil uplink (AWS deployment)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── README-Docker.txt                   # AWS Lightsail + Docker setup notes
│   └── aws_output_screenshots.pdf
│
├── unsupervised_learning_comparisons/      # Part 2: RL agents (DQN, PG, ensembles)
│   ├── train_DQN.ipynb
│   ├── train_PG.ipynb
│   ├── reinforcement_learning_combined_code.ipynb
│   └── pg_vs_dqn.ipynb
│
├── Supervised_Learning_Notes.pdf
├── Unsupervised_Learning_Report.pdf
└── README.md
```


| Path                                        | Purpose                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| `app/`                                      | Local Flask app — play in the browser or call the API                           |
| `supervised_learning_with_AWS_Docker/`      | Data generation, CNN/Transformer training, original AWS/Docker/Anvil deployment |
| `unsupervised_learning_comparisons/`        | DQN, PG, and ensemble training notebooks                                        |
| `app/models/`                               | Checkpoints loaded by the Flask server at startup                               |
| `supervised_learning_with_AWS_Docker/data/` | Supervised training datasets; not required to run the demo                      |


---

## Available models

Weights go in `app/models/`. Models are registered in `app/model/inference.py` (`MODEL_REGISTRY`). On the start page, any model whose file is present and loads without error appears as a selectable option; missing or broken weights show as “not loaded”.


| Key            | Display name    | File                           | Source                 | Backend            |
| -------------- | --------------- | ------------------------------ | ---------------------- | ------------------ |
| `cnn`          | CNN             | `cnn_connect4.pt`              | Supervised             | PyTorch            |
| `transformer`  | Transformer     | `transformer_connect4.keras`   | Supervised             | Keras / TensorFlow |
| `dqn`          | DQN             | `dqn.keras`                    | Reinforcement learning | Keras / TensorFlow |
| `pg`           | Policy Gradient | `pg.keras`                     | Reinforcement learning | Keras / TensorFlow |
| `ensemble_dqn` | Ensemble DQN    | `ensemble_dqn_component.keras` | Reinforcement learning | Keras / TensorFlow |
| `ensemble_pg`  | Ensemble PG     | `ensemble_pg_component.keras`  | Reinforcement learning | Keras / TensorFlow |


To add another model, put its weight file in `app/models/` and add one entry to `MODEL_REGISTRY` in `inference.py`.

---

## Configuration

### `allow_lan_access` (`app/config.py`)

Controls which network interfaces the server binds to when you run `api.py`:


| Value             | Behavior                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `False` (default) | **Localhost only** — reachable at `http://127.0.0.1:5000` from local computer                                                   |
| `True`            | **Localhost + LAN** — also reachable at your local IP (e.g. `http://192.168.1.185:5000`) from other devices on the same network |


Edit `app/config.py`, then restart the server.

### Environment variables


| Variable      | Default                         | Meaning                                                            |
| ------------- | ------------------------------- | ------------------------------------------------------------------ |
| `MODEL_DIR`   | `./models` (relative to `app/`) | Directory containing weight files                                  |
| `PORT`        | `5000`                          | HTTP port                                                          |
| `SECRET_KEY`  | random per run                  | Flask session signing key; set a fixed value in production         |
| `FLASK_DEBUG` | off                             | Set to `1` for auto-reload and debug tracebacks (development only) |


---

## About the Flask app

The app is a small [Flask](https://flask.palletsprojects.com/) server:

**Web UI (`gui/` blueprint)** — Serves HTML at `/` and `/play`. Game state (move list, chosen model, colors) is stored in a Flask **session** cookie.

**JSON API (`model/` blueprint)** — `POST /model/best_move` for programmatic access. Weights stay on the server.

**Inference (`model/inference.py`)** — Loads all registered models at startup. PyTorch runs the CNN. TensorFlow/Keras run the other models.

**Frontend** — Jinja2 templates and static CSS. No separate JavaScript build step.

The last played disc is highlighted with a white ring on the board so model replies are easier to spot after a fast turn.

This project uses Flask’s built-in development server (`app.run()` in `api.py`).