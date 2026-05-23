#!/usr/bin/env python3
"""Flask entry point — registers model API and web GUI blueprints."""

from __future__ import annotations

import os
import secrets

from flask import Flask

from config import allow_lan_access, title
from model.inference import load_models

app = Flask(title)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

load_models()

from gui.api import gui
from model.api import model

app.register_blueprint(model, url_prefix="/model")
app.register_blueprint(gui, url_prefix="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = "0.0.0.0" if allow_lan_access else "127.0.0.1"
    app.run(
        host=host,
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
