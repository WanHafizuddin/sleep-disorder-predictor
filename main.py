from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
