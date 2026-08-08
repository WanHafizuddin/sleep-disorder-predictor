# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt      # install deps
uvicorn main:app --reload            # run dev server at http://127.0.0.1:8000/
pytest                                # run all tests
pytest tests/test_main.py::test_predict_returns_a_known_label -v   # run a single test
```

`pytest.ini` sets `pythonpath = .`, so `pytest` works from the repo root without any `PYTHONPATH` workaround — don't reintroduce one.

On Windows, `uvicorn --reload` can print `WinError 10013` from its reloader/watcher process even when the underlying worker has already bound the port successfully — check `curl http://127.0.0.1:8000/` before assuming the server failed to start. If the port really is stuck, it's usually a previous `uvicorn` process left running; find and kill it before relaunching.

## Architecture

One FastAPI app (`main.py`) does two things:
- Serves `POST /predict`, a JSON API backed by the pickled model.
- Mounts `static/` at `/` (`StaticFiles(..., html=True)`), serving the hand-written frontend. The `/predict` route is registered before the mount, which is what lets the specific route win over the catch-all static mount.

There is no separate frontend server, build step, or JS framework — `static/app.js` is vanilla JS with no dependencies, and `static/index.html`/`static/styles.css` are hand-written to match an approved design mockup (see `docs/superpowers/specs/2026-08-08-fastapi-frontend-design.md`).

### The model contract (`main.py`)

`sleep_disorder_model.pkl` is a `RandomForestClassifier` with a fixed, order-sensitive feature contract — inspect it directly (`model.feature_names_in_`, `model.classes_`) rather than assuming, if this ever needs to change:

- `feature_names_in_`: `Gender`, `Age`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `BMI Category`, `Heart Rate`, `Daily Steps`, `BP Systolic`, `BP Diastolic` — `build_features()` must build the DataFrame with exactly these column names in this order.
- `Gender` encoding: `Male` → `1`, `Female` → `0`.
- `BMI Category` encoding: `Underweight` → `0`, `Normal` → `1`, `Overweight` → `2`, `Obese` → `3`.
- `classes_`: exactly `Insomnia`, `None`, `Sleep Apnea` — these three strings are the contract between the backend and the frontend's result copy (see below).

`PredictionRequest` (Pydantic) enforces the same ranges as the frontend's slider/number-input `min`/`max` attributes; if one changes, update the other.

### The frontend/backend contract (`static/app.js` ↔ `main.py`)

`app.js`'s `fetch("/predict")` body keys (`gender`, `age`, `sleep_duration`, `quality_of_sleep`, `physical_activity`, `stress_level`, `bmi_category`, `heart_rate`, `daily_steps`, `bp_systolic`, `bp_diastolic`) must match `PredictionRequest`'s field names exactly — these are snake_case, distinct from the model's own space-separated column names above.

`app.js`'s `RESULT_CONTENT` object is keyed by the three `classes_` strings (`None`, `Insomnia`, `Sleep Apnea`) and holds the title/tag/message/suggestions copy shown for each. A 4th path, `renderError()`, handles both `422` (validation failure — shown as an "out of range" message) and other failures (shown as a generic "couldn't reach service" message) separately; don't collapse them back into one message.

### The frontend/markup contract (`static/app.js` ↔ `static/index.html`)

`app.js` wires itself up entirely via `getElementById`/`querySelector` against IDs and classes defined in `index.html` (e.g. `form-screen`, `loading-screen`, `result-screen`, `gender-seg`, `bmi-seg`, `age`/`age-value`, ..., `predict-btn`, `result-icon`, `result-title`, `result-tag`, `result-message`, `suggestions-block`, `suggestions-list`, `reset-btn`) and CSS classes (`.hidden`, `.seg-opt`/`.active`, `.tag-outline`/`.tag-accent`). There's no templating layer — if you rename an ID in one file, grep for it in the other two.

`state` in `app.js` is intentionally seeded by reading the DOM's current values (not hardcoded defaults) at startup, so it stays in sync with whatever the browser restores on a soft reload.

### Design reference

`docs/superpowers/specs/2026-08-08-fastapi-frontend-design.md` and `docs/superpowers/plans/2026-08-08-fastapi-frontend.md` document the design tokens (`#f3f2f2` background, `#ec3013` accent, Archivo font, flat/no-radius components) and the full implementation plan this codebase was built from — check there before re-deriving layout/styling decisions from scratch.
