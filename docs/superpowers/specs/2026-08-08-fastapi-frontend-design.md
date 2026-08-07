# Replace Streamlit UI with a FastAPI + static frontend

## Context

The app currently ships as a single Streamlit script (`app.py`) that loads
`sleep_disorder_model.pkl` and renders a form UI. Streamlit was judged
unsuitable once a custom-designed mockup existed for the app (a self-executing
HTML bundle export, `Sleep Disorder Predictor (standalone).html`, pasted into
this session) — Streamlit can't reproduce custom CSS/layout, only its own
widget styling.

The mockup's embedded `Component extends DCLogic` class contains only
placeholder prediction rules (e.g. `stressLevel >= 8 -> Insomnia`); it exists
to demonstrate the visual design, not real model logic. Inspecting the pickle
confirms it expects exactly the 11 fields `app.py` already builds, in the same
encoding, and returns one of three labels:

```
feature_names_in_ = ['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
  'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate',
  'Daily Steps', 'BP Systolic', 'BP Diastolic']
classes_ = ['Insomnia', 'None', 'Sleep Apnea']
```

## Goal

Replace the Streamlit UI with a real frontend that matches the mockup's
design, backed by a small FastAPI service that runs the actual model (not the
mockup's placeholder rules).

## Architecture

One FastAPI app, run with a single `uvicorn` command, serving both the static
frontend and the prediction API — no separate frontend server, no CORS
config needed.

- `GET /` and static assets → the frontend (plain HTML/CSS/JS, no framework,
  no build step)
- `POST /predict` → JSON body with the 11 form fields → JSON response with
  the predicted label
- Model loads once at process startup (module-level), replacing the
  `@st.cache_resource` pattern from `app.py`

## Components

**Frontend** (`static/`)
- `index.html` — header, form section, loading section, result section;
  JS toggles which section is visible (three states: form / loading / result)
- `styles.css` — ported from the mockup's design tokens: `#f3f2f2`
  background, `#ec3013` accent, Archivo font (Google Fonts, with a
  system-font fallback), flat/no-radius buttons and segmented controls,
  range sliders with live value labels next to each label
- `app.js` — vanilla JS, no framework:
  - in-memory state object for the 11 fields (mirrors the mockup's state
    shape: gender, age, sleepDuration, qualityOfSleep, physicalActivity,
    stressLevel, bmi, heartRate, dailySteps, bpSystolic, bpDiastolic)
  - segmented-control click handlers for Gender and BMI Category (toggle
    `active` class, matching the mockup's `.seg-opt` styling)
  - slider/number `input` handlers update state and the live value label
  - "Predict Sleep Disorder" click → show loading section → `fetch
    POST /predict` → on success show result section; on failure show an
    inline error with a way to retry
  - result content (title / tag label / tag class / message / suggestions)
    keyed by the three possible labels, copied from the mockup's
    `resultContent` object:
    - `None` → "No Sleep Disorder Detected", outline tag, no suggestions
    - `Insomnia` → "Insomnia Risk", accent tag, 3 suggestions
    - `Sleep Apnea` → "Sleep Apnea Risk", accent tag, 3 suggestions
  - "Run another prediction" resets to the form section

**Backend** (`main.py`)
- Pydantic request model for the 11 fields, with ranges matching the
  sliders/number inputs (e.g. age 18-80, stress 1-10)
- On `/predict`: build a single-row `pandas.DataFrame` with the exact
  column names/order `feature_names_in_` expects (encoding `Gender` and
  `BMI Category` the same way `app.py` does today), call `model.predict`,
  return `{"result": "<label>"}`
- `requirements.txt`: `fastapi`, `uvicorn`, `scikit-learn`, `pandas`

**Removed**
- `app.py` (Streamlit) and the Streamlit dependency

## Data flow

Browser form state → `POST /predict` (JSON) → FastAPI validates via
Pydantic → builds the model's expected DataFrame → `model.predict` → label
string returned → frontend maps label to result copy/icon/tag and renders
the result section.

## Error handling

- Backend: Pydantic rejects out-of-range/malformed input with 422; the
  process fails fast at startup if the pickle can't load
- Frontend: a failed `fetch` (network or 5xx) shows an inline error message
  in place of the result, with a way to go back and retry, instead of
  leaving the UI stuck on the loading screen

## Testing

Manual only, given the project's size — run `uvicorn`, exercise all three
outcomes in the browser:
- high stress or low sleep quality → Insomnia
- obese/overweight BMI + heart rate > 85 → Sleep Apnea (mirroring the
  mockup's placeholder thresholds is not required; the real model decides)
- moderate values → None

Confirm the UI matches the mockup's layout and that all three result states
render correctly.
