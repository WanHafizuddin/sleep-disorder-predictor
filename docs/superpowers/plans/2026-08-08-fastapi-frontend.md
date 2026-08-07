# FastAPI + Static Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit app with a single FastAPI service that serves a hand-built HTML/CSS/JS frontend (matching the approved mockup) and a `/predict` endpoint backed by the real `sleep_disorder_model.pkl`.

**Architecture:** One FastAPI app (`main.py`) mounts a `static/` directory (HTML/CSS/JS, no framework, no build step) at `/` and exposes `POST /predict`, which validates input with Pydantic, builds the exact feature DataFrame the model expects, and returns the predicted label as JSON. The frontend is plain JS: it tracks form state, POSTs to `/predict`, and swaps between three screens (form / loading / result).

**Tech Stack:** Python, FastAPI, Pydantic, pandas, scikit-learn (via the existing pickle), uvicorn, pytest + httpx for backend tests. Vanilla HTML/CSS/JS for the frontend (no npm, no build step).

## Global Constraints

- The model's `feature_names_in_` (fixed, from inspecting the pickle) is: `Gender`, `Age`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `BMI Category`, `Heart Rate`, `Daily Steps`, `BP Systolic`, `BP Diastolic` — column names and order must match exactly.
- `Gender` encoding: `Male` → `1`, `Female` → `0` (same as the current `app.py`).
- `BMI Category` encoding: `Underweight` → `0`, `Normal` → `1`, `Overweight` → `2`, `Obese` → `3` (same as the current `app.py`).
- The model's `classes_` are exactly `Insomnia`, `None`, `Sleep Apnea` — the frontend's result copy is keyed by these three strings verbatim.
- Field ranges (from the mockup's sliders/inputs, enforced via Pydantic): age 18-80, sleep duration 4.0-10.0 (step 0.1), quality of sleep 1-10, physical activity 0-120, stress level 1-10, heart rate 40-120, daily steps 0-20000, BP systolic 80-200, BP diastolic 50-130.
- One FastAPI app serves both the static frontend and the API — no separate frontend server, no CORS configuration.
- Design tokens from the approved mockup: background `#f3f2f2`, surface `#eae9e9`, text `#201e1d`, accent `#ec3013`; heading font weight 800; flat corners (no border-radius); Archivo font loaded from Google Fonts with a `system-ui, sans-serif` fallback.
- Result copy (title / tag label / tag style / message / suggestions) is fixed per label, copied from the approved mockup — see Task 4.
- No automated frontend tests — per the spec, frontend testing is manual (open the browser, exercise all three outcomes). Backend gets automated pytest tests.

---

### Task 1: Project scaffold — FastAPI app serving static files

**Files:**
- Create: `requirements.txt`
- Create: `main.py`
- Create: `static/index.html` (placeholder, replaced in Task 4)
- Create: `tests/test_main.py`
- Delete: `app.py`

**Interfaces:**
- Produces: `main.py` exposes a module-level `app` (FastAPI instance) that later tasks import and extend.

- [ ] **Step 1: Remove the old Streamlit app**

```bash
git rm app.py
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi
uvicorn[standard]
scikit-learn
pandas
pytest
httpx
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 4: Create the placeholder static page**

Create `static/index.html`:

```html
<!DOCTYPE html>
<html>
<head><title>Sleep Disorder Predictor</title></head>
<body><p>placeholder</p></body>
</html>
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_main.py`:

```python
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_serves_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "placeholder" in response.text
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'` (or import error) since `main.py` doesn't exist yet.

- [ ] **Step 7: Create `main.py`**

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: scaffold FastAPI app serving static files, remove Streamlit app"
```

---

### Task 2: `/predict` endpoint backed by the real model

**Files:**
- Modify: `main.py` (insert model loading, Pydantic request model, `build_features`, and the `/predict` route between `app = FastAPI()` and the `app.mount(...)` line)
- Modify: `tests/test_main.py` (add prediction tests)

**Interfaces:**
- Consumes: `app` from Task 1 (`main.py`).
- Produces: `main.PredictionRequest` (Pydantic model, fields listed below), `main.build_features(payload: PredictionRequest) -> pandas.DataFrame`, and the `POST /predict` route returning `{"result": "<Insomnia|None|Sleep Apnea>"}`. Later tasks (frontend) POST to `/predict` with the exact JSON field names: `gender`, `age`, `sleep_duration`, `quality_of_sleep`, `physical_activity`, `stress_level`, `bmi_category`, `heart_rate`, `daily_steps`, `bp_systolic`, `bp_diastolic`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_main.py`:

```python
from main import BMI_MAP, GENDER_MAP, PredictionRequest, build_features

VALID_PAYLOAD = {
    "gender": "Male",
    "age": 30,
    "sleep_duration": 7.0,
    "quality_of_sleep": 7,
    "physical_activity": 45,
    "stress_level": 5,
    "bmi_category": "Normal",
    "heart_rate": 72,
    "daily_steps": 7000,
    "bp_systolic": 120,
    "bp_diastolic": 80,
}


def test_build_features_encodes_gender_and_bmi():
    payload = PredictionRequest(**{**VALID_PAYLOAD, "gender": "Female", "bmi_category": "Obese"})
    features = build_features(payload)
    assert list(features.columns) == [
        "Gender", "Age", "Sleep Duration", "Quality of Sleep",
        "Physical Activity Level", "Stress Level", "BMI Category",
        "Heart Rate", "Daily Steps", "BP Systolic", "BP Diastolic",
    ]
    row = features.iloc[0]
    assert row["Gender"] == GENDER_MAP["Female"]
    assert row["BMI Category"] == BMI_MAP["Obese"]
    assert row["Age"] == 30
    assert row["Sleep Duration"] == 7.0


def test_predict_returns_a_known_label():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["result"] in {"Insomnia", "None", "Sleep Apnea"}


def test_predict_rejects_out_of_range_age():
    response = client.post("/predict", json={**VALID_PAYLOAD, "age": 999})
    assert response.status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'BMI_MAP'` (or similar) since these don't exist in `main.py` yet.

- [ ] **Step 3: Implement the endpoint**

Replace the contents of `main.py` with:

```python
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "sleep_disorder_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

GENDER_MAP = {"Male": 1, "Female": 0}
BMI_MAP = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}


class PredictionRequest(BaseModel):
    gender: str = Field(pattern="^(Male|Female)$")
    age: int = Field(ge=18, le=80)
    sleep_duration: float = Field(ge=4.0, le=10.0)
    quality_of_sleep: int = Field(ge=1, le=10)
    physical_activity: int = Field(ge=0, le=120)
    stress_level: int = Field(ge=1, le=10)
    bmi_category: str = Field(pattern="^(Underweight|Normal|Overweight|Obese)$")
    heart_rate: int = Field(ge=40, le=120)
    daily_steps: int = Field(ge=0, le=20000)
    bp_systolic: int = Field(ge=80, le=200)
    bp_diastolic: int = Field(ge=50, le=130)


def build_features(payload: PredictionRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "Gender": GENDER_MAP[payload.gender],
        "Age": payload.age,
        "Sleep Duration": payload.sleep_duration,
        "Quality of Sleep": payload.quality_of_sleep,
        "Physical Activity Level": payload.physical_activity,
        "Stress Level": payload.stress_level,
        "BMI Category": BMI_MAP[payload.bmi_category],
        "Heart Rate": payload.heart_rate,
        "Daily Steps": payload.daily_steps,
        "BP Systolic": payload.bp_systolic,
        "BP Diastolic": payload.bp_diastolic,
    }])


@app.post("/predict")
def predict(payload: PredictionRequest):
    features = build_features(payload)
    result = model.predict(features)[0]
    return {"result": result}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (all tests, including Task 1's `test_root_serves_index_html`)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add /predict endpoint backed by the real model"
```

---

### Task 3: Frontend structure and styling (static, no interactivity yet)

**Files:**
- Modify: `static/index.html` (replace placeholder with the full mockup structure)
- Create: `static/styles.css`

**Interfaces:**
- Consumes: the static-file serving from Task 1 (`main.py` mounts `static/` at `/`).
- Produces: DOM element IDs that Task 4's `app.js` wires up: `form-screen`, `loading-screen`, `result-screen`, `gender-seg`, `bmi-seg`, `age` / `age-value`, `sleep-duration` / `sleep-duration-value`, `quality-of-sleep` / `quality-of-sleep-value`, `physical-activity` / `physical-activity-value`, `stress-level` / `stress-level-value`, `heart-rate`, `daily-steps`, `bp-systolic`, `bp-diastolic`, `predict-btn`, `result-icon`, `result-title`, `result-tag`, `result-message`, `suggestions-block`, `suggestions-list`, `reset-btn`. CSS classes: `.hidden` (display:none toggle), `.seg-opt` / `.active` (segmented control state), `.tag-outline` / `.tag-accent` (result tag style).

- [ ] **Step 1: Replace `static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sleep Disorder Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/styles.css">
</head>
<body>
<div class="page">

  <div class="header-row">
    <div>
      <h1>Sleep Disorder Predictor</h1>
      <p class="subtitle">Fill in your health and lifestyle details, and we'll estimate your sleep disorder risk.</p>
    </div>
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="1.6"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path></svg>
  </div>

  <hr class="hr">

  <section id="form-screen">
    <div class="form-grid">
      <div class="form-col">

        <div class="field-block">
          <label>Gender</label>
          <div class="seg" id="gender-seg">
            <button type="button" class="seg-opt active" data-value="Male">Male</button>
            <button type="button" class="seg-opt" data-value="Female">Female</button>
          </div>
        </div>

        <div class="field-block">
          <label class="label-row"><span>Age</span><span id="age-value">30</span></label>
          <input type="range" id="age" min="18" max="80" value="30">
        </div>

        <div class="field-block">
          <label class="label-row"><span>Sleep Duration (hours)</span><span id="sleep-duration-value">7.0</span></label>
          <input type="range" id="sleep-duration" min="4" max="10" step="0.1" value="7.0">
        </div>

        <div class="field-block">
          <label class="label-row"><span>Quality of Sleep (1-10)</span><span id="quality-of-sleep-value">7</span></label>
          <input type="range" id="quality-of-sleep" min="1" max="10" value="7">
        </div>

        <div class="field-block">
          <label class="label-row"><span>Physical Activity (min/day)</span><span id="physical-activity-value">45</span></label>
          <input type="range" id="physical-activity" min="0" max="120" value="45">
        </div>

        <div class="field-block">
          <label class="label-row"><span>Stress Level (1-10)</span><span id="stress-level-value">5</span></label>
          <input type="range" id="stress-level" min="1" max="10" value="5">
        </div>

      </div>

      <div class="form-col">

        <div class="field-block">
          <label>BMI Category</label>
          <div class="seg" id="bmi-seg">
            <button type="button" class="seg-opt" data-value="Underweight">Underweight</button>
            <button type="button" class="seg-opt active" data-value="Normal">Normal</button>
            <button type="button" class="seg-opt" data-value="Overweight">Overweight</button>
            <button type="button" class="seg-opt" data-value="Obese">Obese</button>
          </div>
        </div>

        <div class="field">
          <label>Heart Rate (bpm)</label>
          <input class="input" type="number" id="heart-rate" min="40" max="120" value="72">
        </div>

        <div class="field">
          <label>Daily Steps</label>
          <input class="input" type="number" id="daily-steps" min="0" max="20000" step="500" value="7000">
        </div>

        <div class="field">
          <label>Blood Pressure &mdash; Systolic</label>
          <input class="input" type="number" id="bp-systolic" min="80" max="200" value="120">
        </div>

        <div class="field">
          <label>Blood Pressure &mdash; Diastolic</label>
          <input class="input" type="number" id="bp-diastolic" min="50" max="130" value="80">
        </div>

      </div>
    </div>

    <hr class="hr hr-loose">

    <button type="button" class="btn btn-primary btn-block" id="predict-btn">
      <span>Predict Sleep Disorder</span>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"></path></svg>
    </button>
  </section>

  <section id="loading-screen" class="hidden">
    <svg class="spinner" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2"><rect x="4" y="4" width="16" height="16"></rect></svg>
    <p class="loading-text">Analyzing your sleep data&hellip;</p>
  </section>

  <section id="result-screen" class="hidden">
    <p class="eyebrow">Result</p>
    <div class="result-header">
      <svg id="result-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2"></svg>
      <h2 id="result-title"></h2>
      <span class="tag" id="result-tag"></span>
    </div>
    <p class="result-message" id="result-message"></p>

    <div id="suggestions-block" class="hidden">
      <p class="eyebrow">Suggestions</p>
      <ul class="suggestions" id="suggestions-list"></ul>
    </div>

    <hr class="hr">

    <div class="result-footer">
      <button type="button" class="btn btn-secondary" id="reset-btn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 1 3 6.7M3 12V6m0 6h6"></path></svg>
        <span>Run another prediction</span>
      </button>
      <p class="disclaimer">This is a model estimate, not a medical diagnosis. If you're concerned, please see a doctor.</p>
    </div>
  </section>

</div>
<script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `static/styles.css`**

```css
:root {
  --color-bg: #f3f2f2;
  --color-surface: #eae9e9;
  --color-text: #201e1d;
  --color-accent: #ec3013;
  --color-divider: rgba(32, 30, 29, 0.4);
  --color-muted: #4b4847;
  --font-heading: "Archivo", system-ui, sans-serif;
  --font-body: "Archivo", system-ui, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--color-bg);
  color: var(--color-text);
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.55;
}

.page {
  max-width: 880px;
  margin: 0 auto;
  padding: 64px 24px 96px;
}

.header-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
}

h1 {
  font-family: var(--font-heading);
  font-weight: 800;
  font-size: 40px;
  margin: 0;
  letter-spacing: -0.01em;
}

.subtitle {
  font-size: 16px;
  color: var(--color-muted);
  margin: 12px 0 0;
  max-width: 560px;
}

.hr {
  height: 2px;
  border: 0;
  margin: 32px 0;
  background: var(--color-divider);
}
.hr-loose { margin: 40px 0 28px; }

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
}

.form-col {
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.field-block label,
.field label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin-bottom: 10px;
}

.label-row {
  display: flex;
  justify-content: space-between;
}
.label-row span:last-child {
  color: var(--color-text);
  font-weight: 700;
  text-transform: none;
  letter-spacing: normal;
  font-size: 15px;
}

input[type="range"] {
  width: 100%;
  accent-color: var(--color-accent);
}

.seg {
  display: flex;
  overflow: hidden;
  border: 1px solid var(--color-divider);
  width: 100%;
}
.seg-opt {
  flex: 1;
  padding: 7px 12px;
  font-size: 13px;
  font-family: var(--font-body);
  cursor: pointer;
  background: transparent;
  border: 0;
  border-left: 1px solid var(--color-divider);
  color: var(--color-text);
}
.seg-opt:first-child { border-left: 0; }
.seg-opt.active { background: var(--color-accent); color: var(--color-bg); }
.seg-opt:not(.active):hover { background: rgba(32, 30, 29, 0.07); }

.input {
  width: 100%;
  min-height: 36px;
  padding: 6px 10px;
  font: inherit;
  font-size: 14px;
  color: var(--color-text);
  background: var(--color-surface);
  border: 1px solid var(--color-divider);
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  cursor: pointer;
  font-family: var(--font-heading);
  font-weight: 800;
  font-size: 14px;
  padding: 8px 14px;
  border: 1px solid transparent;
}
.btn-primary { background: var(--color-accent); color: var(--color-bg); }
.btn-primary:hover { background: #dd2b0f; }
.btn-secondary { border-color: var(--color-divider); background: transparent; color: var(--color-text); }
.btn-secondary:hover { background: rgba(32, 30, 29, 0.07); }
.btn-block { width: 100%; justify-content: space-between; }

.hidden { display: none !important; }

#loading-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  padding: 96px 0;
}
.spinner { animation: spin 0.9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { font-size: 15px; color: var(--color-muted); margin: 0; }

.eyebrow {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-muted);
  margin: 0 0 12px;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 20px;
}
#result-title {
  font-family: var(--font-heading);
  font-weight: 800;
  font-size: 30px;
  margin: 0;
}

.tag {
  display: inline-flex;
  align-items: center;
  font-size: 11px;
  padding: 3px 10px;
}
.tag-outline { border: 1px solid var(--color-accent); color: var(--color-accent); }
.tag-accent { background: #fff2ef; color: #7c1405; }

.result-message {
  font-size: 16px;
  line-height: 1.6;
  max-width: 620px;
  margin: 0 0 24px;
}

.suggestions {
  list-style: none;
  margin: 0 0 32px;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.suggestions li {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  font-size: 15px;
  line-height: 1.5;
}
.suggestions li::before {
  content: "";
  width: 6px;
  height: 6px;
  margin-top: 8px;
  flex: none;
  background: var(--color-accent);
}

.result-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.disclaimer {
  font-size: 12px;
  color: var(--color-muted);
  margin: 0;
  max-width: 320px;
  text-align: right;
}

@media (max-width: 640px) {
  .form-grid { grid-template-columns: 1fr; }
  .result-footer { flex-direction: column; align-items: flex-start; }
  .disclaimer { text-align: left; }
}
```

- [ ] **Step 3: Manually verify the layout**

Run: `uvicorn main:app --reload`, open `http://127.0.0.1:8000/` in a browser.
Expected: header with title/subtitle/icon, a divider, a two-column form matching the mockup (Gender segmented control, five sliders on the left; BMI segmented control, four number inputs on the right), a full-width "Predict Sleep Disorder" button. The loading and result sections should not be visible (they're behind `.hidden`).

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/styles.css
git commit -m "feat: build static frontend structure and styling"
```

---

### Task 4: Frontend interactivity — state, API call, screen transitions

**Files:**
- Create: `static/app.js`

**Interfaces:**
- Consumes: the element IDs and CSS classes produced by Task 3; the `POST /predict` contract produced by Task 2 (request field names `gender`, `age`, `sleep_duration`, `quality_of_sleep`, `physical_activity`, `stress_level`, `bmi_category`, `heart_rate`, `daily_steps`, `bp_systolic`, `bp_diastolic`; response `{"result": "<label>"}` where label is one of `Insomnia`, `None`, `Sleep Apnea`).

- [ ] **Step 1: Create `static/app.js`**

```javascript
const state = {
  gender: "Male",
  age: 30,
  sleepDuration: 7.0,
  qualityOfSleep: 7,
  physicalActivity: 45,
  stressLevel: 5,
  bmi: "Normal",
  heartRate: 72,
  dailySteps: 7000,
  bpSystolic: 120,
  bpDiastolic: 80,
};

const RESULT_CONTENT = {
  None: {
    title: "No Sleep Disorder Detected",
    tagLabel: "Healthy",
    tagClass: "tag-outline",
    message: "Your sleep health looks good. Keep up your current routine — consistent sleep and activity habits are working in your favor.",
    suggestions: [],
    icon: '<path d="M20 6 9 17l-5-5"></path>',
  },
  Insomnia: {
    title: "Insomnia Risk",
    tagLabel: "Needs attention",
    tagClass: "tag-accent",
    message: "Your answers point to insomnia — trouble falling or staying asleep. A few small changes can make a real difference.",
    suggestions: [
      "Wind down earlier — ease off screens and stress before bed.",
      "Keep your sleep schedule steady, even on weekends.",
      "Skip caffeine in the afternoon and evening.",
    ],
    icon: '<path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"></path>',
  },
  "Sleep Apnea": {
    title: "Sleep Apnea Risk",
    tagLabel: "Needs attention",
    tagClass: "tag-accent",
    message: "Your answers suggest sleep apnea, where breathing repeatedly stops during sleep. It's worth looking into further.",
    suggestions: [
      "Talk to a doctor about a proper sleep study.",
      "Work toward a healthy BMI — it eases the load on your airway.",
      "Try sleeping on your side instead of your back.",
    ],
    icon: '<circle cx="12" cy="12" r="10"></circle><path d="M12 8v4M12 16h.01"></path>',
  },
};

const formScreen = document.getElementById("form-screen");
const loadingScreen = document.getElementById("loading-screen");
const resultScreen = document.getElementById("result-screen");

function showScreen(name) {
  formScreen.classList.toggle("hidden", name !== "form");
  loadingScreen.classList.toggle("hidden", name !== "loading");
  resultScreen.classList.toggle("hidden", name !== "result");
}

function wireSegmented(containerId, stateKey) {
  const container = document.getElementById(containerId);
  container.querySelectorAll(".seg-opt").forEach((btn) => {
    btn.addEventListener("click", () => {
      state[stateKey] = btn.dataset.value;
      container.querySelectorAll(".seg-opt").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });
}

function wireRange(inputId, labelId, stateKey, parse, format) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  input.addEventListener("input", () => {
    const value = parse(input.value);
    state[stateKey] = value;
    label.textContent = format(value);
  });
}

function wireNumber(inputId, stateKey) {
  const input = document.getElementById(inputId);
  input.addEventListener("input", () => {
    state[stateKey] = parseInt(input.value, 10) || 0;
  });
}

wireSegmented("gender-seg", "gender");
wireSegmented("bmi-seg", "bmi");
wireRange("age", "age-value", "age", (v) => parseInt(v, 10), (v) => String(v));
wireRange("sleep-duration", "sleep-duration-value", "sleepDuration", (v) => parseFloat(v), (v) => v.toFixed(1));
wireRange("quality-of-sleep", "quality-of-sleep-value", "qualityOfSleep", (v) => parseInt(v, 10), (v) => String(v));
wireRange("physical-activity", "physical-activity-value", "physicalActivity", (v) => parseInt(v, 10), (v) => String(v));
wireRange("stress-level", "stress-level-value", "stressLevel", (v) => parseInt(v, 10), (v) => String(v));
wireNumber("heart-rate", "heartRate");
wireNumber("daily-steps", "dailySteps");
wireNumber("bp-systolic", "bpSystolic");
wireNumber("bp-diastolic", "bpDiastolic");

function renderResult(label) {
  const content = RESULT_CONTENT[label] || RESULT_CONTENT.None;
  document.getElementById("result-icon").innerHTML = content.icon;
  document.getElementById("result-title").textContent = content.title;
  const tag = document.getElementById("result-tag");
  tag.textContent = content.tagLabel;
  tag.className = "tag " + content.tagClass;
  document.getElementById("result-message").textContent = content.message;

  const suggestionsBlock = document.getElementById("suggestions-block");
  const suggestionsList = document.getElementById("suggestions-list");
  suggestionsList.innerHTML = "";
  if (content.suggestions.length > 0) {
    content.suggestions.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      suggestionsList.appendChild(li);
    });
    suggestionsBlock.classList.remove("hidden");
  } else {
    suggestionsBlock.classList.add("hidden");
  }
}

function renderError(message) {
  document.getElementById("result-icon").innerHTML = "";
  document.getElementById("result-title").textContent = "Something went wrong";
  const tag = document.getElementById("result-tag");
  tag.textContent = "";
  tag.className = "tag";
  document.getElementById("result-message").textContent = message;
  document.getElementById("suggestions-block").classList.add("hidden");
}

async function predict() {
  showScreen("loading");
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gender: state.gender,
        age: state.age,
        sleep_duration: state.sleepDuration,
        quality_of_sleep: state.qualityOfSleep,
        physical_activity: state.physicalActivity,
        stress_level: state.stressLevel,
        bmi_category: state.bmi,
        heart_rate: state.heartRate,
        daily_steps: state.dailySteps,
        bp_systolic: state.bpSystolic,
        bp_diastolic: state.bpDiastolic,
      }),
    });
    if (!response.ok) {
      throw new Error("Server returned " + response.status);
    }
    const data = await response.json();
    renderResult(data.result);
  } catch (err) {
    renderError("We couldn't reach the prediction service. Please try again.");
  }
  showScreen("result");
}

document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("reset-btn").addEventListener("click", () => showScreen("form"));
```

- [ ] **Step 2: Manually test the None branch**

Run: `uvicorn main:app --reload`, open `http://127.0.0.1:8000/`.
Set moderate values (defaults are fine: Male, age 30, sleep duration 7.0, quality 7, activity 45, stress 5, BMI Normal, heart rate 72, steps 7000, BP 120/80). Click "Predict Sleep Disorder".
Expected: loading spinner appears briefly, then the result screen shows a title, a tag, a message, no suggestions block, and the "Run another prediction" button + disclaimer. Confirm the returned label is reflected correctly (check the Network tab response body against what's displayed).

- [ ] **Step 3: Manually test the Insomnia and Sleep Apnea branches**

Try inputs likely to trigger each label (e.g. push Stress Level to 10 and Quality of Sleep to 1 for one attempt; try BMI Obese with Heart Rate 110 for another) — the real model decides, so adjust inputs and re-run until each label has been observed at least once.
Expected: for `Insomnia` and `Sleep Apnea`, the result screen shows the matching title/message and a 3-item suggestions list; the tag uses the accent style (not outline).

- [ ] **Step 4: Manually test the reset flow and slider labels**

Click "Run another prediction" from a result screen.
Expected: returns to the form screen with previously-entered values still in the inputs (state isn't reset). Drag each slider and confirm its adjacent value label updates live, including Sleep Duration showing one decimal place (e.g. `7.3`).

- [ ] **Step 5: Manually test the error path**

Stop the `uvicorn` server (Ctrl+C) while the form screen is open, then click "Predict Sleep Disorder".
Expected: after the loading screen, the result screen shows "Something went wrong" with the retry message, no tag text, no suggestions, and a working "Run another prediction" button. Restart the server afterward.

- [ ] **Step 6: Commit**

```bash
git add static/app.js
git commit -m "feat: wire up frontend state, prediction call, and screen transitions"
```
