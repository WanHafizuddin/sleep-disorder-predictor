from fastapi.testclient import TestClient

from main import BMI_MAP, GENDER_MAP, PredictionRequest, app, build_features

client = TestClient(app)

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


def test_root_serves_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="form-screen"' in response.text


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
