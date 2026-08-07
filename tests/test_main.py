from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_serves_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "placeholder" in response.text
