import json
from app.main import app


def test_health_endpoint():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health endpoint returned {resp.status_code}"
    data = resp.get_json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
