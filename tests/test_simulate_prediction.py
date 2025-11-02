from app.main import app


def test_predict_numeric_post():
    client = app.test_client()

    form = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2",
    }

    resp = client.post("/predict", data=form)
    assert resp.status_code == 200, f"Predict endpoint returned {resp.status_code}"
    body = resp.get_data(as_text=True)

    # The template formats prediction with 'Confidence' text and an emoji/name
    assert "Confidence" in body, "Response does not contain 'Confidence'"
    # If an image is returned, ensure it points to the static images folder
    assert ("/static/images/" in body) or ("Iris" in body), "Response doesn't include expected content"
