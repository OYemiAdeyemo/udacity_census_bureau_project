from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API!"
    }


def test_predict_below_30():
    data = {
        "age": 25,
        "workclass": "Private",
        "education-num": 10,
        "marital-status": "Married",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Female",
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    # random_forest model predicts 0 for age <=30
    assert response.json()["prediction"] == 0


def test_predict_above_30():
    data = {
        "age": 45,
        "workclass": "State-gov",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    # random_forest model predicts 1 for age > 30
    assert response.json()["prediction"] == 1
