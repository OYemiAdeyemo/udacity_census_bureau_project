import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from model.data import process_data
from model.train_model import train_model, inference, compute_model_metrics
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_data():
    data = {
        "age": [25, 38, 28],
        "workclass": ["Private", "Self-emp-not-inc", "Private"],
        "education": ["Bachelors", "HS-grad", "Masters"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
        "occupation": ["Tech-support", "Exec-managerial", "Sales"],
        "relationship": ["Not-in-family", "Husband", "Unmarried"],
        "race": ["White", "Black", "White"],
        "sex": ["Male", "Male", "Female"],
        "native-country": ["United-States", "United-States", "United-States"],
        "salary": [">50K", "<=50K", ">50K"],
    }
    return pd.DataFrame(data)


def test_train_and_inference(sample_data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = "salary"

    # Process training data
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label=label,
        training=True,
    )

    # Train model
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

    # Inference
    preds = inference(model, X)
    assert preds.shape == y.shape

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_process_data_inference(sample_data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = "salary"

    # Fit encoders on training data
    _, _, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label=label,
        training=True,
    )

    # Process inference data without label
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert isinstance(X, np.ndarray)
    assert y.size == 0  # y should be empty array since label=None
