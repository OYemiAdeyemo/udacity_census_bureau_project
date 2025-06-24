import numpy as np
from model.data import process_data
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_process_data_training(sample_data):
    categorical_features = ["workclass", "education"]
    label = "salary"

    X_processed, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label=label,
        training=True,
    )

    # Check output shapes
    assert X_processed.shape[0] == 2
    assert len(y) == 2
    assert isinstance(X_processed, np.ndarray)
    assert isinstance(y, np.ndarray)

    # Check encoder and label binarizer
    assert encoder is not None
    assert lb is not None
    assert hasattr(encoder, "transform")
    assert hasattr(lb, "transform")


def test_process_data_inference(sample_data):
    categorical_features = ["workclass", "education"]
    label = "salary"

    # First, fit using training=True to get encoder and lb
    _, _, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label=label,
        training=True,
    )

    # Then, run inference
    X_processed, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert X_processed.shape[0] == 2
    assert len(y) == 2
