import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    data = {
        "age": [25, 38],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "salary": [">50K", "<=50K"],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def slicing_data():
    data = {
        "education": ["Bachelors", "HS-grad", "Bachelors", "HS-grad"],
        "age": [25, 40, 35, 50],
        "salary": [1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    return df
