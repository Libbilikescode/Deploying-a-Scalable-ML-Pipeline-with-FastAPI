#import pytest
# TODO: add necessary import
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def load_data():
    """Load census data for tests."""
    data_path = os.path.join(os.getcwd(), "data", "census.csv")
    return pd.read_csv(data_path)

# TODO:Test 1: Verify that the dataset loads correctly and splits into the expected train/test sizes.
def test_train_test_split_size():
    """Test that the census data splits into expected train and test sizes."""
    data = load_data()
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    assert data.shape == (32561, 15)
    assert train.shape[0] == 26048
    assert test.shape[0] == 6513
    assert "salary" in data.columns


# TODO:Test 2: Verify that compute_model_metrics returns expected precision, recall, and F1 values.
def test_compute_model_metrics_expected_values():
    """Test that compute_model_metrics returns expected metric values."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 0.5
    assert round(fbeta, 4) == 0.6667


# TODO:# Test 3: Verify that train_model returns the expected model type and inference returns predictions.
def test_train_model_algorithm_and_inference():
    """Test that train_model uses RandomForestClassifier and predicts labels."""
    data = load_data()
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(model, RandomForestClassifier)
    assert len(preds) == len(y_test)
