from sklearn.ensemble import RandomForestClassifier

from model.data import process_data
from model.model import compute_model_metrics, inference
from model.train_model import (
    evaluate_model_on_slices,
)  # adjust import if needed


def test_evaluate_model_on_slices(slicing_data, cat_features=["education"]):

    # Train the model on this small data
    X_train, y_train, encoder, lb = process_data(
        slicing_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Evaluate slicing on same data (just for test)
    results = evaluate_model_on_slices(
        model=model,
        data=slicing_data,
        categorical_feature="education",
        label="salary",
        cat_features=cat_features,
        encoder=encoder,
        lb=lb,
        process_data_fn=process_data,
        compute_metrics_fn=compute_model_metrics,
        inference_fn=inference,
    )

    # Check keys and metrics presence
    assert "Bachelors" in results
    assert "HS-grad" in results
    for metrics in results.values():
        assert "precision" in metrics
        assert "recall" in metrics
        assert "fbeta" in metrics
        # Metrics should be floats between 0 and 1
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["fbeta"] <= 1.0
