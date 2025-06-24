import pandas as pd
from sklearn.model_selection import train_test_split
import joblib  # to save model and encoders

from model.data import process_data
from model.model import (
    train_model,
    compute_model_metrics,
    inference,
    evaluate_model_on_slices,
)


def main():
    # Load data (adjust path as needed)
    data = pd.read_csv("data/census.csv")

    # Split data
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process test data using the encoders from training
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Run inference on test data
    preds = inference(model, X_test)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(
        f"Precision: {
            precision:.4f}, Recall: {
            recall:.4f}, F1: {
                fbeta:.4f}"
    )

    # Save model and encoders for later use
    joblib.dump(model, "model/random_forest.joblib")
    joblib.dump(encoder, "model/encoder.joblib")
    joblib.dump(lb, "model/label_binarizer.joblib")

    # Evaluate performance on slices of data by a categorical feature, e.g.,
    # 'education'
    slice_results = evaluate_model_on_slices(
        model=model,
        data=test,
        categorical_feature="education",
        label="salary",
        cat_features=cat_features,
        encoder=encoder,
        lb=lb,
        process_data_fn=process_data,
        compute_metrics_fn=compute_model_metrics,
        inference_fn=inference,
    )

    print("Model Performance on Slices of Data (by Education):")
    for category_value, metrics in slice_results.items():
        print(
            f"education = {category_value}: Precision={
                metrics['precision']:.3f}, Recall={
                metrics['recall']:.3f}, F1={
                metrics['fbeta']:.3f}"
        )


if __name__ == "__main__":
    main()
