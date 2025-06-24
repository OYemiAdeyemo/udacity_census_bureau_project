import pandas as pd
import joblib
from model.data import process_data
from model.train_model import inference


def run_inference(input_csv):
    # Load new data to predict on
    data = pd.read_csv(input_csv)

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

    # Load model and encoders
    model = joblib.load("model/random_forest.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/label_binarizer.joblib")

    # Process input data; no label, so label=None
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict
    preds = inference(model, X)

    # Convert binary predictions to original label format
    pred_labels = lb.inverse_transform(preds)

    return pred_labels


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py <input_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    predictions = run_inference(input_csv)
    print("Predictions:")
    print(predictions)
