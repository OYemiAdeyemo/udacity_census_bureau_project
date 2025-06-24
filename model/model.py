from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model  # ✅ must return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def evaluate_model_on_slices(
    model,
    data,
    categorical_feature,
    label,
    cat_features,
    encoder,
    lb,
    process_data_fn,
    compute_metrics_fn,
    inference_fn,
):
    """
    Evaluate model performance on slices of data based on unique values
    in a categorical feature.

    Parameters
    ----------
    model : trained ML model
    data : pd.DataFrame
        Dataset containing features and label.
    categorical_feature : str
        The categorical feature to slice data on.
    label : str
        The label column name.
    cat_features : list[str]
        List of categorical features used in process_data.
    encoder : OneHotEncoder
        Fitted encoder.
    lb : LabelBinarizer
        Fitted label binarizer.
    process_data_fn : function
        Function to process data (e.g., your `process_data`).
    compute_metrics_fn : function
        Function to compute metrics (e.g., your `compute_model_metrics`).
    inference_fn : function
        Function to generate model predictions (e.g., your `inference`).

    Returns
    -------
    dict
        Dictionary mapping each category value to its performance metrics.
        Example: { 'Bachelors': {'precision': 0.85, 'recall': 0.8,
        'fbeta': 0.82}, ... }
    """
    results = {}
    for category_value in data[categorical_feature].unique():
        # Filter data for this slice
        slice_df = data[data[categorical_feature] == category_value]

        # Process the slice data using the existing encoder and lb
        X_slice, y_slice, _, _ = process_data_fn(
            slice_df,
            categorical_features=cat_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Predict on the slice
        preds = inference_fn(model, X_slice)

        # Compute metrics on the slice
        precision, recall, fbeta = compute_metrics_fn(y_slice, preds)

        # Store results
        results[category_value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }

    return results
