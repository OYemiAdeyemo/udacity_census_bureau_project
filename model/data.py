import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """
    Process the data used in the machine learning pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features : list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in X. If None, y will be an empty array.
    training : bool
        True if training mode, False for inference/validation.
    encoder : OneHotEncoder, optional
        Trained encoder to apply to X.
    lb : LabelBinarizer, optional
        Trained label binarizer to apply to y.

    Returns
    -------
    X_processed : np.ndarray
        Processed feature data.
    y : np.ndarray
        Processed label data (empty if label=None).
    encoder : OneHotEncoder
        Trained encoder (if training) or passed encoder.
    lb : LabelBinarizer
        Trained label binarizer (if training) or passed lb.
    """
    if label is not None:
        y = X[label]
        X = X.drop(columns=[label])
    else:
        y = np.array([])

    if categorical_features:
        X_categorical = X[categorical_features].values
        X_continuous = X.drop(columns=categorical_features).values
    else:
        X_categorical = np.empty((len(X), 0))
        X_continuous = X.values

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            print(
                """ Warning: LabelBinarizer not provided or invalid; "
                y will remain unchanged."""
            )
        except Exception as e:
            print(f"Unexpected error during label transform: {e}")

    X_processed = np.concatenate([X_continuous, X_categorical], axis=1)
    return X_processed, y, encoder, lb
