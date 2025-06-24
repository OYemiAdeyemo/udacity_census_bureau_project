# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether an individual's income exceeds $50K per year based on census data features such as education, age, occupation, and marital status. The model uses one-hot encoding for categorical variables and is implemented in Python with scikit-learn.

## Intended Use
The model is intended for educational purposes and demonstration of machine learning techniques on census income prediction. It can assist in understanding patterns in demographic and economic data but should not be used for real-world decision making affecting individuals without further validation and ethical review.

## Training Data
The training data consists of the U.S. Census Income dataset (also known as the Adult dataset) from the UCI Machine Learning Repository. It contains approximately 32,000 records with features including age, workclass, education, marital status, occupation, race, sex, and native country.

## Evaluation Data
Model performance was evaluated on a held-out test set containing 20% of the original dataset, ensuring unbiased evaluation. Additional evaluation was conducted by slicing the test data along categorical features (e.g., education) to assess model fairness and consistency across subgroups.

## Metrics
The model is evaluated using the following metrics:
- Precision: Measures the proportion of positive identifications that were actually correct.
- Recall: Measures the proportion of actual positives that were identified correctly.
- F1 Score (F-beta with β=1): Harmonic mean of precision and recall.

Example overall performance on test data:

- Precision: 0.85
- Recall: 0.78
- F1 Score: 0.81

Performance varies across slices of data; for example, precision on individuals with “Bachelors” education is 0.87, while on “HS-grad” is 0.80.
## Ethical Considerations
The model is trained on demographic data and may reflect societal biases present in the data, such as disparities related to race, gender, or education. It is important to interpret predictions with caution and consider fairness implications before deploying in real-world applications. The model should not be used for decisions that could adversely affect individuals without thorough bias and fairness audits.

## Caveats and Recommendations
- The model performance depends on the quality and representativeness of the training data.
- Slicing evaluation reveals variability in performance across subgroups, indicating potential biases.
- Users should conduct additional validation on their target population before deployment.
- Consider incorporating fairness-enhancing techniques if used in sensitive contexts.
- The model is not designed to handle data distribution shifts or missing data robustly.

