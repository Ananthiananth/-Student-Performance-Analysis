import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def prepare_training_data(student_data):
    """Prepare student data and performance labels for training."""

    data = student_data.copy()

    required_columns = ["Marks", "Attendance"]

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=["Marks", "Attendance"]
    )

    # These labels follow the rules used by the current dashboard.
    data["Performance"] = data["Marks"].apply(
        lambda marks:
        "Excellent" if marks >= 85
        else "Good" if marks >= 70
        else "Average" if marks >= 50
        else "Needs Improvement"
    )

    return data


def train_model(student_data):
    """Train and evaluate a Random Forest prototype."""

    data = prepare_training_data(student_data)

    if len(data) < 10:
        raise ValueError(
            "At least 10 student records are required "
            "for this prototype."
        )

    features = data[["Marks", "Attendance"]]
    target = data["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    return model, accuracy, report


def predict_performance(model, marks, attendance):
    """Predict the performance category for an input student."""

    input_data = pd.DataFrame({
        "Marks": [marks],
        "Attendance": [attendance]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence