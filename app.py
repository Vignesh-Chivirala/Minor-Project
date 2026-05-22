from flask import Flask, render_template, request, redirect, url_for

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "saved_models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_model.pkl")
METADATA_PATH = os.path.join(MODEL_FOLDER, "model_metadata.pkl")
HISTORY_PATH = os.path.join(MODEL_FOLDER, "prediction_history.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global variables
dataset_path = ""
feature_columns = []
target_column_name = ""
model_accuracy = ""
model_algorithm = ""
feature_ranges = {}


def save_model_metadata(features, target, algorithm, accuracy, ranges=None):

    metadata = {
        "features": features,
        "target": target,
        "algorithm": algorithm,
        "accuracy": accuracy,
        "feature_ranges": ranges or {}
    }

    joblib.dump(metadata, METADATA_PATH)


def get_feature_ranges_from_dataset(features, target):

    required_columns = set(features + [target])

    for filename in os.listdir(UPLOAD_FOLDER):

        if not filename.lower().endswith(".csv"):

            continue

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        df = pd.read_csv(file_path)

        if not required_columns.issubset(df.columns):

            continue

        return {
            feature: {
                "min": float(df[feature].min()),
                "max": float(df[feature].max())
            }
            for feature in features
        }

    return {}


def load_model_metadata():

    if not os.path.exists(MODEL_PATH):
        return None

    if not os.path.exists(METADATA_PATH):

        model = joblib.load(MODEL_PATH)

        if not hasattr(model, "feature_names_in_"):
            return None

        metadata = {
            "features": list(model.feature_names_in_),
            "target": "Prediction",
            "algorithm": type(model).__name__,
            "accuracy": "Saved model",
            "feature_ranges": {}
        }

        save_model_metadata(
            metadata["features"],
            metadata["target"],
            metadata["algorithm"],
            metadata["accuracy"],
            metadata["feature_ranges"]
        )

        return metadata

    metadata = joblib.load(METADATA_PATH)

    if "feature_ranges" not in metadata:

        metadata["feature_ranges"] = get_feature_ranges_from_dataset(
            metadata["features"],
            metadata["target"]
        )

        save_model_metadata(
            metadata["features"],
            metadata["target"],
            metadata["algorithm"],
            metadata["accuracy"],
            metadata["feature_ranges"]
        )

    return metadata


def apply_model_metadata(metadata):

    global feature_columns
    global target_column_name
    global model_accuracy
    global model_algorithm
    global feature_ranges

    feature_columns = metadata["features"]
    target_column_name = metadata["target"]
    model_algorithm = metadata["algorithm"]
    model_accuracy = metadata["accuracy"]
    feature_ranges = metadata.get("feature_ranges", {})


def load_prediction_history():

    if not os.path.exists(HISTORY_PATH):
        return []

    try:

        return joblib.load(HISTORY_PATH)

    except (EOFError, ValueError, OSError):

        save_prediction_history([])

        return []


def save_prediction_history(history):

    joblib.dump(history, HISTORY_PATH)


def add_prediction_history(input_values, prediction, target):

    history = load_prediction_history()

    history.insert(
        0,
        {
            "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "target": target,
            "prediction": prediction,
            "inputs": input_values
        }
    )

    save_prediction_history(history)


def create_model_chart(y_test, y_pred):

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    padding = (
        (max_value - min_value) * 0.08
        if max_value != min_value
        else 1
    )

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, axis = plt.subplots(figsize=(9, 5.2))

    fig.patch.set_facecolor("#ffffff")

    axis.scatter(
        y_test,
        y_pred,
        s=64,
        color="#2563eb",
        alpha=0.78,
        edgecolor="#ffffff",
        linewidth=0.8
    )

    axis.plot(
        [min_value - padding, max_value + padding],
        [min_value - padding, max_value + padding],
        color="#0f766e",
        linewidth=2.6,
        label="Ideal prediction"
    )

    axis.set_title(
        "Actual vs Predicted",
        fontsize=16,
        fontweight="bold",
        color="#111827",
        pad=16
    )

    axis.set_xlabel("Actual value", fontsize=11)
    axis.set_ylabel("Predicted value", fontsize=11)

    axis.legend(frameon=False, loc="upper left")

    axis.set_facecolor("#ffffff")
    axis.grid(color="#e2e8f0", linewidth=0.8)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.spines["left"].set_color("#cbd5e1")
    axis.spines["bottom"].set_color("#cbd5e1")

    axis.tick_params(colors="#475569", labelsize=9)

    fig.tight_layout()

    os.makedirs("static", exist_ok=True)

    fig.savefig(
        "static/graph.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html",
        has_saved_model=load_model_metadata() is not None
    )


# =========================
# UPLOAD DATASET
# =========================

@app.route("/upload", methods=["POST"])
def upload():

    global dataset_path

    file = request.files["dataset"]

    dataset_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(dataset_path)

    # Read dataset
    df = pd.read_csv(dataset_path)

    columns = df.columns.tolist()

    return render_template(
        "select_target.html",
        columns=columns,
        has_saved_model=load_model_metadata() is not None
    )


# =========================
# USE TRAINED MODEL
# =========================

@app.route("/trained-model")
def trained_model():

    metadata = load_model_metadata()

    if metadata is None:
        return redirect(url_for("home"))

    apply_model_metadata(metadata)

    return render_template(
        "predict.html",
        features=feature_columns,
        feature_ranges=feature_ranges,
        model_name=model_algorithm,
        accuracy=model_accuracy,
        target=target_column_name,
        prediction_history=load_prediction_history()
    )


# =========================
# TRAIN MODEL
# =========================

@app.route("/train", methods=["POST"])
def train():

    global feature_columns
    global target_column_name
    global model_accuracy
    global model_algorithm
    global feature_ranges

    target_column = request.form["target"]

    target_column_name = target_column

    # Read dataset
    df = pd.read_csv(dataset_path)

    # Features & Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_columns = X.columns.tolist()
    feature_ranges = {
        feature: {
            "min": float(X[feature].min()),
            "max": float(X[feature].max())
        }
        for feature in feature_columns
    }

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Models
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    best_model = None
    best_score = -1
    best_name = ""

    # Train Models
    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)

        if score > best_score:

            best_score = score
            best_model = model
            best_name = name

    # Save Best Model
    joblib.dump(best_model, MODEL_PATH)

    model_accuracy = round(best_score, 4)
    model_algorithm = best_name

    save_model_metadata(
        feature_columns,
        target_column_name,
        model_algorithm,
        model_accuracy,
        feature_ranges
    )

    # Create Graph
    best_predictions = best_model.predict(X_test)

    create_model_chart(
        y_test,
        best_predictions
    )

    return render_template(
        "predict.html",
        features=feature_columns,
        feature_ranges=feature_ranges,
        model_name=model_algorithm,
        accuracy=model_accuracy,
        target=target_column_name,
        prediction_history=load_prediction_history()
    )


# =========================
# FINAL PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if not feature_columns:

        metadata = load_model_metadata()

        if metadata is None:
            return redirect(url_for("home"))

        apply_model_metadata(metadata)

    model = joblib.load(MODEL_PATH)

    values = []
    input_values = []

    for feature in feature_columns:

        value = float(request.form[feature])
        range_info = feature_ranges.get(feature)

        if range_info:

            min_value = range_info["min"]
            max_value = range_info["max"]

            if value < min_value or value > max_value:

                return render_template(
                    "invalid_input.html",
                    feature=feature,
                    value=value,
                    min_value=min_value,
                    max_value=max_value
                )

        values.append(value)

        input_values.append(
            {
                "feature": feature,
                "value": value
            }
        )

    user_data = pd.DataFrame(
        [values],
        columns=feature_columns
    )

    prediction = model.predict(user_data)

    prediction_value = float(prediction[0])

    add_prediction_history(
        input_values,
        prediction_value,
        target_column_name
    )

    return render_template(
        "result.html",
        prediction=prediction_value,
        target=target_column_name,
        input_values=input_values
    )


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)
