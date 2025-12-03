import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import joblib
import os
import mlflow

# ---------------- Hugging Face Setup ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# YOUR MODEL REPO
repo_id = "DIVHF/tourism-package-model"
repo_type = "model"

# ---------------- MLflow Setup ----------------
public_url = "http://127.0.0.1:5000" 
mlflow.set_tracking_uri(public_url)
mlflow.set_experiment("AML_MLOps_Tourism")

# ---------------- Load dataset ----------------
DATASET_PATH = "hf://datasets/DIVHF/Tourism-AML-MLOps/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from HF.")


# ---------------- Encode Categorical Variables ----------------
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# ---------------- Train-test Split ----------------
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Model Definition ----------------
model = GradientBoostingClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

# ---------------- MLflow Run ----------------
with mlflow.start_run():

    mlflow.set_tag("run_name", "GradientBoosting_Tourism")

    # Hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Classification threshold
    threshold = 0.45

    # Predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],

        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # Save model
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved: {model_path}")

# ---------------- Upload Model to Hugging Face ----------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print("Repo not found, creating new model repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Repo created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type
)

print("Model uploaded to Hugging Face successfully.")
