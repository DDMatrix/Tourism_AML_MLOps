import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import joblib
import os
import mlflow

# ---------------- Hugging Face Setup ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

repo_id = "DIVHF/tourism-package-model"
repo_type = "model"

# ---------------- MLflow Setup ----------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("AML_MLOps_Tourism")

# ---------------- Load CLEANED Prepared Data ----------------
Xtrain_path = "hf://datasets/DIVHF/Tourism-AML-MLOps/Xtrain.csv"
Xtest_path = "hf://datasets/DIVHF/Tourism-AML-MLOps/Xtest.csv"
ytrain_path = "hf://datasets/DIVHF/Tourism-AML-MLOps/ytrain.csv"
ytest_path = "hf://datasets/DIVHF/Tourism-AML-MLOps/ytest.csv"

print("Loading cleaned train/test splits from Hugging Face...")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print("Cleaned data loaded successfully.")

# ---------------- Model Definition ----------------
model = GradientBoostingClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

# ---------------- MLflow Run ----------------
with mlflow.start_run():

    mlflow.set_tag("run_name", "GradientBoosting_Tourism_CleanedData")

    # Hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Custom threshold
    threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= threshold).astype(int)

    # Metrics
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

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

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved: {model_path}")

# ---------------- Upload to Hugging Face ----------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print("Repo not found, creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type
)

print("Model uploaded to Hugging Face successfully.")
