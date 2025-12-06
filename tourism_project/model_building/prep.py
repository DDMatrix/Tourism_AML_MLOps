import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os
import pickle

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "DIVHF/Tourism-AML-MLOps"
repo_type = "dataset"

# ---------------- Load dataset from Hugging Face ----------------
DATASET_PATH = "hf://datasets/DIVHF/Tourism-AML-MLOps/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from HF.")

# ---------------- Data Cleaning ----------------
# Drop only CustomerID
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Fix Gender typo and merge rare categories
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df['Occupation'] = df['Occupation'].replace({'Free Lancer': 'Small Business'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Single': 'Unmarried'})

# ---------------- Encode Categorical Variables ----------------
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le   # <-- Save encoder for each column

# Save all encoders to a pickle file
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# ---------------- Train-test Split ----------------
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Save CSVs ----------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ---------------- Upload datasets to HF ----------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type=repo_type
    )

print("All dataset files uploaded to Hugging Face.")

# ---------------- Upload Label Encoders ----------------
api.upload_file(
    path_or_fileobj="label_encoders.pkl",
    path_in_repo="label_encoders.pkl",
    repo_id=repo_id,
    repo_type=repo_type
)

print("label_encoders.pkl uploaded to Hugging Face.")
