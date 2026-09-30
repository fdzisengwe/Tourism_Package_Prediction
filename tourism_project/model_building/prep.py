#Use the project folder name, then the folder where model training code will be stored.
# import necessary libraries
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for Hugging Face authentication to upload files
from huggingface_hub import login, HfApi


# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))  # complete the code to read HF_TOKEN from environment variable
DATASET_PATH = "hf://datasets/fzisengwe/tourism-dataset/tourism.csv"
tourism_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = "ProdTaken"  # complete the code to set the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0

# ----------------------------
# List of numerical features
# ----------------------------
numeric_features = ['Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'] # complete the code to show list of all numerical feature

# ----------------------------
# List of categorical features
# ----------------------------
categorical_features = ['TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched'] # complete the code to show list of all categorical feature

# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------

X = tourism_df[numeric_features + categorical_features]

# ----------------------------
# Define target vector y
# ----------------------------

y = tourism_df[target]   # complete the code to select the target column from tourism_df

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
 X, y,
    test_size=0.2,
    random_state=42
)

# Save splits to CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload each split file to the Hugging Face dataset repository
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="fzisengwe/tourism-dataset",
        repo_type="dataset",   # ← was "space"
    )
