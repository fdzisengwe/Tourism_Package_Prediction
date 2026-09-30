from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

space_repo_id = "fzisengwe/Tourism-Project-Prediction"

try:
    api.repo_info(repo_id=space_repo_id, repo_type="space")
    print(f"Space '{space_repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating Space '{space_repo_id}' (docker SDK)...")
    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="docker",   # ← REQUIRED
        private=False,
    )
    print(f"Space '{space_repo_id}' created.")

# Upload the deployment folder (Dockerfile, app.py, requirements.txt)
try:
    api.upload_folder(
        folder_path="tourism_project/deployment",
        repo_id=space_repo_id,
        repo_type="space",
        path_in_repo="",
    )
    print(f"Deployment files uploaded to Space '{space_repo_id}'.")
except Exception as e:
    print(f"Failed to upload deployment folder: {e}")
    raise
