mport os
import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "fzsiengwe/tourism-dataset"
repo_type = "dataset"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating dataset repo '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

local_csv = hf_hub_download(
    repo_id=repo_id,
    filename="tourism.csv",
    repo_type=repo_type,
)
tourism_df = pd.read_csv(local_csv)
print(f"Dataset loaded successfully. Shape: {tourism_df.shape}")

api.upload_file(
    path_or_fileobj=local_csv,
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Verified tourism.csv in {repo_id}")
