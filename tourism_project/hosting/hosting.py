from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     #complete code for local folder containing your files
    repo_id = "fzsiengwe/Tourism-Project-Prediction",    # Replace <-------Hugging Face user ID --------->/<-------> with your Hugging Face username and model repository name.
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
