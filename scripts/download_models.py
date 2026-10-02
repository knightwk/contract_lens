# scripts/download_models.py
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

MODELS = {
    "BAAI/bge-base-zh-v1.5": "bge-base-zh-v1.5",  # embedding ~400 MB
    "BAAI/bge-reranker-base": "bge-reranker-base",  # reranker  ~1.1 GB
}

if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "models")
    for repo_id, local_name in MODELS.items():
        target = os.path.abspath(os.path.join(root, local_name))
        print(f"downloading {repo_id} -> {target}")
        snapshot_download(repo_id=repo_id, local_dir=target)
        print(f"done: {local_name}")
