import os

# Vector db config
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# Backend config
TOKEN_LIMIT = 1024
MAX_CHUNKS = 5
DB_DIRECTORY = "./database"

# Model config
supported_llm_providers = ["huggingface_endpoint", "huggingface_local"]
LLM_PROVIDER = "huggingface_local"
HUGGINFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
MODEL_REPO = "Qwen/Qwen3-1.7B"