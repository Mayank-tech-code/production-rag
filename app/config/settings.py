import os

from dotenv import load_dotenv


load_dotenv()


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

HF_TOKEN = os.getenv("HF_TOKEN")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "resume_v1"
)
