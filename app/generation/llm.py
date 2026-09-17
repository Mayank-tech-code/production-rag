from huggingface_hub import InferenceClient
from app.config.settings import HF_TOKEN


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def get_llm_client():

    return InferenceClient(
        model=MODEL_NAME,
        token=HF_TOKEN,
        provider="featherless-ai",
    )