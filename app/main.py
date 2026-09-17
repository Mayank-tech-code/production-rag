from app.config.settings import (
    LLM_PROVIDER,
    QDRANT_URL,
    QDRANT_COLLECTION,
)


def main():
    print("Production RAG Application")
    print("---------------------------")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()
