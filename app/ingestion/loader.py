from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    loader = PyMuPDFLoader(str(path))

    documents = loader.load()

    return documents