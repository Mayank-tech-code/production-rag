from langchain_core.documents import Document


def deduplicate_documents(
    documents: list[Document],
) -> list[Document]:

    seen = set()
    unique_documents = []

    for document in documents:

        content = document.page_content.strip()

        if content not in seen:
            seen.add(content)
            unique_documents.append(document)

    return unique_documents

def select_context(
    documents: list[Document],
    max_documents: int = 5,
) -> list[Document]:

    unique_documents = deduplicate_documents(
        documents
    )

    return unique_documents[:max_documents]