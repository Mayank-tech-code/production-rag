from langchain_core.documents import Document


SYSTEM_PROMPT = """
You are a helpful and reliable RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer cannot be found in the context, say:
   "I don't have enough information in the provided documents."
4. Keep the answer clear and concise.
5. When possible, mention the source of the information.
"""


def build_prompt(
    query: str,
    documents: list[Document],
) -> str:

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
--- Context {index} ---
Source: {source}
Page: {page}

{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    return prompt