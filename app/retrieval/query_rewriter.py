import re


def rewrite_query(query: str) -> str:
    """
    Normalize a user query before retrieval.

    This is the first version of query understanding.
    More advanced rewriting will be added later.
    """

    query = query.strip()

    # Normalize whitespace
    query = re.sub(r"\s+", " ", query)

    # Normalize common question punctuation
    query = query.rstrip("?.!")

    return query