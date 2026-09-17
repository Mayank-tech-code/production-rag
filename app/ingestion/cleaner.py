import re


def normalize_whitespace(text: str) -> str:
    """
    Normalize unnecessary whitespace while preserving paragraphs.
    """

    # Replace tabs and repeated spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()