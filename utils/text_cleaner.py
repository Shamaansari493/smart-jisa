# utils/text_cleaner.py
import re

def clean_text(text: str) -> str:
    """Lowercase, normalize whitespace, remove punctuation (simple but enough)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text
