from utils.text_cleaner import clean_text

class IngestionAgent:
    def run(self, title: str, description: str) -> str:
        """
        Accepts title + description (2 params),
        concatenates them, and cleans the combined text.
        """
        combined = f"{title or ''}. {description or ''}"
        return clean_text(combined)
