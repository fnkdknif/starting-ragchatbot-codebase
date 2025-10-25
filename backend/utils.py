"""
Utility functions for the RAG chatbot system
"""


def format_response(text: str, max_length: int = 500) -> str:
    """
    Format a response text by truncating if needed

    Args:
        text: The text to format
        max_length: Maximum length of the formatted text

    Returns:
        Formatted text string
    """
    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def validate_query(query: str) -> bool:
    """
    Validate a user query

    Args:
        query: The query string to validate

    Returns:
        True if query is valid, False otherwise
    """
    if not query or not query.strip():
        return False

    if len(query) > 1000:
        return False

    return True
