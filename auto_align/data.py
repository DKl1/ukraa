"""
Module for loading text data from various file formats.

Provides a function to load sentences from text, CSV files.
"""

import csv
from typing import List


def load_text(file_path: str, fmt: str = "txt") -> List[str]:
    """
    Load sentences from a file in the specified format.

    Args:
        file_path (str): Path to the file.
        fmt (str, optional): File format ('txt', 'csv'). Defaults to "txt".

    Returns:
        List[str]: A list of sentences loaded from the file.

    Raises:
        ValueError: If the specified format is not supported.
    """
    sentences = []
    if fmt == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif fmt == "csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            sentences = [row[0].strip() for row in reader if row]
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return sentences
