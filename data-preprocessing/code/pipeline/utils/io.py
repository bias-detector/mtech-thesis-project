"""File helpers shared by all pipeline stages (one JSON file per article)."""

from __future__ import annotations

import json
import os
from typing import Any


def create_output_folder(folder_path: str) -> None:
    """Create the output folder if it does not already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def make_filename(article_title: str) -> str:
    """Convert article title into a safe .json filename."""
    unsafe = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    name = article_title.strip()
    for char in unsafe:
        name = name.replace(char, "_")
    return name + ".json"


def save_article(article: dict[str, Any], output_folder: str) -> bool:
    """Save one article dict as a pretty-printed JSON file."""
    title = article.get("article_title")
    if not title:
        return False
    file_path = os.path.join(output_folder, make_filename(title))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(article, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True


def article_envelope(
    article: dict[str, Any], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Copy article metadata and attach a (filtered) list of pairs."""
    return {
        "article_title": article.get("article_title"),
        "start_year": article.get("start_year"),
        "end_year": article.get("end_year"),
        "pairs": pairs,
    }


def resolve_article_files(input_folder: str, article: str | None) -> list[str]:
    """List JSON files to process; optional article title or filename."""
    if not article:
        files = [name for name in os.listdir(input_folder) if name.endswith(".json")]
        files.sort()
        return files

    name = os.path.basename(article.strip())
    if not name.endswith(".json"):
        name = make_filename(name)
    if os.path.isfile(os.path.join(input_folder, name)):
        return [name]
    return []
