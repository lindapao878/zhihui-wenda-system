"""Markdown text preprocessing helpers."""
from __future__ import annotations

import re


class MarkdownTableLinearizer:
    """Best-effort conversion of HTML tables into readable text."""

    @staticmethod
    def process(text: str) -> str:
        if not text or "<table" not in text.lower():
            return text

        text = re.sub(r"(?is)<thead>.*?</thead>", "", text)
        text = re.sub(r"(?is)<tbody>|</tbody>", "", text)
        text = re.sub(r"(?is)<tr[^>]*>", "\n", text)
        text = re.sub(r"(?is)</tr>", "", text)
        text = re.sub(r"(?is)<t[dh][^>]*>", " | ", text)
        text = re.sub(r"(?is)</t[dh]>", "", text)
        text = re.sub(r"(?is)</?table[^>]*>", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
