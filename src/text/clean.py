# src/text/clean.py

"""
Simple text cleaner for ClinicalTrials.gov HTML/text fields.

clean_ctgov_text(s, redact_durations=False) -> str
- Unescapes HTML, strips tags, collapses whitespace and excessive newlines,
  replaces bullets and non-breaking spaces.
- If redact_durations=True, heuristically replaces duration phrases
  (e.g. "12 months", "up to 6 years", "12-24 months") with "<DURATION>".

Handles None/empty inputs by returning an empty string.
"""

from __future__ import annotations
import html, re
from typing import Optional

_TIME_RGX = re.compile(
    r"""\b(
        (?:\d+(?:\.\d+)?)\s*(?:year|month|week|day)s?   # 12 months, 96 weeks
        |(?:\d+\s*-\s*\d+)\s*(?:year|month|week|day)s?  # 12-24 months
        |(?:up\s+to|at\s+least|approximately)\s+\d+(?:\.\d+)?\s*(?:year|month|week|day)s?
    )\b""",
    re.I | re.X,
)

_TAGS = re.compile(r"<[^>]+>")      # drop HTML tags
_WS   = re.compile(r"[ \t\r\f\v]+") # normalise spaces
_NL3  = re.compile(r"\n{3,}")       # squash >2 newlines

def clean_ctgov_text(s: Optional[str], redact_durations: bool = False) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = html.unescape(s)
    s = _TAGS.sub(" ", s)      # remove tags
    s = s.replace("\u2022", "*")  # • → bullet
    s = s.replace("\xa0", " ")
    s = _WS.sub(" ", s).strip()
    s = _NL3.sub("\n\n", s)
    if redact_durations:
        s = _TIME_RGX.sub(" <DURATION> ", s)
        s = _WS.sub(" ", s).strip()
    return s
