"""Feature extraction helpers for phishing URL detection."""

from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import tldextract

FEATURE_NAMES = [
    "url_length",
    "dot_count",
    "has_at_symbol",
    "has_hyphen",
    "uses_https",
    "digit_count",
    "has_ip_address",
    "suspicious_keyword_count",
    "domain_length",
]

SUSPICIOUS_KEYWORDS = ("login", "verify", "secure", "account", "update", "bank")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _normalize_url(url: object) -> str:
    """Convert unknown URL input types into a safe lowercase string."""
    if url is None:
        return ""
    url_str = str(url).strip()
    if not url_str or url_str.lower() in {"nan", "none"}:
        return ""
    return url_str


def _domain_length(url: str) -> int:
    """Extract registrable domain safely and return its length."""
    try:
        extracted = tldextract.extract(url)
        return len(extracted.domain or "")
    except Exception:
        return 0


def extract_features(url: object) -> List[float]:
    """Extract numeric phishing features from a single URL."""
    normalized = _normalize_url(url)
    lowered = normalized.lower()

    return [
        float(len(normalized)),
        float(normalized.count(".")),
        float("@" in normalized),
        float("-" in normalized),
        float(lowered.startswith("https://")),
        float(sum(char.isdigit() for char in normalized)),
        float(bool(IPV4_PATTERN.search(normalized))),
        float(sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS)),
        float(_domain_length(normalized)),
    ]


def extract_features_batch(urls: Iterable[object], show_progress: bool = False, desc: str = "") -> np.ndarray:
    """Extract features for many URLs and return a 2D NumPy array."""
    if show_progress:
        from tqdm import tqdm

        rows = [extract_features(url) for url in tqdm(urls, desc=desc, leave=False)]
    else:
        rows = [extract_features(url) for url in urls]

    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)