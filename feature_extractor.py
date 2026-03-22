"""Feature extraction helpers for phishing URL detection."""

from __future__ import annotations

import math
import re
from urllib.parse import SplitResult, urlsplit
from typing import Iterable, List

import numpy as np
import tldextract

FEATURE_NAMES = [
    "url_length",
    "dot_count",
    "slash_count",
    "question_mark_count",
    "equal_count",
    "ampersand_count",
    "percent_count",
    "underscore_count",
    "has_at_symbol",
    "has_hyphen",
    "uses_https",
    "digit_count",
    "digit_ratio",
    "has_ip_address",
    "suspicious_keyword_count",
    "subdomain_count",
    "domain_length",
    "tld_length",
    "path_length",
    "query_length",
    "fragment_length",
    "is_shortener",
    "has_suspicious_tld",
    "url_entropy",
]

SUSPICIOUS_KEYWORDS = ("login", "verify", "secure", "account", "update", "bank")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SHORTENER_DOMAINS = {
    "bit",
    "tinyurl",
    "goo",
    "t",
    "ow",
    "is",
    "adf",
    "shorte",
}
SUSPICIOUS_TLDS = {
    "tk",
    "xyz",
    "top",
    "gq",
    "ml",
    "cf",
    "work",
    "click",
    "link",
}


def _normalize_url(url: object) -> str:
    """Convert unknown URL input types into a safe lowercase string."""
    if url is None:
        return ""
    url_str = str(url).strip()
    if not url_str or url_str.lower() in {"nan", "none"}:
        return ""
    return url_str


def extract_registered_domain(url: object) -> str:
    """Extract a stable registered-domain identifier for grouping and deduping."""
    normalized = _normalize_url(url)
    if not normalized:
        return ""
    try:
        extracted = tldextract.extract(normalized)
        domain = extracted.domain or ""
        suffix = extracted.suffix or ""
        if domain and suffix:
            return f"{domain}.{suffix}".lower()
        return domain.lower()
    except Exception:
        return ""


def _domain_features(url: str) -> tuple[int, int, int, int, int]:
    try:
        extracted = tldextract.extract(url)
        subdomain_parts = [part for part in extracted.subdomain.split(".") if part]
        domain = extracted.domain or ""
        suffix = extracted.suffix or ""
        is_shortener = int(domain.lower() in SHORTENER_DOMAINS)
        has_suspicious_tld = int(suffix.lower() in SUSPICIOUS_TLDS)
        return len(subdomain_parts), len(domain), len(suffix), is_shortener, has_suspicious_tld
    except Exception:
        return 0, 0, 0, 0, 0


def _url_entropy(text: str) -> float:
    if not text:
        return 0.0
    length = len(text)
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _safe_urlsplit(url: str) -> SplitResult:
    try:
        return urlsplit(url if "://" in url else f"http://{url}")
    except ValueError:
        return SplitResult("", "", "", "", "")


def extract_features(url: object) -> List[float]:
    """Extract numeric phishing features from a single URL."""
    normalized = _normalize_url(url)
    lowered = normalized.lower()
    parsed = _safe_urlsplit(normalized)

    subdomain_count, domain_length, tld_length, is_shortener, has_suspicious_tld = _domain_features(normalized)

    url_len = max(len(normalized), 1)
    digit_count = sum(char.isdigit() for char in normalized)

    return [
        float(len(normalized)),
        float(normalized.count(".")),
        float(normalized.count("/")),
        float(normalized.count("?")),
        float(normalized.count("=")),
        float(normalized.count("&")),
        float(normalized.count("%")),
        float(normalized.count("_")),
        float("@" in normalized),
        float("-" in normalized),
        float(lowered.startswith("https://")),
        float(digit_count),
        float(digit_count / url_len),
        float(bool(IPV4_PATTERN.search(normalized))),
        float(sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS)),
        float(subdomain_count),
        float(domain_length),
        float(tld_length),
        float(len(parsed.path)),
        float(len(parsed.query)),
        float(len(parsed.fragment)),
        float(is_shortener),
        float(has_suspicious_tld),
        float(_url_entropy(lowered)),
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