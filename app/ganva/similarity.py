"""
Algorithm S3-2: normalised Levenshtein similarity, plus the D5 multi-word rule.
"""
from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from .normalize import normalize_arabic_text


def levenshtein_distance(a: str, b: str) -> int:
    """Unit-cost insert/delete/substitute edit distance."""
    return Levenshtein.distance(a, b)


def _similarity_normalized(t1: str, t2: str) -> float:
    if t1 == t2:
        return 1.0
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 1.0
    return (max_len - levenshtein_distance(t1, t2)) / max_len


def calculate_similarity(text1: str, text2: str) -> float:
    """S3-2 exactly: normalise both strings, return (maxLen - lev) / maxLen."""
    return _similarity_normalized(normalize_arabic_text(text1), normalize_arabic_text(text2))


@dataclass(frozen=True)
class SpanMatch:
    similarity: float
    span: str  # the normalised part of the transcript that matched best


def best_span_similarity(transcript: str, reference: str, multiword_rule: str = "best_ngram") -> SpanMatch:
    """
    D5: responses often carry extra words ("هذا مهرج", "مهرج مهرج"). Compare the
    reference with the whole transcript and with every contiguous n-gram of up to
    (reference tokens + 1) tokens, and keep the best score.
    """
    t = normalize_arabic_text(transcript)
    r = normalize_arabic_text(reference)
    best = SpanMatch(_similarity_normalized(t, r), t)
    if multiword_rule != "best_ngram" or not t:
        return best
    tokens = t.split(" ")
    max_n = min(len(tokens), len(r.split(" ")) + 1)
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            span = " ".join(tokens[i : i + n])
            sim = _similarity_normalized(span, r)
            if sim > best.similarity:
                best = SpanMatch(sim, span)
    return best
