#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
french_text_metrics_full_csv.py
-----------------------------------
Compute detailed French text metrics (structural, lexical, and readability)
for a chosen text column in a CSV dataset.

Includes:
- Length metrics (chars, words, avg sentence length)
- Lexicon metrics (unique words, TTR)
- Abbreviation & section counts
- Negations, numbers, temporal refs
- Flesch–Kincaid readability (French adaptation)
"""

import regex as re
import unicodedata
from typing import List, Dict
import pandas as pd
from pyphen import Pyphen

# --------------------------- Regex definitions ---------------------------

_RE_WORD = re.compile(r"\b[\p{L}\p{N}][\p{L}\p{N}\-_/]*\b", re.UNICODE)
_RE_UPPER_ABBR = re.compile(r"\b[A-ZÀ-Ý0-9]{2,}\b")
_RE_ABBR_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b")

# --------------------------- Core text functions ---------------------------

def _strip_accents(s: str) -> str:
    """Remove accents from text."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into words (Unicode safe)."""
    return _RE_WORD.findall(text)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences by strong punctuation and newlines."""
    return [s.strip() for s in re.split(r"[\.!?;\n]+", text) if s.strip()]


def _count_sections_by_blanklines(text: str) -> int:
    """Count sections (blocks separated by ≥2 newlines)."""
    blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()]
    return len(blocks)


def _count_abbr(text: str) -> int:
    """Count uppercase or dotted abbreviations."""
    return len(_RE_UPPER_ABBR.findall(text)) + len(_RE_ABBR_DOTTED.findall(text))


# --------------------------- Additional lexical & readability metrics ---------------------------

def _ttr(words: List[str]) -> float:
    """Type-Token Ratio → lexical diversity."""
    return round(len(set(words)) / len(words), 4) if words else 0.0


def _detect_negations(text: str) -> int:
    """Count simple French negation terms."""
    neg_words = ["pas", "aucun", "sans", "ni", "jamais"]
    return sum(len(re.findall(rf"\b{w}\b", text, re.IGNORECASE)) for w in neg_words)


def _count_numbers(text: str) -> int:
    """Count numeric expressions."""
    return len(re.findall(r"\d+", text))


def _count_temporal_refs(text: str) -> int:
    """Count temporal references (dates, years, months)."""
    months = [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre"
    ]
    patterns = [r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", r"\b\d{4}\b"] + months
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def _flesch_kincaid_fr(text: str, words: List[str], sentences: List[str]) -> float:
    """Flesch–Kincaid readability score adapted for French."""
    dic = Pyphen(lang="fr")

    if not words or not sentences:
        return 0.0

    syllables = sum(len(dic.inserted(w).split("-")) for w in words)
    ASL = len(words) / len(sentences)      # average sentence length
    ASW = syllables / len(words)           # average syllables per word
    score = 206.835 - 1.015 * ASL - 84.6 * ASW
    return round(score, 2)


# --------------------------- Compute metrics (core logic) ---------------------------

def _normalize_for_vocab(words: List[str], lowercase=True, remove_accents=True) -> List[str]:
    """Normalize words for lexical diversity counting."""
    out = words
    if lowercase:
        out = [t.lower() for t in out]
    if remove_accents:
        out = [_strip_accents(t) for t in out]
    return out


def compute_text_metrics(text: str) -> Dict[str, float]:
    """Compute all text metrics for one text sample."""
    if not isinstance(text, str) or not text.strip():
        return {
            "len_chars": 0,
            "len_words": 0,
            "sent_len_avg": 0.0,
            "lexicon_size": 0,
            "n_sections": 0,
            "n_abbr": 0,
            "ttr": 0.0,
            "n_negations": 0,
            "n_numbers": 0,
            "n_temporal_refs": 0,
            "flesch_fr": 0.0,
        }

    words = _tokenize_words(text)
    sents = _split_sentences(text)
    norm_words = _normalize_for_vocab(words)

    len_chars = len(text)
    len_words = len(words)
    sent_len = sum(len(_tokenize_words(s)) for s in sents)
    lexicon_size = len(set(norm_words))
    n_sections = _count_sections_by_blanklines(text)
    n_abbr = _count_abbr(text)
    ttr = _ttr(norm_words)
    n_negations = _detect_negations(text)
    n_numbers = _count_numbers(text)
    n_temporal_refs = _count_temporal_refs(text)
    flesch_fr = _flesch_kincaid_fr(text, words, sents)

    return {
        "len_chars": int(len_chars),
        "len_words": int(len_words),
        "sent_len_avg": float(round(sent_len_avg, 3)),
        "lexicon_size": int(lexicon_size),
        "n_sections": int(n_sections),
        "n_abbr": int(n_abbr),
        "ttr": float(ttr),
        "n_negations": int(n_negations),
        "n_numbers": int(n_numbers),
        "n_temporal_refs": int(n_temporal_refs),
        "flesch_fr": float(flesch_fr),
    }


# --------------------------- Main CLI script ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute detailed French text metrics for a given CSV column.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--column", required=True, help="Text column to analyze (e.g., observationBlob or txt_rw).")
    parser.add_argument("--output", default="metrics_output_mean.csv", help="Output CSV file path.")
    args = parser.parse_args()

    # Load dataset (CSV only)
    df = pd.read_csv(args.input, sep=";")

    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in dataset. Available: {list(df.columns)}")


    metrics = df[args.column].apply(compute_text_metrics)
    metrics_df = pd.DataFrame(metrics.tolist())
    
    summary = metrics_df.mean(numeric_only=True).round(3)
    for metric, value in summary.items():
        print(f" {metric}: {value}")

    
