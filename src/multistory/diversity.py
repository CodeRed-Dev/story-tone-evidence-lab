"""Diversity metrics from the multilingual paper and prose adaptations."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence

from .text import jaccard_distance, mean, ngrams, shannon_entropy, tokenize


def categorical_entropy(labels: Iterable[str]) -> float:
    return shannon_entropy(labels)


def maximum_entropy(sample_count: int, category_count: int) -> float:
    """Exact maximum entropy for k samples assigned to m categories."""

    if sample_count < 0:
        raise ValueError("sample_count must not be negative")
    if category_count < 1:
        raise ValueError("category_count must be positive")
    if sample_count <= 1 or category_count == 1:
        return 0.0
    quotient, remainder = divmod(sample_count, category_count)
    high = (quotient + 1) / sample_count
    low = quotient / sample_count
    high_term = -remainder * high * math.log(high) if remainder and high else 0.0
    low_groups = category_count - remainder
    low_term = -low_groups * low * math.log(low) if low_groups and low else 0.0
    return high_term + low_term


def normalized_entropy(labels: Sequence[str], category_count: int) -> float:
    if not labels:
        return 0.0
    upper = maximum_entropy(len(labels), category_count)
    return categorical_entropy(labels) / upper if upper else 0.0


def lexical_metrics(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    bigrams = ngrams(tokens, 2)
    token_count = len(tokens)
    return {
        "token_count": float(token_count),
        "type_token_ratio": len(set(tokens)) / token_count if token_count else 0.0,
        "distinct_1": len(set(tokens)) / token_count if token_count else 0.0,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "token_entropy": shannon_entropy(tokens),
    }


def _mean_pairwise_distance(sets: Sequence[set]) -> float:
    return mean(jaccard_distance(left, right) for left, right in combinations(sets, 2))


def corpus_diversity(texts: Sequence[str]) -> Dict[str, Any]:
    token_lists: List[List[str]] = [tokenize(text) for text in texts]
    story_metrics = [lexical_metrics(text) for text in texts]
    all_tokens = [token for story_tokens in token_lists for token in story_tokens]
    all_bigrams = [gram for story_tokens in token_lists for gram in ngrams(story_tokens, 2)]
    return {
        "sample_count": len(texts),
        "mean_story_token_count": mean(item["token_count"] for item in story_metrics),
        "mean_story_type_token_ratio": mean(
            item["type_token_ratio"] for item in story_metrics
        ),
        "corpus_distinct_1": len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0,
        "corpus_distinct_2": len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0,
        "corpus_token_entropy": shannon_entropy(all_tokens),
        "pairwise_token_jaccard_distance": _mean_pairwise_distance(
            [set(tokens) for tokens in token_lists]
        ),
        "pairwise_bigram_jaccard_distance": _mean_pairwise_distance(
            [set(ngrams(tokens, 2)) for tokens in token_lists]
        ),
    }
