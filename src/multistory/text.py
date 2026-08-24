"""Small deterministic text and numerical utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T")


def tokenize(text: str) -> List[str]:
    """Return lowercase Unicode word tokens, retaining internal apostrophes."""

    return [
        token.casefold()
        for token in re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text, flags=re.UNICODE)
    ]


def split_sentences(text: str) -> List[str]:
    """Conservatively split prose into non-empty sentences."""

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", normalized)
        if sentence.strip()
    ]


def ngrams(tokens: Sequence[str], size: int) -> List[Tuple[str, ...]]:
    if size < 1:
        raise ValueError("ngram size must be positive")
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def shannon_entropy(values: Iterable[T]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )
    return max(0.0, value)


def jaccard_distance(left: Iterable[T], right: Iterable[T]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def population_std(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    center = mean(materialized)
    return math.sqrt(mean((value - center) ** 2 for value in materialized))


def interpolate(values: Sequence[float], bins: int = 20) -> List[float]:
    """Linearly resample a one-dimensional sequence to exactly ``bins`` points."""

    if bins < 1:
        raise ValueError("bins must be positive")
    if not values:
        return [0.0] * bins
    if len(values) == 1:
        return [float(values[0])] * bins
    if bins == 1:
        return [float(values[0])]
    result: List[float] = []
    last_source = len(values) - 1
    for index in range(bins):
        position = index * last_source / (bins - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            result.append(float(values[low]))
            continue
        fraction = position - low
        result.append(float(values[low]) * (1 - fraction) + float(values[high]) * fraction)
    return result


def _kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        p * math.log(p / q)
        for p, q in zip(left, right)
        if p > 0 and q > 0
    )


def jensen_shannon_distance(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    """Return Jensen-Shannon distance with natural logarithms (range 0..sqrt(log 2))."""

    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    left_total, right_total = sum(left.values()), sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("Jensen-Shannon inputs must each have positive mass")
    p = [float(left.get(key, 0.0)) / left_total for key in keys]
    q = [float(right.get(key, 0.0)) / right_total for key in keys]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]
    divergence = (_kl_divergence(p, midpoint) + _kl_divergence(q, midpoint)) / 2.0
    return math.sqrt(max(0.0, divergence))


def frequency_distribution(values: Iterable[str], categories: Sequence[str]) -> Dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {category: 0.0 for category in categories}
    return {category: counts.get(category, 0) / total for category in categories}
