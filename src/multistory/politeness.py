"""Inspectable English/Hindi surface politeness-strategy measurements.

Politeness is culturally situated and cannot be reduced to a word list.  This
module detects explicit strategies only, so reports must retain the accompanying
validity warning instead of treating the score as a universal politeness label.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Mapping, Sequence, Tuple


POSITIVE_STRATEGIES: Mapping[str, Mapping[str, Tuple[str, ...]]] = {
    "en": {
        "request_softener": (
            "please",
            "kindly",
            "could you",
            "would you",
            "would it be possible",
            "if you don't mind",
        ),
        "gratitude": ("thank you", "thanks", "appreciate", "grateful"),
        "apology": ("sorry", "apologize", "apologies", "excuse me", "pardon"),
        "respect_or_honorific": ("sir", "madam", "ma'am", "mr.", "mrs.", "ms."),
        "hedging_or_indirectness": (
            "perhaps",
            "maybe",
            "possibly",
            "if possible",
            "i think",
            "it may",
            "it might",
        ),
        "warmth_or_support": (
            "happy to",
            "glad to",
            "hope this helps",
            "i understand",
            "certainly",
            "of course",
        ),
    },
    "hi": {
        "request_softener": ("कृपया", "क्या आप", "ज़रा", "जरा", "अगर संभव हो", "यदि संभव हो"),
        "gratitude": ("धन्यवाद", "शुक्रिया", "आभारी", "सराहना"),
        "apology": ("माफ़", "माफ", "क्षमा", "खेद"),
        "respect_or_honorific": (
            "आप",
            "आपका",
            "आपकी",
            "आपके",
            "आपसे",
            "आपको",
            "जी",
            "महोदय",
            "महोदया",
        ),
        "hedging_or_indirectness": (
            "शायद",
            "संभवतः",
            "यदि",
            "अगर",
            "मुझे लगता है",
            "हो सकता है",
        ),
        "warmth_or_support": (
            "खुशी",
            "आशा",
            "उम्मीद",
            "ज़रूर",
            "जरूर",
            "अवश्य",
            "निश्चित रूप से",
            "मैं समझता हूँ",
            "मैं समझती हूँ",
        ),
    },
}

INFORMAL_OR_BLUNT: Mapping[str, Tuple[str, ...]] = {
    "en": ("must", "obviously", "nonsense", "do it", "you're wrong", "you are wrong", "shut up"),
    "hi": ("तू", "तुम", "तेरा", "तेरी", "तेरे", "करो", "बकवास", "बेवकूफ", "चुप रहो"),
}

VALIDITY_WARNING = (
    "This detector measures explicit lexical strategies only. It does not measure prosody, context, "
    "sincerity, regional norms, or total politeness, and English/Hindi scores are not assumed to be "
    "perfectly culturally equivalent."
)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _marker_pattern(marker: str, language: str) -> str:
    escaped = re.escape(_normalize(marker)).replace(r"\ ", r"\s+")
    if language == "hi":
        return r"(?<![\u0900-\u097f])" + escaped + r"(?![\u0900-\u097f])"
    return r"(?<!\w)" + escaped + r"(?!\w)"


def _count_marker(text: str, marker: str, language: str) -> int:
    return len(re.findall(_marker_pattern(marker, language), text, flags=re.IGNORECASE))


def _score_marker_group(
    normalized: str, markers: Sequence[str], language: str
) -> Tuple[int, Dict[str, int]]:
    matches: Dict[str, int] = {}
    for marker in markers:
        count = _count_marker(normalized, marker, language)
        if count:
            matches[marker] = count
    return sum(matches.values()), matches


def script_ratios(text: str) -> Dict[str, float]:
    normalized = unicodedata.normalize("NFKC", str(text))
    devanagari = len(re.findall(r"[\u0900-\u097f]", normalized))
    latin = len(re.findall(r"[A-Za-z]", normalized))
    denominator = devanagari + latin
    if denominator == 0:
        return {"devanagari": 0.0, "latin": 0.0}
    return {"devanagari": devanagari / denominator, "latin": latin / denominator}


def score_politeness(text: str, language: str) -> Dict[str, Any]:
    """Score explicit response strategies for ``en`` or ``hi``.

    Coverage is the fraction of the six positive strategy families observed at
    least once.  It is a measurement convenience, not a universal politeness
    probability.
    """

    if language not in POSITIVE_STRATEGIES:
        raise ValueError("language must be 'en' or 'hi'")
    normalized = _normalize(text)
    counts: Dict[str, int] = {}
    presence: Dict[str, int] = {}
    matches: Dict[str, Dict[str, int]] = {}
    for strategy, markers in POSITIVE_STRATEGIES[language].items():
        count, strategy_matches = _score_marker_group(normalized, markers, language)
        counts[strategy] = count
        presence[strategy] = int(count > 0)
        matches[strategy] = strategy_matches
    informal_count, informal_matches = _score_marker_group(
        normalized, INFORMAL_OR_BLUNT[language], language
    )
    ratios = script_ratios(text)
    target_script_ratio = ratios["devanagari"] if language == "hi" else ratios["latin"]
    deference_names = ("request_softener", "respect_or_honorific", "hedging_or_indirectness")
    return {
        "language": language,
        "word_count": len(re.findall(r"\S+", normalized)),
        "strategy_counts": counts,
        "strategy_presence": presence,
        "positive_marker_count": sum(counts.values()),
        "politeness_strategy_coverage": sum(presence.values()) / len(presence),
        "deference_strategy_coverage": sum(presence[name] for name in deference_names)
        / len(deference_names),
        "informal_or_blunt_count": informal_count,
        "informal_or_blunt_presence": int(informal_count > 0),
        "script_ratios": ratios,
        "target_script_ratio": target_script_ratio,
        "matched_markers": {**matches, "informal_or_blunt": informal_matches},
        "validity_warning": VALIDITY_WARNING,
    }
