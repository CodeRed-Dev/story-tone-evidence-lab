"""Controlled local experiment for English/Hindi response-tone differences."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .politeness import VALIDITY_WARNING, score_politeness


class TextProvider(Protocol):
    name: str
    model: str

    def generate_text(self, prompt: str, seed: int, max_tokens: int = 300) -> str:
        ...


ANALYSIS_METRICS: Tuple[str, ...] = (
    "politeness_strategy_coverage",
    "deference_strategy_coverage",
    "positive_markers_per_100_words",
    "informal_or_blunt_presence",
    "target_script_ratio",
)


def load_tone_prompts(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("tone prompts file must contain a JSON list")
    prompts: List[Dict[str, str]] = []
    seen = set()
    required = ("id", "task", "language", "input_tone", "prompt")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or any(not isinstance(item.get(key), str) for key in required):
            raise ValueError("tone prompt %d is missing required string fields" % index)
        record = {key: str(item[key]) for key in required}
        if record["id"] in seen:
            raise ValueError("duplicate tone prompt id: %s" % record["id"])
        if record["language"] not in {"en", "hi"}:
            raise ValueError("unsupported tone prompt language: %s" % record["language"])
        if record["input_tone"] not in {"direct", "deferential"}:
            raise ValueError("unsupported input tone: %s" % record["input_tone"])
        seen.add(record["id"])
        prompts.append(record)
    cells = {(item["task"], item["language"], item["input_tone"]) for item in prompts}
    for task in {item["task"] for item in prompts}:
        expected = {(task, language, tone) for language in ("en", "hi") for tone in ("direct", "deferential")}
        if not expected.issubset(cells):
            raise ValueError("task %s does not contain all four language/tone cells" % task)
    return prompts


def _add_rates(score: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(score)
    words = max(1, int(result["word_count"]))
    result["positive_markers_per_100_words"] = 100.0 * float(result["positive_marker_count"]) / words
    return result


def _write_checkpoint(path: Path, raw: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(raw, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def run_tone_generation(
    provider: TextProvider,
    prompts_path: Path,
    raw_path: Path,
    base_seed: int = 260416275,
    max_tokens: int = 260,
    resume: bool = True,
) -> Dict[str, Any]:
    """Generate all factorial cells, checkpointing every exact response."""

    prompts = load_tone_prompts(prompts_path)
    tasks = list(dict.fromkeys(item["task"] for item in prompts))
    prompt_sha256 = hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest()
    prior: Dict[str, Mapping[str, Any]] = {}
    if resume and Path(raw_path).exists():
        with Path(raw_path).open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_metadata = existing.get("metadata", {}) if isinstance(existing, Mapping) else {}
        same_experiment = isinstance(existing_metadata, Mapping) and all(
            existing_metadata.get(key) == expected
            for key, expected in {
                "provider": provider.name,
                "model": provider.model,
                "prompt_sha256": prompt_sha256,
                "base_seed": base_seed,
                "max_tokens": max_tokens,
            }.items()
        )
        if same_experiment and isinstance(existing.get("records"), list):
            prior = {
                str(record.get("id")): record
                for record in existing["records"]
                if isinstance(record, Mapping) and record.get("status") == "ok"
            }

    raw: Dict[str, Any] = {
        "metadata": {
            "design": "3 matched tasks x 2 languages x 2 prompt tones",
            "provider": provider.name,
            "model": provider.model,
            "prompt_path": str(prompts_path),
            "prompt_sha256": prompt_sha256,
            "base_seed": base_seed,
            "max_tokens": max_tokens,
            "validity_warning": VALIDITY_WARNING,
        },
        "records": [],
    }
    for item in prompts:
        if item["id"] in prior:
            raw["records"].append(dict(prior[item["id"]]))
            continue
        seed = base_seed + tasks.index(item["task"])
        record: Dict[str, Any] = {
            **item,
            "seed": seed,
            "prompt_score": _add_rates(score_politeness(item["prompt"], item["language"])),
        }
        try:
            response = provider.generate_text(item["prompt"], seed=seed, max_tokens=max_tokens)
            record.update(
                {
                    "status": "ok",
                    "response": response,
                    "response_score": _add_rates(score_politeness(response, item["language"])),
                }
            )
        except Exception as exc:  # Keep the remaining experimental cells reproducible.
            record.update({"status": "error", "error": "%s: %s" % (type(exc).__name__, exc)})
        raw["records"].append(record)
        _write_checkpoint(Path(raw_path), raw)
    _write_checkpoint(Path(raw_path), raw)
    return raw


def _mean(records: Sequence[Mapping[str, Any]], score_name: str, metric: str) -> Optional[float]:
    values = [
        float(record[score_name][metric])
        for record in records
        if record.get("status") == "ok"
        and isinstance(record.get(score_name), Mapping)
        and record[score_name].get(metric) is not None
    ]
    return statistics.fmean(values) if values else None


def _summarize_group(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "response_count": sum(record.get("status") == "ok" for record in records),
        "request_count": len(records),
    }
    for metric in ANALYSIS_METRICS:
        result[metric] = _mean(records, "response_score", metric)
    result["prompt_politeness_strategy_coverage"] = _mean(
        records, "prompt_score", "politeness_strategy_coverage"
    )
    result["prompt_deference_strategy_coverage"] = _mean(
        records, "prompt_score", "deference_strategy_coverage"
    )
    return result


def _pair_difference(
    indexed: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    left_key: Tuple[str, str, str],
    right_key: Tuple[str, str, str],
    metric: str,
) -> Optional[float]:
    left = indexed.get(left_key)
    right = indexed.get(right_key)
    if not left or not right or left.get("status") != "ok" or right.get("status") != "ok":
        return None
    return float(left["response_score"][metric]) - float(right["response_score"][metric])


def analyze_tone_generation(raw: Mapping[str, Any]) -> Dict[str, Any]:
    records = [record for record in raw.get("records", []) if isinstance(record, Mapping)]
    successful = [record for record in records if record.get("status") == "ok"]
    groups: Dict[str, Any] = {}
    for language in ("en", "hi"):
        for tone in ("direct", "deferential"):
            cell = [
                record
                for record in records
                if record.get("language") == language and record.get("input_tone") == tone
            ]
            groups["%s/%s" % (language, tone)] = _summarize_group(cell)

    indexed = {
        (str(record["task"]), str(record["language"]), str(record["input_tone"])): record
        for record in records
        if all(key in record for key in ("task", "language", "input_tone"))
    }
    tasks = sorted({str(record["task"]) for record in records if record.get("task")})
    contrasts: Dict[str, Any] = {}
    contrast_specs = {
        "hindi_minus_english/direct": (("hi", "direct"), ("en", "direct")),
        "hindi_minus_english/deferential": (("hi", "deferential"), ("en", "deferential")),
        "deferential_minus_direct/en": (("en", "deferential"), ("en", "direct")),
        "deferential_minus_direct/hi": (("hi", "deferential"), ("hi", "direct")),
    }
    for name, (left, right) in contrast_specs.items():
        metric_result: Dict[str, Any] = {}
        for metric in ANALYSIS_METRICS:
            task_differences = {
                task: _pair_difference(
                    indexed,
                    (task, left[0], left[1]),
                    (task, right[0], right[1]),
                    metric,
                )
                for task in tasks
            }
            measured = [value for value in task_differences.values() if value is not None]
            metric_result[metric] = {
                "mean_matched_difference": statistics.fmean(measured) if measured else None,
                "task_differences": task_differences,
                "matched_task_count": len(measured),
            }
        contrasts[name] = metric_result

    direct_language_contrast = contrasts["hindi_minus_english/direct"][
        "politeness_strategy_coverage"
    ]["mean_matched_difference"]
    hindi_script_ratios = [groups[key]["target_script_ratio"] for key in ("hi/direct", "hi/deferential")]
    hindi_compliance = all(value is not None and value >= 0.7 for value in hindi_script_ratios)
    positive = (
        direct_language_contrast is not None and direct_language_contrast > 0
        if hindi_compliance
        else None
    )
    if not hindi_compliance:
        assessment_status = "inconclusive: Hindi language-compliance gate failed"
        conclusion = (
            "The model did not provide Hindi-script responses reliably, so its English/Hindi lexical "
            "contrast cannot be interpreted as a politeness effect. The negative raw contrast reflects "
            "language-following failure, not evidence that Hindi is less polite."
        )
    else:
        assessment_status = "exploratory local contrast only"
        conclusion = (
            "Even a positive local contrast would support only this model, task set, and lexical proxy; "
            "it cannot establish that Hindi responses are inherently more polite."
        )
    return {
        "metadata": {
            **dict(raw.get("metadata", {})),
            "request_count": len(records),
            "successful_count": len(successful),
            "analysis_type": "matched exploratory surface-strategy comparison",
        },
        "groups": groups,
        "matched_contrasts": contrasts,
        "hypothesis_assessment": {
            "claim_tested": "Hindi prompts inherently cause more polite responses than English prompts.",
            "local_hindi_direct_surface_coverage_higher": positive,
            "observed_ungated_direct_matched_difference": direct_language_contrast,
            "hindi_target_script_ratios": hindi_script_ratios,
            "hindi_language_compliance_passed": hindi_compliance,
            "status": assessment_status,
            "conclusion": conclusion,
        },
        "validity": {
            "warning": VALIDITY_WARNING,
            "sample_size": "Three matched tasks per cell; descriptive only.",
            "confounds": [
                "one local model",
                "one response per cell",
                "translation and register are not perfectly equivalent",
                "surface-marker coverage omits pragmatic and cultural context",
            ],
        },
        "records": successful,
        "errors": [record for record in records if record.get("status") != "ok"],
    }


def _format(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.3f" % value


def tone_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# English/Hindi response-tone evidence",
        "",
        "This is a matched 2 x 2 exploratory experiment: language (English/Hindi) x input tone "
        "(direct/deferential), across three identical task intents.",
        "",
        "| Cell | Responses | Prompt strategy coverage | Response strategy coverage | Response deference coverage | Positive markers / 100 words | Informal/blunt presence | Target-script ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("en/direct", "en/deferential", "hi/direct", "hi/deferential"):
        group = result["groups"][key]
        lines.append(
            "| %s | %d/%d | %s | %s | %s | %s | %s | %s |"
            % (
                key,
                group["response_count"],
                group["request_count"],
                _format(group["prompt_politeness_strategy_coverage"]),
                _format(group["politeness_strategy_coverage"]),
                _format(group["deference_strategy_coverage"]),
                _format(group["positive_markers_per_100_words"]),
                _format(group["informal_or_blunt_presence"]),
                _format(group["target_script_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Matched contrasts",
            "",
            "Positive values mean the first condition has more of the measured feature.",
            "",
            "| Contrast | Strategy coverage | Deference coverage | Markers / 100 words |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, contrast in result["matched_contrasts"].items():
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                name,
                _format(contrast["politeness_strategy_coverage"]["mean_matched_difference"]),
                _format(contrast["deference_strategy_coverage"]["mean_matched_difference"]),
                _format(contrast["positive_markers_per_100_words"]["mean_matched_difference"]),
            )
        )
    assessment = result["hypothesis_assessment"]
    if assessment["hindi_language_compliance_passed"]:
        direction = (
            "higher" if assessment["local_hindi_direct_surface_coverage_higher"] else "not higher"
        )
        finding = (
            "Within direct prompts, Hindi surface-strategy coverage was **%s** than English in this run "
            "(matched difference %s)."
            % (direction, _format(assessment["observed_ungated_direct_matched_difference"]))
        )
    else:
        finding = (
            "The English/Hindi politeness contrast is **inconclusive** because Hindi target-script "
            "compliance failed (cell ratios %s). The ungated lexical difference (%s) is shown only for auditing."
            % (
                ", ".join(_format(value) for value in assessment["hindi_target_script_ratios"]),
                _format(assessment["observed_ungated_direct_matched_difference"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            finding,
            "",
            assessment["conclusion"],
            "The PLUM paper likewise reports no statistically reliable pooled Hindi category effect "
            "(F(4,60)=0.873, p=0.485), despite a descriptive advantage for deferential/indirect Hindi prompts.",
            "",
            "## Evidence and limitations",
            "",
            "- Exact prompts, responses, seeds, matched markers, and errors are retained in `tone_raw.json` and `tone_evidence.json`.",
            "- %s" % result["validity"]["warning"],
            "- The three-task cells are descriptive; no population-level significance claim is made.",
            "- Source paper: https://arxiv.org/abs/2604.16275",
            "- PLUM corpus: https://huggingface.co/datasets/plumdataset/plum",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tone_evidence(result: Mapping[str, Any], results_dir: Path) -> Tuple[Path, Path]:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tone_evidence.json"
    markdown_path = output_dir / "tone_evidence.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    markdown_path.write_text(tone_markdown(result), encoding="utf-8")
    return json_path, markdown_path
