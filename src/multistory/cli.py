"""Command-line entry point for reproductions and new local experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .experiment import analyze_generation, run_baseline, run_generation
from .linearity import analyze_linearity_corpus, write_linearity_evidence
from .prompts import research_narrative_parameters
from .providers import OllamaProvider
from .reporting import baseline_markdown, experiment_markdown, write_json, write_text
from .schema import CONDITIONS, StoryBrief
from .tone_experiment import analyze_tone_generation, run_tone_generation, write_tone_evidence


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_briefs(path: Path, limit: Optional[int]) -> List[StoryBrief]:
    value = _load_json(path)
    if not isinstance(value, list):
        raise ValueError("briefs file must contain a JSON list")
    briefs = [StoryBrief.from_dict(item) for item in value]
    return briefs[:limit] if limit else briefs


def _save_baseline(data_dir: Path, results_dir: Path) -> Any:
    baseline = run_baseline(data_dir)
    write_json(results_dir / "baseline_reproduction.json", baseline)
    write_text(results_dir / "baseline_reproduction.md", baseline_markdown(baseline))
    return baseline


def _save_analysis(raw: Any, baseline: Any, results_dir: Path, vad_path: Optional[Path]) -> Any:
    result = analyze_generation(raw, baseline, vad_path=vad_path)
    write_json(results_dir / "experiment_results.json", result)
    write_text(results_dir / "experiment_results.md", experiment_markdown(result))
    return result


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    _add_paths(parser)
    parser.add_argument("--briefs", type=Path, default=Path("examples/briefs.json"))
    parser.add_argument("--brief-count", type=int, default=1)
    parser.add_argument("--model", default="orca-mini:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--languages", nargs="+", choices=("en", "zh", "ja"), default=["en", "zh", "ja"])
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--vad-lexicon", type=Path)
    parser.add_argument(
        "--narrative-parameter-set",
        choices=("neutral", "research"),
        default="neutral",
        help="neutral preserves the original design; research adds explicit storytelling parameter cells",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine multilingual prompting with hierarchical narrative analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="analyze official human/GPT annotations")
    _add_paths(baseline)

    experiment = subparsers.add_parser("experiment", help="generate and analyze local stories")
    _add_generation_options(experiment)

    full = subparsers.add_parser("full", help="run baseline and local experiment")
    _add_generation_options(full)

    analyze = subparsers.add_parser("analyze", help="reanalyze an existing raw generation file")
    _add_paths(analyze)
    analyze.add_argument("--raw", type=Path, default=Path("results/experiment_raw.json"))
    analyze.add_argument("--vad-lexicon", type=Path)

    linearity = subparsers.add_parser(
        "linearity", help="compare human/GPT temporal-structure cues"
    )
    _add_paths(linearity)
    linearity.add_argument("--bootstrap-iterations", type=int, default=2000)
    linearity.add_argument("--seed", type=int, default=260403136)

    tone = subparsers.add_parser(
        "tone", help="run the matched English/Hindi response-tone experiment"
    )
    tone.add_argument("--results-dir", type=Path, default=Path("results"))
    tone.add_argument("--prompts", type=Path, default=Path("examples/tone_prompts.json"))
    tone.add_argument("--model", default="orca-mini:latest")
    tone.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    tone.add_argument("--seed", type=int, default=260416275)
    tone.add_argument("--temperature", type=float, default=0.2)
    tone.add_argument("--timeout", type=int, default=600)
    tone.add_argument("--max-tokens", type=int, default=260)
    tone.add_argument("--no-resume", action="store_true")
    return parser


def _run_local(args: argparse.Namespace, baseline: Any) -> Any:
    briefs = _load_briefs(args.briefs, args.brief_count)
    if not briefs:
        raise ValueError("briefs selection is empty")
    provider = OllamaProvider(
        model=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
    )
    raw = run_generation(
        provider,
        briefs,
        args.results_dir / "experiment_raw.json",
        conditions=args.conditions,
        languages=args.languages,
        replicates=args.replicates,
        base_seed=args.seed,
        narrative_parameters=(
            research_narrative_parameters()
            if args.narrative_parameter_set == "research"
            else None
        ),
    )
    result = _save_analysis(raw, baseline, args.results_dir, args.vad_lexicon)
    print(
        "Generated %s/%s stories; results: %s"
        % (
            result["metadata"]["successful_count"],
            result["metadata"]["request_count"],
            args.results_dir / "experiment_results.md",
        )
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "linearity":
        result = analyze_linearity_corpus(
            args.data_dir / "narratives.json",
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        _, report_path = write_linearity_evidence(result, args.results_dir)
        print("Analyzed %d stories; results: %s" % (result["metadata"]["story_count"], report_path))
        return 0
    if args.command == "tone":
        provider = OllamaProvider(
            model=args.model,
            base_url=args.ollama_url,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
        )
        raw = run_tone_generation(
            provider,
            args.prompts,
            args.results_dir / "tone_raw.json",
            base_seed=args.seed,
            max_tokens=args.max_tokens,
            resume=not args.no_resume,
        )
        result = analyze_tone_generation(raw)
        _, report_path = write_tone_evidence(result, args.results_dir)
        print(
            "Generated %d/%d responses; results: %s"
            % (result["metadata"]["successful_count"], result["metadata"]["request_count"], report_path)
        )
        return 0 if result["metadata"]["successful_count"] == result["metadata"]["request_count"] else 2
    if args.command == "baseline":
        baseline = _save_baseline(args.data_dir, args.results_dir)
        print(
            "Analyzed %s joined annotations; results: %s"
            % (baseline["dataset"]["joined_annotation_count"], args.results_dir / "baseline_reproduction.md")
        )
        return 0
    if args.command == "analyze":
        baseline = _save_baseline(args.data_dir, args.results_dir)
        result = _save_analysis(_load_json(args.raw), baseline, args.results_dir, args.vad_lexicon)
        print("Analyzed %s successful stories." % result["metadata"]["successful_count"])
        return 0 if result["metadata"]["successful_count"] else 2
    baseline = _save_baseline(args.data_dir, args.results_dir)
    result = _run_local(args, baseline)
    return 0 if result["metadata"]["successful_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
