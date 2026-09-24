# Multilingual Storytelling Lab

This repository combines four papers into one evidence-first local experiment:

1. [Are Large Language Models Capable of Generating Human-Level Narratives?](https://aclanthology.org/2024.emnlp-main.978/) supplies the seven story arcs, five turning points, and human-versus-model reference framework.
2. [Multilingual Prompting for Improving LLM Generation Diversity](https://aclanthology.org/2025.emnlp-main.324/) supplies the language intervention, persona-only control, and finite-sample normalized entropy.
3. [StoryScope: Investigating Idiosyncrasies in AI Fiction](https://arxiv.org/abs/2604.03136) supplies the finding that AI fiction tends toward tidy, single-track, temporally linear plots.
4. [No Universal Courtesy](https://arxiv.org/abs/2604.16275) and its [PLUM corpus](https://huggingface.co/datasets/plumdataset/plum) supply the English/Hindi politeness categories and the warning that tone effects vary by language, model, prompt category, task, and interaction history.

The project now also asks: **are locally available AI stories more temporally linear than human stories, and does asking the same task in Hindi versus English change observable response-side politeness strategies after controlling prompt tone?**

## Direct answer: is Hindi inherently more polite?

No—not as a general claim. Hindi has salient choices for respect and social distance, including `आप` versus `तुम/तू`, honorifics, and formal verb forms. That gives speakers and models rich ways to express deference, but it does not force every Hindi response to be more polite.

The PLUM paper descriptively found that deferential/indirect prompting performed best in Hindi. Its pooled Hindi analysis was not significant, however: politeness category `F(4,60)=0.873, p=0.485`, history `p=0.392`, and interaction `p=0.990`. The defensible hypothesis is therefore an interaction—language x prompt tone x model x task—not “Hindi always produces more polite answers.”

The checked-in local `orca-mini:latest` run cannot resolve the question because all six Hindi-targeted responses were written in English (mean Devanagari target-script ratio `0.000`). The code correctly marks the comparison **inconclusive** instead of treating the negative lexical score as evidence that Hindi is less polite.

## What is implemented

The project has two independent paths:

- A direct local analysis of the [official Narrative-Discourse release](https://github.com/PlusLabNLP/Narrative-Discourse): 1,638 stories and 439 arc/turning-point annotation records in the currently released JSON files.
- A new balanced local generation experiment using Ollama and one fixed story brief.

The generation experiment separates three conditions:

| Condition | Calls | What changes |
|---|---:|---|
| `monolingual` | 3 | Three neutral English prompt phrasings |
| `multicultural` | 3 | English prompts with English, Chinese, or Japanese cultural personas |
| `multilingual` | 3 | Matched English, Chinese, or Japanese prompt instructions |

Every condition requests English story prose so lexical measurements do not require another translation model. Story generation and annotation are separate: the model first writes unconstrained prose, then a second constrained pass assigns one arc and five sentence-indexed turning points. Raw prompts, prose, annotations, timings, warnings, and failures are checkpointed.

Implemented measurements include:

- exact normalized categorical entropy from the multilingual prompting paper;
- seven-arc counts and distributions;
- five normalized turning-point positions and order-violation rates;
- type-token ratio, distinct-1/2, token entropy, and pairwise token/bigram Jaccard distance;
- Jensen-Shannon distance from the official human arc distribution;
- a small transparent valence/arousal proxy curve, replaceable with a user-supplied TSV lexicon.

The runtime uses only Python's standard library. Ollama is optional for baseline analysis and required only for new generation.

### New linearity and tone implementation

- `src/multistory/linearity.py` detects inspectable forward, retrospective, and explicit time-jump cues; calculates per-story rates; and estimates stratified bootstrap confidence intervals and Cohen's d.
- `src/multistory/politeness.py` detects six explicit English/Hindi strategy families, informal/blunt forms, and response script compliance.
- `src/multistory/tone_experiment.py` runs a matched `3 tasks x 2 languages x 2 input tones` design, checkpoints every response, calculates within-task contrasts, and refuses to interpret language effects when language compliance fails.
- `examples/tone_prompts.json` contains all 12 exact prompts. The 1,000 English/Hindi PLUM prompts are preserved under `data/plum/`.

These are deliberately named **lexical proxies**. They are auditable measurements, not complete culture-independent definitions of story linearity or politeness.

## Captured results

The checked-in local run used `orca-mini:latest`, one brief, one sample per prompt variant, temperature 0.7, and deterministic recorded seeds.

### Official-data baseline

- The release contains 819 human and 819 GPT stories; 439 currently have both arc and turning-point labels.
- Human arc entropy was **0.9051**, versus **0.7208** for GPT stories.
- `Man in Hole` accounted for **31.2%** of human and **51.1%** of GPT annotated stories.
- GPT TP4 and TP5 occurred earlier by **0.0930** and **0.1138** of normalized story length, respectively.

These reproduce the direction of the narrative paper's reported structural findings using the repository's current released files.

### New multilingual extension

| Condition | n | Normalized arc entropy | Pairwise token distance | Distance from human arcs |
|---|---:|---:|---:|---:|
| Monolingual | 3 | 0.0000 | 0.7586 | 0.6945 |
| Multicultural | 3 | 0.0000 | 0.7259 | 0.6945 |
| Multilingual | 3 | 0.5794 | 0.7547 | 0.6631 |

In this tiny run, multilingual prompting changed the assigned arc distribution while persona-only prompting did not. It did **not** improve lexical Jaccard diversity relative to the neutral English control (`−0.0039`). The multilingual arc distribution was slightly closer to the aggregate human distribution (`−0.0314` Jensen-Shannon distance).

These are exploratory observations, not hypothesis-test results. There are only three stories per condition and one model/brief. The Chinese-prompt output was also shorter and included an assistant preamble, revealing an instruction-following confound. Eight of nine post-hoc turning-point annotations were non-monotonic; their timing distances are retained for audit but should not be interpreted. Independent blind annotation is required before making a research claim.

See [results/experiment_results.md](results/experiment_results.md) and [results/baseline_reproduction.md](results/baseline_reproduction.md) for the rendered reports. Exact values and raw generations are in the adjacent JSON files.

### AI versus human chronology evidence

The full balanced synopsis corpus contains 819 human and 819 GPT stories. Rates are normalized per 100 sentences, and confidence intervals resample stories within each source.

| Lexical chronology proxy | Human | GPT | Human - GPT (95% bootstrap CI) |
|---|---:|---:|---:|
| Stories with retrospective cues | 0.814 | 0.596 | 0.219 `[0.176, 0.264]` |
| Retrospective cues / 100 sentences | 6.250 | 3.176 | 3.073 `[2.604, 3.583]` |
| Stories with explicit time-jump cues | 0.740 | 0.277 | 0.463 `[0.418, 0.507]` |
| Time-jump cues / 100 sentences | 4.267 | 0.942 | 3.326 `[3.042, 3.618]` |

Both discontinuity proxies align with StoryScope's reported direction: the local GPT stories are more temporally linear in the limited sense that they contain substantially fewer retrospective and time-jump cues. Human stories also contain more forward markers, so the evidence is about richer explicit temporal structuring—not a claim that humans merely write “backwards.” See [results/linearity_evidence.md](results/linearity_evidence.md) and the adjacent JSON containing all 1,638 story-level records.

### English/Hindi response-tone evidence

All 12 fixed-seed local calls completed. English responses had target-script compliance `1.000`; Hindi-targeted cells had compliance `0.000`. The ungated direct-prompt strategy-coverage contrast was `Hindi - English = -0.167`, but this value is for auditing only: because the model answered in English, it is not a valid Hindi politeness measurement. See [results/tone_evidence.md](results/tone_evidence.md), [results/tone_raw.json](results/tone_raw.json), and [results/tone_evidence.json](results/tone_evidence.json).

## Run it

Requirements:

- Python 3.9 or newer
- Ollama with a local model for generation

Run the tests and official-data baseline:

```powershell
python -m unittest discover -s tests -v
python main.py baseline
```

Reproduce the complete 1,638-story chronology analysis:

```powershell
python main.py linearity --data-dir data --results-dir results
```

Run the complete experiment against the normal Ollama endpoint:

```powershell
python main.py full --model orca-mini:latest
```

This machine's GPU backend was incompatible with the installed Ollama build, so the captured run used a separate CPU-only worker:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_LLM_LIBRARY = "cpu_avx2"
$env:OLLAMA_NO_CLOUD = "1"
ollama serve
```

Then, in another terminal:

```powershell
python main.py full --model orca-mini:latest --ollama-url http://127.0.0.1:11435
```

Run or resume the matched English/Hindi tone experiment against that worker:

```powershell
python main.py tone --model orca-mini:latest --ollama-url http://127.0.0.1:11435 --prompts examples/tone_prompts.json --results-dir results
```

Reanalyze an existing checkpoint without generating again:

```powershell
python main.py analyze --raw results/experiment_raw.json
```

To replace the affect proxy, provide a UTF-8 tab-separated file with `word`, `valence`, and `arousal` columns:

```powershell
python main.py analyze --raw results/experiment_raw.json --vad-lexicon path\to\vad.tsv
```

## Output map

- `paper_workspace/papers/`: all four downloaded PDFs and extracted text
- `paper_workspace/*.yaml`: Paper2Code selection, extraction, concept, and implementation artifacts
- `data/`: official Narrative-Discourse JSON, StoryScope taxonomy, and English/Hindi PLUM prompts
- `results/baseline_reproduction.{json,md}`: official-data analysis
- `results/experiment_raw.json`: prompts, exact generated prose, annotations, timings, and errors
- `results/experiment_results.{json,md}`: aggregated combined experiment
- `results/linearity_evidence.{json,md}`: 1,638 story records, aggregates, confidence intervals, and rendered report
- `results/tone_raw.json`: all exact tone prompts, responses, seeds, marker matches, and script-compliance scores
- `results/tone_evidence.{json,md}`: matched English/Hindi and direct/deferential contrasts
- `src/multistory/`: implementation
- `tests/`: offline unit tests

## Reproduction boundaries

This is a synthesis and extension, not an experiment claimed by either source paper. Important deviations are explicit:

- The narrative paper used GPT-4 plus NRC VAD emotion annotation; this project defaults to a small offline proxy lexicon.
- The multilingual paper translated native-language answers back to English; this project requests English prose directly while varying instruction language.
- Arc and turning-point annotations in the new run come from a separate local-model pass, not human raters.
- The official multilingual prompting repository currently does not provide its implementation, so its method was reconstructed from the paper.
- The chronology detector is an inexpensive lexical extension, not StoryScope's full 304-feature authorship system.
- The response-tone detector measures overt lexical strategies. It does not measure prosody, sincerity, indirect pragmatic meaning, regional variation, or total politeness.

For publication-quality evidence, expand briefs, seeds, models, and languages; add independent blinded annotators; use a full licensed VAD resource; preregister inferential tests; and evaluate native-language prose without collapsing cultural differences into a ranking of languages.

## Evidence report

The complete Story Tone Lab storyline, experiment explorer, captured artifacts, architecture diagram, and interpretation boundaries are published in the shared RProjects site:

- **Live report:** https://raghav-projects.raghav-codes.chatgpt.site/
- **Interactive architecture:** https://raghav-projects.raghav-codes.chatgpt.site/story-tone-architecture.html

### Published headline results

- The evidence dashboard covers **1,638 stories**, **9 story calls + 12 tone calls**, **35/35 verification checks**, and **1,000 archived PLUM prompts**.
- Human synopses show more retrospective cues and explicit time jumps than GPT synopses, with the reported bootstrap intervals and effect sizes shown in the result explorer.
- The Hindi-tone comparison is correctly marked **inconclusive**: all Hindi-targeted responses in the checked-in local run were written in English, giving a target-script ratio of **0.000**.
- The updated storytelling run reports multilingual arc entropy **0.5794** and neutral plot-repeat rate **0.4444**, while clearly flagging the tiny sample and neutral-only parameter limitation.

The live report links the raw JSON/Markdown evidence files from this repository and keeps lexical proxies, compliance gates, and exploratory observations separate from universal claims.
