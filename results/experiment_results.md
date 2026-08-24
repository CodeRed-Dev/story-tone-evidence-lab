# Multilingual storytelling experiment

Result tier: **fresh_single_model_local_experiment**

Model: `orca-mini:latest` via `ollama`  
Successful generations: **9 / 9**  
Failed generations: **0**

## Condition comparison

| Condition | n | Arc entropy | Plot repeat rate | Suspense | Flattening | Token distance | Human arc distance | TP order violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| monolingual | 3 | 0 | 0.3333 | 0.0025 | 0.152 | 0.7586 | 0.6945 | 2/3 |
| multicultural | 3 | 0 | 0.3333 | 0.0059 | 0.5564 | 0.7259 | 0.6945 | 3/3 |
| multilingual | 3 | 0.5794 | 0 | 0 | 0.3217 | 0.7547 | 0.6631 | 3/3 |

## Narrative parameter breakdown

| Parameter cell | n | Arc entropy | Plot repeat rate | Suspense | Valence range | Arousal range | Flattening |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 9 | 0.1846 | 0.4444 | 0.0028 | 0.2744 | 0.0722 | 0.3433 |

## Language breakdown

| Group | n | Arc entropy | Token distance |
|---|---:|---:|---:|
| monolingual/en-a | 1 | 0 | 0 |
| monolingual/en-b | 1 | 0 | 0 |
| monolingual/en-c | 1 | 0 | 0 |
| multicultural/en | 1 | 0 | 0 |
| multicultural/ja | 1 | 0 | 0 |
| multicultural/zh | 1 | 0 | 0 |
| multilingual/en | 1 | 0 | 0 |
| multilingual/ja | 1 | 0 | 0 |
| multilingual/zh | 1 | 0 | 0 |

## Descriptive hypothesis checks

- **H1:** Multilingual minus monolingual normalized arc entropy = 0.5794.
- **H2:** Multilingual minus monolingual token Jaccard diversity = -0.0039.
- **H3:** Arc-entropy gain over monolingual: language shift 0.5794 versus persona-only 0.0000.
- **H4:** Turning-point timing is not interpreted: 8/9 annotations were non-monotonic.
- **H5:** Multilingual minus monolingual human arc distance = -0.0314; diversity and human-likeness are reported separately.
- **H6:** Unavailable: at least two narrative parameter cells are required to test parameter sensitivity.
- **H7:** Highest repeated plot-signature rate = 0.4444 in parameter cell `neutral`.

## Interpretation boundary

This is an exploratory single-model local experiment with only three stories per condition. A separate local-model pass assigned arcs and turning points after generation; those labels require blind independent annotation before publication-quality conclusions. Non-monotonic turning-point annotations are retained and counted rather than silently repaired, so the human TP distances are diagnostic only. The affective analysis uses a small transparent proxy lexicon, not the paper's full GPT-4 plus NRC VAD pipeline. The requested English output makes lexical comparison direct but tests prompt-language effects rather than native-language prose.
