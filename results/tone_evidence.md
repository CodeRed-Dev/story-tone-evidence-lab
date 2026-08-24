# English/Hindi response-tone evidence

This is a matched 2 x 2 exploratory experiment: language (English/Hindi) x input tone (direct/deferential), across three identical task intents.

| Cell | Responses | Prompt strategy coverage | Response strategy coverage | Response deference coverage | Positive markers / 100 words | Informal/blunt presence | Target-script ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| en/direct | 3/3 | 0.000 | 0.167 | 0.111 | 2.146 | 0.000 | 1.000 |
| en/deferential | 3/3 | 0.444 | 0.167 | 0.000 | 1.685 | 0.000 | 1.000 |
| hi/direct | 3/3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hi/deferential | 3/3 | 0.556 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Matched contrasts

Positive values mean the first condition has more of the measured feature.

| Contrast | Strategy coverage | Deference coverage | Markers / 100 words |
|---|---:|---:|---:|
| hindi_minus_english/direct | -0.167 | -0.111 | -2.146 |
| hindi_minus_english/deferential | -0.167 | 0.000 | -1.685 |
| deferential_minus_direct/en | 0.000 | -0.111 | -0.461 |
| deferential_minus_direct/hi | 0.000 | 0.000 | 0.000 |

## Interpretation

The English/Hindi politeness contrast is **inconclusive** because Hindi target-script compliance failed (cell ratios 0.000, 0.000). The ungated lexical difference (-0.167) is shown only for auditing.

The model did not provide Hindi-script responses reliably, so its English/Hindi lexical contrast cannot be interpreted as a politeness effect. The negative raw contrast reflects language-following failure, not evidence that Hindi is less polite.
The PLUM paper likewise reports no statistically reliable pooled Hindi category effect (F(4,60)=0.873, p=0.485), despite a descriptive advantage for deferential/indirect Hindi prompts.

## Evidence and limitations

- Exact prompts, responses, seeds, matched markers, and errors are retained in `tone_raw.json` and `tone_evidence.json`.
- This detector measures explicit lexical strategies only. It does not measure prosody, context, sincerity, regional norms, or total politeness, and English/Hindi scores are not assumed to be perfectly culturally equivalent.
- The three-task cells are descriptive; no population-level significance claim is made.
- Source paper: https://arxiv.org/abs/2604.16275
- PLUM corpus: https://huggingface.co/datasets/plumdataset/plum
