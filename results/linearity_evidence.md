# AI versus human narrative linearity evidence

This report analyzes all 1638 local synopses using an inspectable lexical chronology proxy. It is a directional extension of StoryScope, not a reproduction of its 304-feature model.

| Measure | Human mean | GPT mean | Human - GPT | Bootstrap 95% CI | Cohen's d |
|---|---:|---:|---:|---:|---:|
| Stories with retrospective cues | 0.814 | 0.596 | 0.219 | [0.176, 0.264] | 0.493 |
| Retrospective cues / 100 sentences | 6.250 | 3.176 | 3.073 | [2.604, 3.583] | 0.611 |
| Stories with time-jump cues | 0.740 | 0.277 | 0.463 | [0.418, 0.507] | 1.044 |
| Time-jump cues / 100 sentences | 4.267 | 0.942 | 3.326 | [3.042, 3.618] | 1.140 |
| Stories with forward cues | 0.977 | 0.675 | 0.302 | [0.266, 0.336] | 0.867 |
| Forward cues / 100 sentences | 12.455 | 3.526 | 8.929 | [8.369, 9.473] | 1.548 |
| Forward share (when ordered cue exists) | 0.679 | 0.551 | 0.128 | [0.096, 0.159] | 0.430 |

## Interpretation

The two local discontinuity proxies **align** with StoryScope's reported direction that human fiction is more temporally complex than AI fiction.

Presence measures are proportions from 0 to 1. The confidence intervals resample stories within each source. A positive human-minus-GPT discontinuity contrast supports the claimed direction; it does not prove that every human story is nonlinear or every AI story is linear.

## Provenance

- Input SHA-256: `a15f0d13dc242b91aa988d609c444ec269e769d289af64e5d1ddd75c109df793`
- Bootstrap iterations: 2000
- StoryScope: https://arxiv.org/abs/2604.03136
