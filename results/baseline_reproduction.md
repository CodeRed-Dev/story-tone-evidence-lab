# Baseline reproduction

This report is a direct local analysis of the official released narrative annotations.
It is not a fresh generation run.

## Dataset

| Measure | Count |
|---|---:|
| Full narrative corpus | 1638 |
| Story-arc labels | 439 |
| Turning-point labels | 439 |
| Joined annotated stories | 439 |
| Missing narrative IDs | 0 |
| Incomplete annotation IDs | 0 |

## Story arcs

| Arc | Human | GPT |
|---|---:|---:|
| Rags to Riches | 4.5% | 13.1% |
| Riches to Rags | 13.9% | 1.3% |
| Man in Hole | 31.2% | 51.1% |
| Double Man in Hole | 20.8% | 12.7% |
| Icarus | 5.0% | 3.0% |
| Cinderella | 15.3% | 17.3% |
| Oedipus | 9.4% | 1.7% |
| **Normalized entropy** | 0.9051 | 0.7208 |

## Mean normalized turning-point positions

| Turning point | Human | GPT | GPT − Human |
|---|---:|---:|---:|
| TP1 | 0.1289 | 0.1426 | 0.0137 |
| TP2 | 0.2661 | 0.2747 | 0.0086 |
| TP3 | 0.4782 | 0.4467 | -0.0315 |
| TP4 | 0.6664 | 0.5734 | -0.093 |
| TP5 | 0.8796 | 0.7658 | -0.1138 |

Arc Jensen–Shannon distance: **0.2743**.

## Interpretation boundary

These values describe aggregate structural differences in the released annotated subset. They do not classify individual texts or measure overall writing quality.
