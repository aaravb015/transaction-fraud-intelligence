# Transaction fraud detection

A Python project to see whether a customer's past transactions help identify fraud better than looking at the current transaction alone.

The data is synthetic. I compared rules, logistic regression with customer history, CatBoost without history and CatBoost with history. Each model gets the same daily review budget.

## Run it

```bash
pip install -r requirements.txt
python fraud.py
```

This runs the development experiment with seed 42 and writes a model comparison and review queue to `outputs/`. Use `--seed 123` to try another development seed.

## Results

Original V2 reserved-period results, averaged across five seeds at 50 reviews per day. These were not recalculated after the cleanup.

| Model | Average precision | Precision | Recall | Fraud value captured |
| --- | ---: | ---: | ---: | ---: |
| Rules | 0.045 | 0.077 | 0.228 | 0.299 |
| CatBoost (current transaction) | 0.086 | 0.077 | 0.231 | 0.051 |
| Logistic regression (history) | 0.388 | 0.196 | 0.588 | 0.438 |
| CatBoost (history) | 0.582 | 0.224 | 0.671 | 0.552 |

Customer history helped more than simply using a more complex model. Low-and-slow fraud was still difficult to detect.

The data is simulated, so these numbers are not evidence of real-world accuracy or savings. The review queue is an offline ranking, not a live approval or decline system.

The [original report](https://github.com/aaravb015/transaction-fraud-intelligence/blob/55577e228143f45e06c919b5f0801782a64f9e2e/docs/final_evaluation_report.md) and [evaluation evidence](https://github.com/aaravb015/transaction-fraud-intelligence/tree/55577e228143f45e06c919b5f0801782a64f9e2e/artifacts/final_evaluation) are preserved in the original commit. The exact evaluator that produced the final archive was not checked in. The simplified script only uses development days 0–101; the original final period (days 102–119) is consumed and should not be used for tuning.

## Tests

```bash
pip install pytest
python -m pytest -q
```
