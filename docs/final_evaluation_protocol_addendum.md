# Final-evaluation protocol addendum

This addendum is recorded **before the reserved period is materialized or scored**.

During implementation review, one detail in `docs/final_evaluation_protocol.md` was tightened to make the freeze protection stronger: the final evaluator does **not** reopen the simulator horizon during the original development-generation loop and does not attempt to preserve tails of fraud episodes that the frozen V2 simulator had deliberately dropped at the day-102 boundary.

Doing that would cause those previously dropped events to consume random draws inside the original fraud loop and could change later development-period events. That would violate the requirement that the frozen development world remain exactly unchanged.

The implemented rule is therefore:

1. Run the frozen V2 simulator exactly as frozen through its development-only generation sequence.
2. Preserve the exact normal and fraud RNG consumption used by frozen V2.
3. Only **after** all frozen development draws are complete, open the horizon to days 102–119 and generate the reserved continuation using new draws from the resulting RNG states.
4. Use duration-scaled conditional probabilities for customers who had not already received a one-off phone change, travel episode, legitimate burst, or fraud episode in development.
5. Reuse the frozen fraud mechanics for new reserved-period episodes.
6. Require exact row-for-row equality of the entire `< day 102` transaction prefix and exact feature-for-feature equality of every development transaction before any final result is accepted.

This change is an engineering safeguard, not a response to final-period evidence: no final-period rows or outcomes have been generated or inspected at the time of this addendum.

If either exact-prefix check fails during the official run, the final output is invalid and the runner stops.