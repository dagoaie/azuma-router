# Results — review pipeline comparison on the rate-limiter sample

Four Convoy `review-*` pipelines (scope → parallel[bugs, security, clean-code]
→ report, all real OpenAI Codex models, no OpenRouter) were run against the
same commit of `rate_limiter.py` / `SPEC.md`, which has six deliberately
seeded issues (see `ANSWER_KEY.md`).

| Pipeline | Step models (scope / bugs / security / clean-code / report) | Seeded issues found | False positives | Wall time | Total tokens |
|---|---|---|---|---|---|
| `review-azuma-gpt-1` | Luna#none × 5 | 6/6 | 0 | 87s | 264K |
| `review-gpt` (existing) | Luna#none / Terra#medium / Sol#high / Luna#none / Sol#medium | 6/6 | 0 | 171s | 432K |
| `review-azuma-gpt-3` | Luna#none / Terra#high / Sol#high / Terra#medium / Sol#high | 6/6 | 0 | 252s | 599K |
| `review-azuma-gpt-2` | Sol#high × 5 | 6/6 (+2 legit extra) | 0 | 411s | 1096K |

## Conclusion

For this sample/spec, every configuration found all six seeded bugs,
security issues and clean-code problems, with zero noise (no invented
findings) in any run. Quality was not the differentiator — cost and speed
were: `review-azuma-gpt-1` (cheapest models, no reasoning effort on every
step) matched the most expensive configuration's recall and precision at
roughly **1/5th the time and tokens**.

The only real gain from spending more (`review-azuma-gpt-2`, all Sol at high
effort) was two additional, legitimate but non-critical findings (a log-
injection edge case and a note about missing type hints/tests) that the
cheaper variants either omitted or explicitly triaged away with sound
reasoning.

**Recommended default for this class of change (small, self-contained diff,
clear spec): `review-azuma-gpt-1`.** Reserve `review-azuma-gpt-2`/-3 or
`review-gpt` for larger, higher-stakes diffs where the marginal extra
findings are more likely to matter.

Full per-variant reports are under each run's `reports/report.md` in
`~/.convoy/runs/` (run ids recorded alongside this file were current as of
2026-09-17; Convoy prunes old runs over time).

- `review-azuma-gpt-1`: `20260917-083502-64a9`
- `review-azuma-gpt-2`: `20260917-083655-0u9b`
- `review-azuma-gpt-3`: `20260917-084402-g9iz`
- `review-gpt`: `20260917-084837-y79i`
