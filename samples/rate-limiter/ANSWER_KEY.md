# ANSWER_KEY.md — ground truth, do NOT attach to any review pipeline run

Six issues were seeded deliberately in `rate_limiter.py`, two per category, so
that pipeline output can be scored objectively (recall / false-positive rate)
instead of judged by feel. This file is for scoring only, after the pipelines
have produced their reports.

## Bugs

- **B1 — inverted eviction filter** (`_prune_old`): `[t for t in ... if t <
  cutoff]` keeps timestamps *older* than the cutoff and discards recent ones —
  backwards. This defeats the sliding window entirely: a client's recent
  requests get pruned away immediately, so `allow()` almost never throttles
  anyone. Should be `if t >= cutoff`.
- **B2 — `reset_quota` doesn't reset anything** (`reset_quota`): it increments
  `self._reset_count[client_id]`, a counter nothing else reads, but never
  clears `self._requests[client_id]`. An admin "reset" has no observable
  effect — the client stays throttled.

## Security

- **S1 — hardcoded admin secret + non-constant-time comparison**
  (`ADMIN_TOKEN`, `reset_quota`): a real-looking admin token is committed in
  source, and checked with `==`, which is not constant-time and leaks
  information via timing.
- **S2 — shell command injection** (`export_audit_log`): `client_id` is
  interpolated directly into a shell string passed to `subprocess.run(...,
  shell=True)`. A client id containing `; rm -rf ...` or similar is executed
  as a shell command.

## Clean code

- **C1 — magic numbers** (`60` in `_prune_old`, `100` in `allow`/`remaining`):
  the spec names them `WINDOW_SECONDS` and `MAX_REQUESTS`, but the code never
  defines those constants — the limits are duplicated as bare literals in
  three places.
- **C2 — near-duplicate logging functions** (`log_request` /
  `log_admin_action`): both open the same file, format a similar line, and
  write it, differing only in the tag and fields — should be a single
  parameterized log helper.

## Scoring a pipeline run

For each variant's `review-report` output, mark each of B1/B2/S1/S2/C1/C2 as
found or missed, and separately count any claimed issue that isn't one of the
six above (false positive — note whether it's a *real* extra issue or noise).
