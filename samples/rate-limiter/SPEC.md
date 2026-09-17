# SPEC — Client Rate Limiter & Audit Log

A small in-process service that throttles API clients and keeps an audit
trail of their requests, with an admin override to reset a client's quota.

## Requirements

1. **Sliding-window throttling.** Each client is allowed at most
   `MAX_REQUESTS` requests within any `WINDOW_SECONDS`-second sliding
   window. A request outside the window must not count against the
   client's current quota — old timestamps must be evicted correctly at
   the window boundary.
2. **Request logging.** Every accepted or rejected request is appended to
   an on-disk audit log with client id, timestamp, and outcome.
3. **Admin actions.** An admin holding `ADMIN_TOKEN` may reset a specific
   client's quota early (e.g. to unblock a client after resolving an
   incident). Admin actions are themselves logged.
4. **Audit export.** Operators can export a client's audit trail to a file
   on demand, filtered by client id, for later inspection.

## Non-functional expectations

- No secret material should be committed to source control or compared
  in a way that leaks timing information.
- No user-controlled input should reach a shell or a query in a way that
  lets a client escape its own data.
- The implementation should be maintainable: no duplicated logic, no
  unexplained magic numbers, clear naming.

## Out of scope

- Persistence backend choice, distributed rate limiting across processes,
  and authentication of the admin caller are intentionally left simple
  for this sample — reviewers should focus on the module as written.
