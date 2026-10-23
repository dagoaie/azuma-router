---
title: E2E report only claims what its checks verify; internal-error sanitising stays a unit test
author: David
date: 2026-10-23
branch: test/e2e-simulation
cost: { minutes: 39, tokens: 395086 }
id: l-02c8399d
scope: repo
status: active
touches:
  - docs/reports/e2e/2026-10-23-e2e-report.md
  - docs/reports/e2e/2026-10-23-results.json
  - tests/e2e/client.py
  - tests/e2e/driver.py
  - tests/e2e/report.py
  - tests/e2e/scenarios.py
  - tests/e2e/test_client.py
  - tests/e2e/test_e2e_simulation.py
  - tests/e2e/test_report.py
  - CLAUDE.md
---

## Decided
- tests/e2e/report.py: each executive-summary sentence is emitted only if its scenario's checks all passed (✅/❌ per scenario). Throughput/OPE/block tables are declared descriptive, not criteria.
- The -32603 "Internal error" sanitising is NOT claimed by the e2e report: no stdio probe can reach `except Exception` in daemon.py without modifying src. It is covered by tests/test_jsonrpc_loop.py (monkeypatched health raising RuntimeError). The report cites that test.
- load_history keeps only runs dated before the report date and sorts by recorded meta.date (not file name), so a backfilled report never compares against the future.
- DaemonProcess.close() is idempotent; scenarios use driver.running_daemon() so timeouts/failed starts never leak a process.

## Tried and ruled out
- Convoy `--yolo` launch is denied by the auto-mode classifier; use `convoy --pipeline review-gpt --manual --no-worktree --detach` and approve each benign read-only request with `convoy runs <id> approve <req>`. Its quality gate needs score >= 90 (first run: 88).

## Watch out
- Regenerate the report AFTER committing code (`uv run python scripts/e2e_report.py`) so the commit sha in the header is right.
- e2e is not wired into CI yet (open scope decision).
