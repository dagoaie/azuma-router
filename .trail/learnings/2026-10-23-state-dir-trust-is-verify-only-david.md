---
title: State-dir trust is verify-only (owner + no group/other write); never chmod or repair
author: David
date: 2026-10-23
branch: fix/state-dir-trust
id: l-9b4a3f3e
scope: repo
status: active
---

## Decided
- `wal.require_trusted_paths` requires uid == geteuid() and no group/other write bit on every EXISTING state artifact (WAL active+rotated, snapshot dirs/files). Read bits are ignored (0644 passes). Called first in `restore_snapshot` (before rmtree/rename) and in `Router.boot` for the WAL; failure exits 3 as `unsafe_state_permissions`.
- FR-25 ("restore nunca altera permisos") is kept: the check only verifies. The contract gained FR-25b and ADR-0020 records the decision, as the audit learning required.

## Tried and ruled out
- Auto-repair with chmod/chown: hides tampering and edits a directory we may not own.
- Requiring exactly 0700/0600: breaks backups restored as 0644 without adding safety; the risk is write access.
- Checking the WAL parent dir: false positives on shared dirs such as /tmp.

## Watch out
- Symlinks are followed on purpose (we judge what would be read), but WAL files are still opened with O_NOFOLLOW.
- ADR numbers: 0019 is `max_pending` (branch fix/max-pending), 0020 is this one.
