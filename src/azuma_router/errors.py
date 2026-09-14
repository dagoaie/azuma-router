"""Domain exceptions.

Single module by project convention; every error derives from ``AzumaError``.
"""

from typing import Any


class AzumaError(Exception):
    """Base class for every azuma-router error."""


class ConfigError(AzumaError):
    """``azuma.toml`` is invalid. ``field`` is the dotted path of the offending entry (ADR-0017)."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


class InvalidContextError(AzumaError):
    """The featurizer refused a routing request; never silently coerced (ADR-0004/0015)."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


class IncompatibleSnapshotError(AzumaError):
    """Snapshot shapes/ids differ from the running config; no migration path exists (ADR-0009)."""


class JournalWriteError(AzumaError):
    """Fatal failure writing or flushing to the decision journal (WAL)."""


class SnapshotWriteError(AzumaError):
    """Fatal failure publishing a snapshot; the last good snapshot stays in place (ADR-0009)."""


class NumericalIntegrityError(AzumaError):
    """Non-finite score or numerical overflow detected (MF-4)."""


class LedgerEventNotFoundError(AzumaError):
    """Target ledger event_id was not found in QuotaLedger (FR-16 / MF-3)."""

    def __init__(self, event_id: int, arm_id: str) -> None:
        super().__init__(f"Ledger event {event_id} not found for arm {arm_id}")
        self.event_id = event_id
        self.arm_id = arm_id


class LedgerEventDuplicateError(AzumaError):
    """A ledger event_id was recorded twice; ids are the WAL replay idempotency key (FR-23)."""

    def __init__(self, event_id: int, arm_id: str) -> None:
        super().__init__(f"Ledger event {event_id} already recorded for arm {arm_id}")
        self.event_id = event_id
        self.arm_id = arm_id


class InvalidRequestLineError(AzumaError):
    """A stdin line is not a valid JSON-RPC 2.0 request; ``response`` is the error to write."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(str(response["error"]["message"]))
        self.response = response


class RpcError(AzumaError):
    """A JSON-RPC 2.0 application error (codes per ADR-0014)."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
