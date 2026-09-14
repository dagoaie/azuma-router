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


class RpcError(AzumaError):
    """A JSON-RPC 2.0 application error (codes per ADR-0014)."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
