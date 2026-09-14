"""Quota ledger, windowing calculations, and shadow pricing barrier.

Complies with contract §3.4, §4.2, FR-14..FR-17, ADR-0005, ADR-0006.
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from azuma_router.config import ArmConfig, Config
from azuma_router.errors import LedgerEventDuplicateError, LedgerEventNotFoundError


@dataclass
class LedgerEvent:
    """Single consumption entry in the ledger."""

    event_id: int
    arm_id: str
    units: float
    ts: float


def log_barrier(u: float | np.ndarray, mu: float | np.ndarray) -> float | np.ndarray:
    """Logarithmic convex barrier shadow price penalty (ADR-0006)."""
    u_clip = np.clip(u, 0.0, 1.0 - 1e-6)
    return -mu * np.log(1.0 - u_clip)


def calc_est_units(
    arm: ArmConfig,
    input_tokens: int,
    expected_output_tokens: int | None,
) -> float:
    """Estimates consumption units for a request on a given arm."""
    if arm.billing_mode != "subscription":
        return 0.0
    out_toks = expected_output_tokens if expected_output_tokens is not None else 0
    tot_toks = input_tokens + out_toks
    return arm.units_per_request + (arm.units_per_1k_tokens * float(tot_toks) / 1000.0)


def _get_period_seconds(period: str | None) -> float:
    """Returns period duration in seconds for daily or weekly reset windows."""
    return 86400.0 if period == "daily" else 604800.0


class QuotaLedger:
    """In-memory exact ledger tracking per-arm consumption across rolling & fixed windows.

    Complies with ADR-0005.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._next_event_id: int = 1
        self._event_ids: set[int] = set()
        self.events: dict[str, deque[LedgerEvent]] = {arm.arm_id: deque() for arm in config.arms}

    def next_event_id(self) -> int:
        """Allocates the id for the next event; the caller journals it before calling record."""
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    def has_event(self, event_id: int) -> bool:
        """Returns True if event_id is present in active (unpruned) ledger events."""
        return event_id in self._event_ids

    def record(self, event_id: int, arm_id: str, units: float, ts: float) -> None:
        """Records an estimated consumption event under an id minted by next_event_id or a WAL."""
        if event_id in self._event_ids:
            raise LedgerEventDuplicateError(event_id, arm_id)
        self._event_ids.add(event_id)
        self._next_event_id = max(self._next_event_id, event_id + 1)
        self.events.setdefault(arm_id, deque()).append(
            LedgerEvent(event_id=event_id, arm_id=arm_id, units=units, ts=ts)
        )

    def correct(self, event_id: int, arm_id: str, actual_units: float) -> None:
        """Corrects an existing ledger event in-place with actual observed units (FR-16)."""
        if arm_id in self.events:
            for ev in self.events[arm_id]:
                if ev.event_id == event_id:
                    ev.units = actual_units
                    return
        raise LedgerEventNotFoundError(event_id, arm_id)

    def _prune(self, arm: ArmConfig, arm_events: deque[LedgerEvent], now: float) -> None:
        """Drops events no window nor a live pending can still reference (P3-MF-1). Runs for
        every arm: per-token arms have no windows but record an event per select (P5-MF-4)."""
        retention_s = self._retention_s(arm)
        if retention_s <= 0:
            return
        cutoff = now - retention_s
        while arm_events and arm_events[0].ts < cutoff:
            self._event_ids.discard(arm_events.popleft().event_id)

    def _retention_s(self, arm: ArmConfig) -> float:
        """Events stay correctable for the whole pending TTL, even once outside every window."""
        return max(self._get_max_window_duration_s(arm), float(self.config.bandit.pending_ttl_s))

    def _get_max_window_duration_s(self, arm: ArmConfig) -> float:
        """Calculates max history window duration in seconds needed for pruning."""
        max_dur = 0.0
        for w in arm.windows:
            if w.kind == "rolling" and w.duration_s is not None:
                max_dur = max(max_dur, float(w.duration_s))
            elif w.kind == "fixed_reset":
                # Max duration for daily is 86400, weekly is 604800
                period_s = _get_period_seconds(w.period)
                max_dur = max(max_dur, period_s * 2)
        return max_dur

    def utilization(self, now: float) -> np.ndarray:
        """Computes float32[k_max, w_max] utilization view at current timestamp (FR-14)."""
        k_max = self.config.bandit.k_max
        w_max = self.config.bandit.w_max
        util = np.zeros((k_max, w_max), dtype=np.float32)

        for i, arm in enumerate(self.config.arms):
            arm_events = self.events.get(arm.arm_id, deque())
            self._prune(arm, arm_events, now)
            if arm.billing_mode != "subscription" or not arm.windows:
                continue

            for j, w in enumerate(arm.windows):
                if j >= w_max:
                    break

                if w.kind == "rolling":
                    dur = float(w.duration_s or 0)
                    cutoff = now - dur
                    consumed = sum(ev.units for ev in arm_events if ev.ts >= cutoff)
                elif w.kind == "fixed_reset":
                    period_s = _get_period_seconds(w.period)
                    anchor_str = (w.reset_anchor or "").replace("Z", "+00:00")
                    anchor_dt = datetime.fromisoformat(anchor_str)
                    anchor_ts = anchor_dt.timestamp()

                    if now >= anchor_ts:
                        periods_elapsed = math.floor((now - anchor_ts) / period_s)
                        last_reset_ts = anchor_ts + float(periods_elapsed) * period_s
                    else:
                        last_reset_ts = anchor_ts

                    consumed = sum(ev.units for ev in arm_events if ev.ts >= last_reset_ts)
                else:
                    consumed = 0.0

                util[i, j] = np.float32(consumed / w.capacity)

        return util

    def dump(self) -> dict[str, Any]:
        """Serializes current active ledger state for snapshot metadata (FR-27)."""
        serialized: dict[str, Any] = {}
        for arm_id, ev_queue in self.events.items():
            serialized[arm_id] = [
                {
                    "event_id": ev.event_id,
                    "arm_id": ev.arm_id,
                    "units": ev.units,
                    "ts": ev.ts,
                }
                for ev in ev_queue
            ]
        return {"next_event_id": self._next_event_id, "events": serialized}

    @classmethod
    def load(cls, data: dict[str, Any], config: Config) -> "QuotaLedger":
        """Restores QuotaLedger state from snapshot dump data."""
        ledger = cls(config)
        for ev_list in data["events"].values():
            for item in ev_list:
                ledger.record(
                    int(item["event_id"]), item["arm_id"], float(item["units"]), float(item["ts"])
                )
        ledger._next_event_id = max(ledger._next_event_id, int(data["next_event_id"]))
        return ledger
