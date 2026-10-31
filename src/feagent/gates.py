"""Gate constructors shared by every stage that reports pass/warn/fail/skip."""
from __future__ import annotations

from typing import Any

from .types import Gate


def gate(status: str, gate_id: str, name: str, message: str, value: float | None = None,
         threshold: float | None = None, hint: str = "") -> Gate:
    return Gate(id=gate_id, name=name, status=status, value=value, threshold=threshold,
                message=message, hint=hint)


def passed(gate_id: str, name: str, message: str, value: float | None = None,
           threshold: float | None = None, hint: str = "") -> Gate:
    return gate("pass", gate_id, name, message, value, threshold, hint)


def warned(gate_id: str, name: str, message: str, value: float | None = None,
           threshold: float | None = None, hint: str = "") -> Gate:
    return gate("warn", gate_id, name, message, value, threshold, hint)


def failed(gate_id: str, name: str, message: str, value: float | None = None,
           threshold: float | None = None, hint: str = "") -> Gate:
    return gate("fail", gate_id, name, message, value, threshold, hint)


def skipped(gate_id: str, name: str, message: str, value: float | None = None,
            threshold: float | None = None, hint: str = "") -> Gate:
    return gate("skip", gate_id, name, message, value, threshold, hint)


def threshold_gate(gate_id: str, name: str, value: float, limit: float, *, message_fmt: str,
                   hint: str, worse: str = "above", fail_status: str = "fail",
                   **fmt: Any) -> Gate:
    """Compare ``value`` with ``limit``; ``worse`` says which side is bad.

    ``message_fmt`` is formatted with ``value``, ``limit`` and ``**fmt``. The hint is
    only attached when the gate does not pass; the stored threshold is ``float(limit)``.
    """
    bad = value > limit if worse == "above" else value < limit
    message = message_fmt.format(value=value, limit=limit, **fmt)
    return gate(fail_status if bad else "pass", gate_id, name, message, value, float(limit),
                hint if bad else "")
