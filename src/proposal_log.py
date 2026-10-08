"""Append-only log of every Deriv proposal we saw, bought or not.

The backtest must be driven by the proposal prices Deriv actually offered, not by an
estimated fair value times a measured markup. To replay those quotes you first have to
store them: timestamp, symbol, contract type, duration, barrier, spot, ask price,
payout per point and proposal id.

Rejected quotes are logged too. Logging only the contracts we bought would create
selection bias and make it impossible to reconstruct the opportunity set, so a row is
written for every proposal returned by `proposal`, before the premium gate decides.

This file is append-only: one row per call, header written only when the file is new.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

FIELDS = [
    "epoch",
    "symbol",
    "contracttype",
    "duration",
    "durationunit",
    "barrier",
    "spot",
    "askprice",
    "payout",
    "payoutperpoint",
    "proposalid",
]


def append_proposal(path: "str | Path", row: dict[str, Any]) -> None:
    """Append one proposal row, writing the header if the file does not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in FIELDS})
