"""Submission row construction and strict output validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data import parse_time

from .data import OUTPUT_COLUMNS
from .llm import ALLOWED_ACTIONS, ALLOWED_TYPES


def validate_rows(
    rows: Sequence[Mapping[str, str]], expected_ids: Iterable[str],
    targets_by_id: Mapping[str, Mapping[str, str]] | None = None,
    history_by_id: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    expected = list(expected_ids)
    if len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} output rows, received {len(rows)}")
    if [row.get("message_id") for row in rows] != expected:
        raise ValueError("Output row order or message_id values do not match messages.csv")
    for row in rows:
        if row.get("action") not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action for {row.get('message_id')}")
        if row.get("message_type") not in ALLOWED_TYPES:
            raise ValueError(f"Invalid message_type for {row.get('message_id')}")
        try:
            confidence = float(row.get("confidence", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid confidence for {row.get('message_id')}") from exc
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence is out of range for {row.get('message_id')}")
        if not row.get("reason") or len(row["reason"]) > 180:
            raise ValueError(f"Reason is missing or too long for {row.get('message_id')}")
        if not row.get("evidence_message_ids"):
            raise ValueError(f"Evidence field is missing for {row.get('message_id')}")
        evidence_ids = row["evidence_message_ids"].split(";")
        if evidence_ids == ["none"]:
            continue
        if "none" in evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"Evidence IDs must be unique and cannot mix with none for {row.get('message_id')}")
        if targets_by_id is not None and history_by_id is not None:
            target = targets_by_id.get(str(row.get("message_id")))
            if target is None:
                raise ValueError(f"Missing target record for {row.get('message_id')}")
            for evidence_id in evidence_ids:
                evidence = history_by_id.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"Unknown evidence ID {evidence_id} for {row.get('message_id')}")
                if evidence.get("user_id") != target.get("user_id"):
                    raise ValueError(f"Cross-recipient evidence {evidence_id} for {row.get('message_id')}")
                if parse_time(evidence["created_at"]) >= parse_time(target["created_at"]):
                    raise ValueError(f"Non-historical evidence {evidence_id} for {row.get('message_id')}")


def write_output(
    path: Path, rows: Sequence[Mapping[str, str]], expected_ids: Iterable[str],
    targets_by_id: Mapping[str, Mapping[str, str]] | None = None,
    history_by_id: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    validate_rows(rows, expected_ids, targets_by_id, history_by_id)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
