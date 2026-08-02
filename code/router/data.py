"""Dataset loading, validation, and strictly time-causal context building."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


MESSAGE_COLUMNS = (
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type", "media_id",
    "forwarded_count",
)
OUTPUT_COLUMNS = (
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
)
TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_time(value):
    value = value.strip()
    if not value:
        return None

    for fmt in (TIME_FORMAT, "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: {value!r}")


def number(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    try:
        return float((row.get(name) or "").strip())
    except ValueError:
        return default


def flag(row: Mapping[str, str], name: str) -> bool:
    return (row.get(name) or "").strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path, required: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [column for column in required if column not in fields]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def index_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


@dataclass(frozen=True)
class Context:
    """Only records that precede the message are exposed to policy/retrieval."""

    target: Dict[str, str]
    user: Optional[Dict[str, str]]
    group: Optional[Dict[str, str]]
    membership: Optional[Dict[str, str]]
    business: Optional[Dict[str, str]]
    business_history: Optional[Dict[str, str]]
    notification_history: List[Dict[str, str]]
    history: List[Dict[str, str]]
    events_by_message: Mapping[str, Dict[str, str]]

    @property
    def created_at(self) -> datetime:
        return parse_time(self.target["created_at"])


class Dataset:
    """Participant-facing data only.  Samples are deliberately excluded."""

    def __init__(self, root: Path):
        self.root = root
        self.messages = read_csv(root / "messages.csv", MESSAGE_COLUMNS)
        self.history = read_csv(root / "message_history.csv", MESSAGE_COLUMNS)
        self.events = read_csv(
            root / "message_events.csv",
            ("user_id", "message_id", "message_opened", "message_replied",
             "reaction_time_minutes", "notification_dismissed", "muted_after_message",
             "message_reported"),
        )
        self.users = read_csv(root / "users.csv", ("user_id", "do_not_disturb_window"))
        self.groups = read_csv(root / "groups.csv", ("group_id", "group_type"))
        self.memberships = read_csv(root / "group_members.csv", ("group_id", "user_id"))
        self.businesses = read_csv(root / "business_accounts.csv", ("business_id", "verified"))
        self.business_history = read_csv(root / "user_business_history.csv", ("user_id", "business_id"))
        self.daily = read_csv(root / "daily_notification_summary.csv", ("user_id", "date"))
        self.images = read_csv(root / "images.csv", ("image_id", "file_path"))
        self.voice_notes = read_csv(root / "voice_notes.csv", ("voice_note_id", "file_path"))

        self.user_by_id = index_by(self.users, "user_id")
        self.group_by_id = index_by(self.groups, "group_id")
        self.business_by_id = index_by(self.businesses, "business_id")
        self.event_by_message = index_by(self.events, "message_id")
        self.image_by_id = index_by(self.images, "image_id")
        self.voice_by_id = index_by(self.voice_notes, "voice_note_id")
        self.membership_by_key = {
            (row["group_id"], row["user_id"]): row for row in self.memberships
        }
        self.business_history_by_key = {
            (row["user_id"], row["business_id"]): row for row in self.business_history
        }
        self._validate_references()

    def _validate_references(self) -> None:
        ids = [row["message_id"] for row in self.messages]
        if len(ids) != len(set(ids)):
            raise ValueError("messages.csv contains duplicate message_id values")
        for row in self.messages + self.history:
            if row["user_id"] not in self.user_by_id:
                raise ValueError(f"Unknown user_id: {row['user_id']}")
            if row.get("group_id") and row["group_id"] not in self.group_by_id:
                raise ValueError(f"Unknown group_id: {row['group_id']}")
            if row.get("business_id") and row["business_id"] not in self.business_by_id:
                raise ValueError(f"Unknown business_id: {row['business_id']}")
            if row.get("media_type") == "image" and row.get("media_id") not in self.image_by_id:
                raise ValueError(f"Unknown image media_id: {row.get('media_id')}")
            if row.get("media_type") == "voice" and row.get("media_id") not in self.voice_by_id:
                raise ValueError(f"Unknown voice media_id: {row.get('media_id')}")

    def media_path(self, message: Mapping[str, str]) -> Optional[Path]:
        if message.get("media_type") == "image":
            row = self.image_by_id.get(message.get("media_id", ""))
        elif message.get("media_type") == "voice":
            row = self.voice_by_id.get(message.get("media_id", ""))
        else:
            row = None
        return self.root / row["file_path"] if row else None

    def context_for(self, target: Dict[str, str]) -> Context:
        timestamp = parse_time(target["created_at"])
        # This filter is the temporal-causality boundary for all downstream code.
        prior = [
            row for row in self.history
            if row["user_id"] == target["user_id"] and parse_time(row["created_at"]) < timestamp
        ]
        daily = [
            row for row in self.daily
            if row["user_id"] == target["user_id"]
            and datetime.strptime(row["date"], "%Y-%m-%d").date() < timestamp.date()
        ]
        membership = self.membership_by_key.get((target.get("group_id", ""), target["user_id"]))
        if membership and membership.get("joined_at") and parse_time(membership["joined_at"]) >= timestamp:
            membership = None
        business_history = self._business_history_as_of(
            self.business_history_by_key.get((target["user_id"], target.get("business_id", ""))),
            timestamp,
        )
        return Context(
            target=target,
            user=self.user_by_id.get(target["user_id"]),
            group=self.group_by_id.get(target.get("group_id", "")),
            membership=membership,
            business=self.business_by_id.get(target.get("business_id", "")),
            business_history=business_history,
            notification_history=daily,
            history=prior,
            events_by_message=self.event_by_message,
        )

    @staticmethod
    def _business_history_as_of(
        row: Optional[Dict[str, str]], timestamp: datetime,
    ) -> Optional[Dict[str, str]]:
        """Return a copy with future-dated state removed.

        The business-history table is a snapshot. Date-stamped changes are only
        usable when they happened before the message. If the only relationship
        activity is future-dated, volatile aggregate counters are also cleared.
        """
        if not row:
            return None
        result = dict(row)
        for field in ("promotions_opted_out_at", "last_reply_at"):
            value = result.get(field, "")
            if value and parse_time(value) >= timestamp:
                result[field] = ""
        last_activity = result.get("last_activity_at", "")
        if last_activity and parse_time(last_activity) >= timestamp:
            result["last_activity_at"] = ""
            for field in (
                "activity_count_180d", "messages_opened_30d",
                "messages_dismissed_30d", "messages_replied_30d",
            ):
                # The snapshot cannot reconstruct the earlier aggregate. Blank
                # means unknown/unavailable, not a fabricated zero-behavior signal.
                result[field] = ""
        return result


def is_quiet_hour(window: str, timestamp: datetime) -> bool:
    """Return whether timestamp is inside a user DND interval, including overnight."""
    try:
        start_s, end_s = window.split("-", 1)
        start = time.fromisoformat(start_s)
        end = time.fromisoformat(end_s)
    except (ValueError, AttributeError):
        return False
    current = timestamp.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end
