"""Time-causal retrieval and behavior-aware evidence ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from .data import Context, parse_time


TOKEN = re.compile(r"[a-z0-9]{2,}")


def tokens(text: str) -> Set[str]:
    return set(TOKEN.findall((text or "").lower()))


@dataclass(frozen=True)
class Evidence:
    message: Mapping[str, str]
    event: Mapping[str, str]
    relevance: float
    support: float


def behavior_support(event: Mapping[str, str], action: str) -> float:
    opened = event.get("message_opened") == "1"
    replied = event.get("message_replied") == "1"
    dismissed = event.get("notification_dismissed") == "1" or event.get("muted_after_message") == "1"
    reported = event.get("message_reported") == "1"
    reaction = float(event.get("reaction_time_minutes") or 9999)
    if action == "mute":
        return 1.0 if reported else 0.9 if dismissed else 0.0
    if action == "notify":
        return 1.0 if replied and reaction <= 15 else 0.7 if opened and reaction <= 15 else 0.0
    return 0.85 if opened and not replied and reaction >= 60 else 0.55 if opened else 0.0


def retrieve(
    context: Context,
    action: str,
    limit: int = 3,
    target_media_text: str = "",
    history_media_text: Mapping[str, str] | None = None,
) -> List[Evidence]:
    target = context.target
    target_tokens = tokens(" ".join((target.get("message_text", ""), target_media_text)))
    history_media_text = history_media_text or {}
    target_time = context.created_at
    candidates: List[Evidence] = []
    for row in context.history:
        # Defend the invariant even if a Context is constructed outside Dataset.
        if parse_time(row["created_at"]) >= target_time:
            continue
        event = context.events_by_message.get(row["message_id"], {})
        row_tokens = tokens(" ".join((row.get("message_text", ""), history_media_text.get(row["message_id"], ""))))
        overlap = len(target_tokens & row_tokens) / max(1, len(target_tokens | row_tokens))
        entity = 0.0
        entity += 0.35 if target.get("group_id") and target.get("group_id") == row.get("group_id") else 0.0
        entity += 0.35 if target.get("business_id") and target.get("business_id") == row.get("business_id") else 0.0
        entity += 0.20 if target.get("sender_user_id") and target.get("sender_user_id") == row.get("sender_user_id") else 0.0
        entity += 0.10 if target.get("conversation_type") == row.get("conversation_type") else 0.0
        media = 0.10 if target.get("media_id") and target.get("media_id") == row.get("media_id") else 0.0
        days = max(0.0, (target_time - parse_time(row["created_at"])).days)
        recency = max(0.0, 0.15 * (1.0 - min(days, 180.0) / 180.0))
        relevance = overlap * 0.75 + entity + media + recency
        support = behavior_support(event, action)
        if relevance >= 0.30 and (support > 0 or overlap >= 0.60):
            candidates.append(Evidence(row, event, relevance, support))
    candidates.sort(key=lambda item: (item.relevance * 0.65 + item.support * 0.35, item.message["created_at"]), reverse=True)
    return candidates[:limit]
