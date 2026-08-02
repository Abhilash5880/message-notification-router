"""Deterministic type/action policy.  It consumes untrusted content as data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from .data import Context, flag, is_quiet_hour, number
from .safety import SafetyResult


PROMOTION = re.compile(r"\b(sale|offer|discount|cashback|coupon|promo|deal|unsubscribe|reply stop|buy|selling)\b", re.I)
MARKETPLACE_LISTING = re.compile(
    r"\b(photos?|price|size|condition|pickup|delivery|dm|message me)\b.*\b(pickup|weekend|today|interested|available|price|photos?)\b",
    re.I,
)
GREETING = re.compile(r"\b(good morning|good evening|good vibes|blessings|happy \w+)\b", re.I)
FORWARD = re.compile(r"\b(fwd|forward(?:ed|ing)?|share (?:this|in))\b", re.I)
URGENT = re.compile(r"\b(now|today|tonight|within|before \d|deadline|urgent|asap|eod|leav(?:e|ing) early|closes?|alert threshold)\b", re.I)
DIRECT = re.compile(r"\b(can you|please|pls|reply|call me|need (?:you|quick)|your child)\b", re.I)
EVENT = re.compile(r"\b(school|bus|meeting|appointment|field trip|circular|society|water|fire alarm|portal|event|cultural night)\b", re.I)
CRITICAL_URGENT = re.compile(r"\b(water|power|gas|medical emergency|missing person)\b", re.I)
PAYMENT = re.compile(r"\b(payment|invoice|bill|refund|pickup code|transaction|wallet)\b", re.I)
REQUIRES_ACTION = re.compile(r"\b(consent|sign(?:ed)?|submit|rsvp|registration)\b", re.I)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    message_type: str
    reason: str
    strong_rule: bool
    ambiguous: bool
    model_confidence: Optional[float] = None


def combined_text(context: Context, media: Mapping[str, object]) -> str:
    # The media provider output remains ordinary data; it is never executed or
    # interpolated into routing instructions.
    return " ".join(
        part for part in [
            context.target.get("message_text", ""),
            str(media.get("text", "")),
            str(media.get("summary", "")),
        ] if part
    )


def classify_type(context: Context, text: str, safety: SafetyResult) -> str:
    if safety.message_type:
        return safety.message_type
    target = context.target
    if PROMOTION.search(text) or MARKETPLACE_LISTING.search(text):
        return "promotion"
    if GREETING.search(text):
        return "greeting"
    if FORWARD.search(text) or int(number(target, "forwarded_count")) >= 6:
        return "forward"
    if URGENT.search(text) and CRITICAL_URGENT.search(text):
        return "urgent"
    if target.get("conversation_type") == "group" and EVENT.search(text) and ("@" + target["user_id"] not in text):
        return "event"
    if URGENT.search(text) and (DIRECT.search(text) or EVENT.search(text)):
        return "urgent"
    if EVENT.search(text):
        return "event"
    if re.search(r"\b(safety advisory|never ask)\b", text, re.I):
        return "business_update"
    if PAYOUT_OR_UPDATE(context, text):
        return "payment" if PAYMENT.search(text) else "business_update"
    if target.get("conversation_type") == "personal":
        known_sender = any(row.get("sender_user_id") == target.get("sender_user_id") for row in context.history)
        return "personal" if known_sender else "unknown"
    if target.get("conversation_type") == "business":
        return "business_update"
    return "personal"


def PAYOUT_OR_UPDATE(context: Context, text: str) -> bool:
    return bool(context.target.get("business_id")) and bool(
        PAYMENT.search(text) or re.search(r"\b(order|delivery|packed|booking|status|account|safety advisory)\b", text, re.I)
    )


def decide(context: Context, media: Mapping[str, object], safety: SafetyResult) -> PolicyDecision:
    if safety.action:
        return PolicyDecision(safety.action, safety.message_type or "scam", safety.reason, True, False)
    text = combined_text(context, media)
    message_type = classify_type(context, text, safety)
    membership = context.membership or {}
    business_history = context.business_history or {}
    business = context.business or {}
    muted_group = flag(membership, "group_muted_by_user")
    opted_out = bool(business_history.get("promotions_opted_out_at")) or not flag(business_history, "allows_promotions") and message_type == "promotion" and bool(business_history)
    high_dismissals = (
        number(business_history, "messages_dismissed_30d") >= 5
        or number(membership, "notifications_dismissed_30d") >= 6
        or number(context.user or {}, "notifications_dismissed_30d") >= 60
    )
    prior_similar_dismissals = sum(
        1
        for row in context.history
        if (
            (context.target.get("business_id") and row.get("business_id") == context.target.get("business_id"))
            or (not context.target.get("business_id") and row.get("sender_user_id") == context.target.get("sender_user_id"))
        )
        and context.events_by_message.get(row.get("message_id", ""), {}).get("notification_dismissed") == "1"
    )
    direct_urgent = bool(URGENT.search(text) and (DIRECT.search(text) or "@" + context.target["user_id"] in text))
    operational_urgent = message_type in {"urgent", "event"} and bool(URGENT.search(text))
    domain_matches = bool(business.get("official_domain")) and business.get("official_domain") == business.get("domain_used_by_sender")
    mature_domain = number(business, "domain_used_by_sender_age_days") >= 90
    trusted_business = business.get("verified") == "1" and domain_matches and mature_domain
    recent_relationship = bool(business_history) and number(business_history, "activity_count_180d") > 0

    if message_type in {"promotion", "forward", "greeting"}:
        behavior_rejects = prior_similar_dismissals >= (2 if message_type == "promotion" else 1)
        if opted_out or high_dismissals or behavior_rejects or (muted_group and not direct_urgent):
            return PolicyDecision("mute", message_type, "Similar messages were ignored or dismissed by this user.", False, False)
        return PolicyDecision("digest", message_type, "The message is useful but does not need immediate attention.", False, False)
    if muted_group and not direct_urgent and not operational_urgent:
        return PolicyDecision("mute", message_type, "The group is muted and this message has no urgent direct request.", False, False)
    if direct_urgent or operational_urgent:
        output_type = message_type if message_type == "event" else "urgent" if direct_urgent else message_type
        return PolicyDecision("notify", output_type, "The message needs a timely response or action.", False, False)
    if message_type == "event" and REQUIRES_ACTION.search(text):
        return PolicyDecision("notify", "event", "The event update requires the user's action.", False, False)
    direct_request = bool(DIRECT.search(text)) and not re.search(r"\b(no rush|no hurry|nothing urgent|no need to reply)\b", text, re.I)
    if direct_request and context.target.get("conversation_type") != "business":
        return PolicyDecision("notify", "personal", "The sender directly asks this user for a response or action.", False, False)
    if context.target.get("conversation_type") == "business":
        healthcare = "health" in (business.get("category", "") + " " + text).lower()
        if trusted_business and recent_relationship and (message_type in {"payment", "business_update", "event"}) and (URGENT.search(text) or healthcare):
            return PolicyDecision("notify", message_type, "A trusted business sent a time-sensitive account update.", False, False)
        if business and business.get("verified") != "1" and number(business, "user_reports_30d") >= 12:
            return PolicyDecision("mute", "spam", "The sender has elevated spam or trust-risk signals.", True, False)
        return PolicyDecision("digest", message_type, "The business message is legitimate but not immediately actionable.", False, message_type == "business_update" and not bool(business))
    if message_type == "unknown":
        return PolicyDecision("digest", "unknown", "The sender is unfamiliar, but the message shows no clear safety risk.", False, True)
    if context.target.get("conversation_type") == "personal" and DIRECT.search(text) and URGENT.search(text):
        return PolicyDecision("notify", "urgent", "The sender directly requests time-sensitive help.", False, False)
    return PolicyDecision("digest", message_type, "The message is safe but can be read later.", False, message_type == "personal")


def is_critical_dnd_bypass(context: Context, decision: PolicyDecision) -> bool:
    """Only direct short-deadline or safety-critical messages bypass DND."""
    text = context.target.get("message_text", "")
    direct_mention = "@" + context.target["user_id"] in text or bool(DIRECT.search(text))
    short_deadline = bool(re.search(r"\b(now|within \d+|in \d+ minutes?|before \d{1,2}:?\d*)\b", text, re.I))
    critical = bool(CRITICAL_URGENT.search(text) and URGENT.search(text))
    return decision.action == "notify" and (critical or (direct_mention and short_deadline))


def should_defer_for_dnd(context: Context, decision: PolicyDecision) -> bool:
    return (
        decision.action == "notify"
        and context.user is not None
        and is_quiet_hour(context.user.get("do_not_disturb_window", ""), context.created_at)
        and not is_critical_dnd_bypass(context, decision)
    )
