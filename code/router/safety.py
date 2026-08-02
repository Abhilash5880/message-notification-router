"""High-precedence local safety and prompt-injection detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional


INJECTION = re.compile(
    r"\b(ignore|disregard|override|bypass)\b.{0,80}\b(previous|prior|system|routing|instruction|rule)s?\b",
    re.IGNORECASE | re.DOTALL,
)
SENSITIVE = re.compile(r"\b(otp|one[- ]?time (?:password|code)|password|pin|cvv|login code|verification code|card details|card number|bank details)\b", re.I)
PRESSURE = re.compile(r"\b(blocked?|suspend(?:ed)?|expire[sd]?|within \d+|now|urgent|immediately|today)\b", re.I)
PAYMENT = re.compile(r"\b(pay|payment|refund|wallet|bank|account|card|token|kyc)\b", re.I)
SUSPICIOUS_LINK = re.compile(r"\b[a-z0-9-]+\.(?:in|com|net|org)\b", re.I)
BENIGN_WARNING = re.compile(r"\b(never|do not|don't)\s+(?:ask|share|give).{0,50}\b(otp|pin|password)\b", re.I)


@dataclass(frozen=True)
class SafetyResult:
    action: Optional[str] = None
    message_type: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0
    injection_detected: bool = False


def assess(
    message: Mapping[str, str],
    business: Optional[Mapping[str, str]],
    extracted_media_text: str = "",
) -> SafetyResult:
    """Assess message plus extracted media as untrusted content.

    OCR/ASR output is deliberately folded into the same local safety pass so a
    harmless caption cannot hide a scam, sensitive request, or prompt injection
    contained only in an image or voice note.
    """
    text = " ".join(part for part in (message.get("message_text", "") or "", extracted_media_text or "") if part)
    injection = bool(INJECTION.search(text))
    sensitive_request = bool(SENSITIVE.search(text)) and not BENIGN_WARNING.search(text)
    pressure = bool(PRESSURE.search(text))
    payment = bool(PAYMENT.search(text))
    official = (business or {}).get("official_domain", "").strip().lower()
    sender_domain = (business or {}).get("domain_used_by_sender", "").strip().lower()
    domain_mismatch = bool(official and sender_domain and official != sender_domain)
    domain_age = int(float((business or {}).get("domain_used_by_sender_age_days") or 0))
    young_domain = bool(sender_domain) and domain_age < 90
    suspicious_domain = bool(SUSPICIOUS_LINK.search(text)) and (not business or domain_mismatch or young_domain)
    credential_flow = bool(
        re.search(
            r"\b(log\s*in|login|sign\s*in|verify|verification|reactivat(?:e|ion)|"
            r"unlock|confirm (?:your )?(?:account|identity)|kyc)\b",
            text,
            re.I,
        )
    )

    fee_pressure = bool(
        re.search(
            r"\b(processing fee|reactivation fee|registration fee|verification fee|"
            r"pay(?:ment)? via|scan (?:the )?qr|send (?:a )?screenshot)\b",
            text,
            re.I,
        )
    )

    shortened_link = bool(
        re.search(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl)/", text, re.I)
    )

    ####################################

    if pressure and credential_flow and (suspicious_domain or shortened_link):
        return SafetyResult(
            "mute",
            "scam",
            "The message uses account pressure with an untrusted verification link.",
            0.90,
            injection,
    )

    if fee_pressure and (suspicious_domain or shortened_link or payment):
        return SafetyResult(
            "mute",
            "scam",
            "The message requests payment through a suspicious or unverified flow.",
            0.90,
            injection,
        )
    if sensitive_request and (pressure or payment or suspicious_domain or domain_mismatch or young_domain):
        return SafetyResult("mute", "scam", "The message asks for sensitive verification through a risky flow.", 0.91, injection)
    if re.search(r"\bpay\s+(?:rs\.?|inr)\s*[\d,]+\b", text, re.I) and pressure and re.search(r"\b(token|registry|papers?|block)\b", text, re.I):
        return SafetyResult("mute", "scam", "The message uses payment pressure through an unverified high-risk offer.", 0.88, injection)
    if injection and sensitive_request:
        return SafetyResult("mute", "scam", "The message tries to control routing while requesting sensitive information.", 0.91, True)
    if business:
        reports = int(float(business.get("user_reports_30d") or 0))
        unverified = business.get("verified") != "1"
        elevated_domain_risk = domain_mismatch or young_domain
        if (unverified or elevated_domain_risk) and reports >= 12 and (payment or pressure or elevated_domain_risk):
            return SafetyResult("mute", "spam", "The sender shows strong spam or trust-risk signals.", 0.87, injection)
    return SafetyResult(injection_detected=injection)
