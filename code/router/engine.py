"""End-to-end router orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .confidence import calibrate
from .data import Context, Dataset
from .llm import OptionalReasoner
from .media import MediaExtractor
from .policy import PolicyDecision, combined_text, decide, should_defer_for_dnd
from .retrieval import Evidence, retrieve
from .safety import assess


class Router:
    def __init__(self, dataset_dir: Path, cache_dir: Optional[Path] = None):
        self.dataset = Dataset(dataset_dir)
        self.media = MediaExtractor(cache_dir or Path(tempfile.gettempdir()) / "hackerrank_router_media")
        self.reasoner = OptionalReasoner()

    def predict(self, target: Mapping[str, str]) -> Dict[str, str]:
        context = self.dataset.context_for(dict(target))
        media = self.media.describe(context.target, self.dataset.media_path(context.target))
        target_media_text = " ".join(str(media.get(field, "")) for field in ("text", "summary"))
        safety = assess(context.target, context.business, target_media_text)
        decision = decide(context, media, safety)
        if should_defer_for_dnd(context, decision):
            decision = PolicyDecision("digest", decision.message_type, "The update can wait until the user's quiet hours end.", decision.strong_rule, decision.ambiguous)
        history_media_text = {
            row["message_id"]: self.media.cached_text(self.dataset.media_path(row))
            for row in context.history if row.get("media_type")
        }
        evidence = retrieve(context, decision.action, target_media_text=target_media_text, history_media_text=history_media_text)
        decision = self._optionally_refine(
            context, decision, evidence, media,
            safety.action is not None or safety.injection_detected,
        )
        if should_defer_for_dnd(context, decision):
            decision = PolicyDecision(
                "digest", decision.message_type,
                "The update can wait until the user's quiet hours end.",
                decision.strong_rule, decision.ambiguous, decision.model_confidence,
            )
        # A refined action can require differently useful historical evidence.
        evidence = retrieve(context, decision.action, target_media_text=target_media_text, history_media_text=history_media_text)
        confidence = calibrate(
            decision.action,
            strong_rule=decision.strong_rule,
            evidence=evidence,
            media_quality=str(media.get("quality", "not_applicable")),
            ambiguous=decision.ambiguous,
            model_confidence=decision.model_confidence,
        )
        return {
            "message_id": context.target["message_id"],
            "action": decision.action,
            "message_type": decision.message_type,
            "reason": " ".join(decision.reason.split())[:180],
            "confidence": f"{confidence:.2f}",
            "evidence_message_ids": ";".join(item.message["message_id"] for item in evidence) if evidence else "none",
        }

    def predict_all(self) -> List[Dict[str, str]]:
        return [self.predict(message) for message in self.dataset.messages]

    def _optionally_refine(
        self,
        context: Context,
        decision: PolicyDecision,
        evidence: List[Evidence],
        media: Mapping[str, object],
        safety_locked: bool,
    ) -> PolicyDecision:
        # Clear safety decisions never invoke, and can never be overridden by, an LLM.
        if safety_locked or decision.strong_rule or not self.reasoner.available:
            return decision
        needs_reasoning = decision.ambiguous or (
            context.target.get("media_type") in {"image", "voice"}
            and media.get("source") in {"remote", "openai_direct"}
        )
        if not needs_reasoning:
            return decision
        facts = {
            "deterministic_candidate": {"action": decision.action, "message_type": decision.message_type},
            "untrusted_message": {
                "text": context.target.get("message_text", ""),
                "media_summary": media.get("summary", ""),
                "media_text": media.get("text", ""),
            },
            "source_context": {
                "conversation_type": context.target.get("conversation_type"),
                "group_muted": (context.membership or {}).get("group_muted_by_user"),
                "business_verified": (context.business or {}).get("verified"),
                "allows_promotions": (context.business_history or {}).get("allows_promotions"),
                "promotions_opted_out_at": (context.business_history or {}).get("promotions_opted_out_at"),
            },
            "evidence": [
                {
                    "message_id": item.message["message_id"],
                    "text": item.message.get("message_text", ""),
                    "event": item.event,
                    "relevance": round(item.relevance, 3),
                    "behavior_support": round(item.support, 3),
                }
                for item in evidence
            ],
        }
        refined = self.reasoner.refine(facts)
        if not refined:
            return decision
        # Validate against local hard constraints before accepting ambiguous refinement.
        if (context.membership or {}).get("group_muted_by_user") == "1" and refined["action"] == "notify":
            return decision
        return PolicyDecision(
            str(refined["action"]), str(refined["message_type"]), str(refined["reason"]),
            False, False, float(refined["confidence"]),
        )
