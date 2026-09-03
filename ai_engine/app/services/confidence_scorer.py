"""
Confidence Scorer.

Combines the intent recognizer's confidence with the LLM's tool-selection
confidence into ONE final score, and classifies it into an action:
  - "execute": confident enough to run the tool immediately
  - "confirm": run write/mutating tools only after explicit user confirmation
  - "clarify": too uncertain, ask the user a clarifying question instead

Kept as its own small service (rather than inlined in the orchestrator) so
the thresholds/policy can be tuned or A/B tested independently.
"""
from __future__ import annotations

from dataclasses import dataclass

WRITE_TOOL_PREFIXES = ("create_", "cancel_", "delete_", "update_", "send_")


@dataclass
class ConfidenceDecision:
    final_confidence: float
    action: str  # "execute" | "confirm" | "clarify"


class WeightedConfidenceScorer:
    def __init__(
        self,
        low_confidence_threshold: float = 0.35,
        high_risk_confidence_threshold: float = 0.75,
        intent_weight: float = 0.4,
        tool_weight: float = 0.6,
    ):
        self._low = low_confidence_threshold
        self._high_risk = high_risk_confidence_threshold
        self._intent_weight = intent_weight
        self._tool_weight = tool_weight

    def score(self, intent_confidence: float, tool_confidence: float, tool_name: str) -> ConfidenceDecision:
        final = round(
            intent_confidence * self._intent_weight + tool_confidence * self._tool_weight, 4
        )

        if final < self._low:
            return ConfidenceDecision(final_confidence=final, action="clarify")

        is_write_tool = any(tool_name.startswith(p) for p in WRITE_TOOL_PREFIXES)
        if is_write_tool and final < self._high_risk:
            return ConfidenceDecision(final_confidence=final, action="confirm")

        return ConfidenceDecision(final_confidence=final, action="execute")
