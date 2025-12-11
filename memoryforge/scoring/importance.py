"""Importance scoring algorithms for memory entries.

Implements multiple scoring strategies:
- Rule-based: Fast, deterministic scoring using keywords and patterns
- LLM-based: Accurate but slower scoring using language models
- Hybrid: Combines both approaches for optimal balance
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()


@dataclass
class ScoringContext:
    """Context for scoring a piece of content."""

    content: str
    role: str = "unknown"
    conversation_history: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScoringResult:
    """Result of importance scoring."""

    score: float
    confidence: float
    reasoning: str
    factors: dict


class ImportanceScorer(ABC):
    """Abstract base class for importance scorers."""

    @abstractmethod
    async def score(self, context: ScoringContext) -> ScoringResult:
        """Score the importance of content."""
        pass


class RuleBasedScorer(ImportanceScorer):
    """Fast rule-based importance scorer.

    Uses keyword matching and pattern detection for quick scoring.
    Suitable for real-time applications where latency is critical.
    """

    def __init__(self):
        # High importance keywords (decisions, requirements, specs)
        self._high_keywords = {
            # Decisions
            "decided", "decision", "chose", "selected", "picked",
            "will use", "going with", "agreed", "confirmed",
            # Requirements
            "must", "required", "need to", "have to", "should",
            "requirement", "spec", "specification",
            # Technical specs
            "api", "database", "schema", "architecture", "design",
            "endpoint", "authentication", "authorization",
            # Critical
            "important", "critical", "essential", "key", "priority",
            "deadline", "milestone", "blocker",
        }

        # Medium importance keywords (context, clarifications)
        self._medium_keywords = {
            "because", "reason", "since", "therefore",
            "explained", "clarified", "means", "implies",
            "option", "alternative", "approach", "method",
            "suggest", "recommend", "consider", "maybe",
            "feature", "functionality", "capability",
        }

        # Low importance keywords (casual, acknowledgments)
        self._low_keywords = {
            "okay", "ok", "sure", "yes", "no", "thanks",
            "got it", "understood", "i see", "alright",
            "hello", "hi", "hey", "bye", "goodbye",
        }

        # Patterns for code-related content
        self._code_patterns = [
            r"```[\s\S]*?```",  # Code blocks
            r"`[^`]+`",  # Inline code
            r"def \w+\(",  # Python function
            r"class \w+",  # Class definition
            r"import \w+",  # Import statement
            r"\w+\(\)",  # Function calls
        ]

    async def score(self, context: ScoringContext) -> ScoringResult:
        """Score content using rules."""
        content_lower = context.content.lower()
        factors = {}

        # Base score
        base_score = 0.5

        # Check high importance keywords
        high_matches = sum(1 for kw in self._high_keywords if kw in content_lower)
        high_boost = min(0.3, high_matches * 0.05)
        factors["high_keywords"] = high_matches

        # Check medium importance keywords
        medium_matches = sum(1 for kw in self._medium_keywords if kw in content_lower)
        medium_boost = min(0.15, medium_matches * 0.03)
        factors["medium_keywords"] = medium_matches

        # Check low importance keywords
        low_matches = sum(1 for kw in self._low_keywords if kw in content_lower)
        low_penalty = min(0.2, low_matches * 0.05)
        factors["low_keywords"] = low_matches

        # Check for code content
        code_matches = sum(
            1 for pattern in self._code_patterns if re.search(pattern, context.content)
        )
        code_boost = min(0.2, code_matches * 0.05)
        factors["code_patterns"] = code_matches

        # Length factor (very short or very long content)
        content_length = len(context.content)
        length_penalty = 0
        length_boost = 0
        if content_length < 20:
            length_penalty = 0.1
        elif content_length > 500:
            length_boost = 0.1
        factors["content_length"] = content_length

        # Role-based adjustment
        role_boost = 0
        if context.role == "user":
            role_boost = 0.05  # User messages often contain requirements
        elif context.role == "assistant":
            if any(kw in content_lower for kw in ["here's", "here is", "solution", "implementation"]):
                role_boost = 0.1
        factors["role"] = context.role

        # Question detection
        is_question = "?" in context.content
        question_boost = 0.05 if is_question else 0
        factors["is_question"] = is_question

        # Calculate final score
        score = base_score + high_boost + medium_boost - low_penalty + code_boost
        score = score + role_boost + question_boost - length_penalty + length_boost
        score = max(0.0, min(1.0, score))

        # Confidence based on number of factors matched
        total_matches = high_matches + medium_matches + low_matches + code_matches
        confidence = min(1.0, 0.5 + total_matches * 0.05)

        return ScoringResult(
            score=round(score, 2),
            confidence=round(confidence, 2),
            reasoning=self._generate_reasoning(factors),
            factors=factors,
        )

    def _generate_reasoning(self, factors: dict) -> str:
        """Generate human-readable reasoning."""
        parts = []

        if factors.get("high_keywords", 0) > 0:
            parts.append(f"Found {factors['high_keywords']} high-importance keywords")
        if factors.get("code_patterns", 0) > 0:
            parts.append(f"Contains {factors['code_patterns']} code patterns")
        if factors.get("low_keywords", 0) > 2:
            parts.append("Contains casual/acknowledgment language")

        return "; ".join(parts) if parts else "No significant patterns detected"


class LLMScorer(ImportanceScorer):
    """LLM-based importance scorer.

    Uses a language model to evaluate content importance.
    More accurate but slower and requires API calls.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def score(self, context: ScoringContext) -> ScoringResult:
        """Score content using LLM."""
        prompt = self._build_prompt(context)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()
            return self._parse_response(result_text)

        except Exception as e:
            logger.error("LLM scoring failed", error=str(e))
            return ScoringResult(
                score=0.5,
                confidence=0.3,
                reasoning="LLM scoring failed, using default",
                factors={"error": str(e)},
            )

    def _build_prompt(self, context: ScoringContext) -> str:
        return f"""Rate the importance of this content for future reference in a software development context.

Role: {context.role}
Content: {context.content}

Recent context (if any): {context.conversation_history[:300] if context.conversation_history else "None"}

Evaluate based on:
1. Is this a decision, requirement, or specification?
2. Does it contain technical details worth remembering?
3. Would forgetting this cause problems later?
4. Is it just casual conversation or acknowledgment?

Respond in this exact format:
SCORE: [0.0-1.0]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""

    def _parse_response(self, text: str) -> ScoringResult:
        """Parse LLM response into ScoringResult."""
        lines = text.strip().split("\n")

        score = 0.5
        confidence = 0.7
        reasoning = ""

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return ScoringResult(
            score=max(0.0, min(1.0, score)),
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning or "LLM evaluation",
            factors={"method": "llm"},
        )


class HybridScorer(ImportanceScorer):
    """Hybrid scorer combining rule-based and LLM approaches.

    Uses rule-based scoring for fast initial assessment,
    and LLM scoring for ambiguous cases or when high accuracy is needed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        llm_threshold: float = 0.4,
        confidence_threshold: float = 0.6,
    ):
        self._rule_scorer = RuleBasedScorer()
        self._llm_scorer = LLMScorer(api_key=api_key)
        self._llm_threshold = llm_threshold
        self._confidence_threshold = confidence_threshold

    async def score(self, context: ScoringContext) -> ScoringResult:
        """Score using hybrid approach."""
        # First, get rule-based score
        rule_result = await self._rule_scorer.score(context)

        # Decide if we need LLM evaluation
        needs_llm = (
            rule_result.confidence < self._confidence_threshold
            or abs(rule_result.score - 0.5) < self._llm_threshold
        )

        if not needs_llm:
            return rule_result

        # Get LLM score
        llm_result = await self._llm_scorer.score(context)

        # Combine scores (weighted average based on confidence)
        total_confidence = rule_result.confidence + llm_result.confidence
        if total_confidence == 0:
            combined_score = (rule_result.score + llm_result.score) / 2
        else:
            combined_score = (
                rule_result.score * rule_result.confidence
                + llm_result.score * llm_result.confidence
            ) / total_confidence

        return ScoringResult(
            score=round(combined_score, 2),
            confidence=round(max(rule_result.confidence, llm_result.confidence), 2),
            reasoning=f"Hybrid: {rule_result.reasoning} | LLM: {llm_result.reasoning}",
            factors={
                "rule_score": rule_result.score,
                "llm_score": llm_result.score,
                "method": "hybrid",
            },
        )


async def score_content(
    content: str,
    role: str = "unknown",
    context: str = "",
    method: str = "rule",
    api_key: str | None = None,
) -> float:
    """Convenience function for quick scoring.

    Args:
        content: The content to score
        role: Role of the speaker (user, assistant)
        context: Recent conversation context
        method: Scoring method (rule, llm, hybrid)
        api_key: API key for LLM methods

    Returns:
        Importance score between 0.0 and 1.0
    """
    scoring_context = ScoringContext(
        content=content,
        role=role,
        conversation_history=context,
    )

    if method == "rule":
        scorer = RuleBasedScorer()
    elif method == "llm":
        scorer = LLMScorer(api_key=api_key)
    else:
        scorer = HybridScorer(api_key=api_key)

    result = await scorer.score(scoring_context)
    return result.score
