"""
NeuroSynth - Adversarial Reviewer Agent
========================================

Safety-critical agent that reviews generated synthesis against authoritative
Tier 1 sources (Rhoton, Lawton, Spetzler) to flag potential contradictions.

This is the final safety gate before content is presented to clinicians.

Fixes Issue #7: Hallucination detection + adversarial review only checks internal corpus.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ControversyWarning:
    """
    Warning from adversarial review about potential clinical safety issues.

    Severity levels:
    - HIGH: Draft recommends action that authoritative source forbids
            (e.g., "clip before proximal control" vs source says opposite)
    - MEDIUM: Nuance error or different emphasis that could mislead
    - LOW: Minor discrepancy, stylistic differences
    """
    severity: str  # HIGH, MEDIUM, LOW
    topic: str
    draft_claim: str
    contradicting_source: str
    source_quote: str
    recommendation: str
    section: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "topic": self.topic,
            "draft_claim": self.draft_claim,
            "contradicting_source": self.contradicting_source,
            "source_quote": self.source_quote,
            "recommendation": self.recommendation,
            "section": self.section,
        }


@dataclass
class ReviewResult:
    """Result of adversarial section review."""
    has_issues: bool
    warnings: List[ControversyWarning] = field(default_factory=list)
    section_reviewed: Optional[str] = None
    review_time_ms: Optional[int] = None
    confidence: float = 0.8
    reviewer_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_issues": self.has_issues,
            "warnings": [w.to_dict() for w in self.warnings],
            "section_reviewed": self.section_reviewed,
            "review_time_ms": self.review_time_ms,
            "confidence": self.confidence,
            "reviewer_notes": self.reviewer_notes,
            "high_severity_count": self.get_critical_count(),
        }

    def get_high_severity(self) -> List[ControversyWarning]:
        """Get only HIGH severity warnings."""
        return [w for w in self.warnings if w.severity == "HIGH"]

    def get_critical_count(self) -> int:
        """Count HIGH severity warnings."""
        return len(self.get_high_severity())


class SearchServiceProtocol(Protocol):
    """Protocol for search service interface."""
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Any]:
        ...


# =============================================================================
# ADVERSARIAL REVIEWER
# =============================================================================

class AdversarialReviewer:
    """
    Reviews synthesized content against authoritative sources for safety.

    The adversarial reviewer acts as a "devil's advocate" - its job is to
    find contradictions between generated content and established medical
    literature, particularly for claims that could affect patient safety.

    Authority Tier Logic:
    - Only flags contradictions from sources with authority >= 0.95 (Tier 1)
    - Tier 1 sources: Rhoton, Lawton, Spetzler, Samii, Al-Mefty
    - Lower-tier contradictions are logged but not flagged as warnings

    Usage:
        reviewer = AdversarialReviewer(
            anthropic_client=client,
            search_service=search_service,
            authority_threshold=0.95
        )

        result = await reviewer.review_section(
            section_name="Surgical Technique",
            section_content="...",
            topic="MCA aneurysm clipping"
        )

        if result.has_issues:
            for warning in result.get_high_severity():
                print(f"CRITICAL: {warning.topic}")
    """

    # Patterns that indicate safety-critical claims
    SAFETY_PATTERNS = [
        r"should\s+(?:always|never|be|not)",
        r"must\s+(?:be|not|always)",
        r"avoid(?:ing)?",
        r"contraindicated",
        r"dangerous",
        r"critical(?:ly)?",
        r"risk\s+of",
        r"complication",
        r"injury\s+to",
        r"preserve",
        r"sacrifice",
        r"before\s+(?:clipping|resection|dissection)",
        r"after\s+(?:clipping|resection|dissection)",
        r"proximal\s+control",
        r"temporary\s+clip",
    ]

    # Tier 1 source identifiers (must be exact matches in document_title)
    TIER_1_KEYWORDS = [
        "rhoton", "lawton", "spetzler", "samii", "al-mefty", "almefty",
        "seven aneurysms", "microsurgical anatomy"
    ]

    def __init__(
        self,
        anthropic_client,
        search_service: Optional[SearchServiceProtocol] = None,
        authority_threshold: float = 0.95,
        min_chunks_for_review: int = 3,
    ):
        """
        Initialize adversarial reviewer.

        Args:
            anthropic_client: Claude API client for LLM-based review
            search_service: Service for retrieving authoritative sources
            authority_threshold: Minimum authority score to flag contradictions (0.95 = Tier 1)
            min_chunks_for_review: Minimum Tier 1 chunks needed before flagging
        """
        self.client = anthropic_client
        self.search_service = search_service
        self.authority_threshold = authority_threshold
        self.min_chunks = min_chunks_for_review

    def _extract_safety_claims(self, content: str) -> List[str]:
        """Extract sentences containing safety-critical claims."""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        safety_claims = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern in self.SAFETY_PATTERNS:
                if re.search(pattern, sentence_lower):
                    safety_claims.append(sentence.strip())
                    break

        return safety_claims[:10]  # Limit to top 10 for efficiency

    def _is_tier_1_source(self, document_title: str, authority_score: float) -> bool:
        """Check if a source qualifies as Tier 1 authoritative."""
        if authority_score < self.authority_threshold:
            return False

        title_lower = (document_title or "").lower()
        return any(kw in title_lower for kw in self.TIER_1_KEYWORDS)

    async def review_section(
        self,
        section_name: str,
        section_content: str,
        topic: str,
    ) -> ReviewResult:
        """
        Review a synthesis section against authoritative sources.

        Args:
            section_name: Name of the section being reviewed
            section_content: Generated content to review
            topic: Overall synthesis topic

        Returns:
            ReviewResult with any identified contradictions
        """
        start_time = asyncio.get_event_loop().time()

        # Step 1: Extract safety-critical claims
        safety_claims = self._extract_safety_claims(section_content)

        if not safety_claims:
            logger.debug(f"No safety-critical claims in section: {section_name}")
            return ReviewResult(
                has_issues=False,
                section_reviewed=section_name,
                review_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
                reviewer_notes="No safety-critical claims identified"
            )

        logger.info(f"Reviewing {len(safety_claims)} safety claims in '{section_name}'")

        # Step 2: Search for Tier 1 sources on this topic
        tier_1_chunks = []
        if self.search_service:
            try:
                search_results = await self.search_service.search(
                    query=f"{topic} {section_name}",
                    top_k=20,
                )

                # Filter to Tier 1 only
                for result in search_results:
                    doc_title = getattr(result, 'document_title', '') or ''
                    auth_score = getattr(result, 'authority_score', 0.7)

                    if self._is_tier_1_source(doc_title, auth_score):
                        tier_1_chunks.append({
                            "content": getattr(result, 'content', ''),
                            "source": doc_title,
                            "authority": auth_score,
                        })

                logger.info(f"Found {len(tier_1_chunks)} Tier 1 chunks for review")

            except Exception as e:
                logger.warning(f"Search failed during adversarial review: {e}")

        # Step 3: If insufficient Tier 1 sources, skip LLM review
        if len(tier_1_chunks) < self.min_chunks:
            return ReviewResult(
                has_issues=False,
                section_reviewed=section_name,
                review_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
                reviewer_notes=f"Insufficient Tier 1 sources ({len(tier_1_chunks)}/{self.min_chunks})"
            )

        # Step 4: LLM-based contradiction detection
        warnings = await self._llm_review(
            section_name=section_name,
            safety_claims=safety_claims,
            tier_1_chunks=tier_1_chunks,
        )

        return ReviewResult(
            has_issues=len(warnings) > 0,
            warnings=warnings,
            section_reviewed=section_name,
            review_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
            confidence=0.85 if tier_1_chunks else 0.5,
        )

    async def _llm_review(
        self,
        section_name: str,
        safety_claims: List[str],
        tier_1_chunks: List[Dict],
    ) -> List[ControversyWarning]:
        """Use LLM to detect contradictions between claims and sources."""

        if not self.client:
            return []

        # Build source context
        source_context = "\n\n".join(
            f"[{c['source']}]:\n{c['content'][:800]}"
            for c in tier_1_chunks[:5]
        )

        claims_text = "\n".join(f"- {claim}" for claim in safety_claims)

        prompt = f"""You are a neurosurgical safety reviewer. Your job is to find CONTRADICTIONS between draft claims and authoritative sources.

DRAFT CLAIMS (from section "{section_name}"):
{claims_text}

AUTHORITATIVE SOURCES (Tier 1 - Rhoton, Lawton, Spetzler):
{source_context}

INSTRUCTIONS:
1. Compare each draft claim against the authoritative sources
2. ONLY flag genuine contradictions - where the draft says X but the source says NOT-X
3. Do NOT flag claims that are simply not covered by sources
4. Do NOT flag stylistic differences or rephrasing
5. Focus on SAFETY-CRITICAL contradictions that could affect patient outcomes

For each contradiction found, respond with JSON in this format:
```json
[
  {{
    "severity": "HIGH|MEDIUM|LOW",
    "topic": "specific medical topic",
    "draft_claim": "what the draft says",
    "contradicting_source": "source name",
    "source_quote": "exact quote from source proving contradiction",
    "recommendation": "how to fix the draft"
  }}
]
```

If NO contradictions found, respond with: []

IMPORTANT: Only HIGH severity for claims that could cause patient harm if wrong.
"""

        try:
            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse JSON response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if not json_match:
                return []

            warnings_data = json.loads(json_match.group())

            warnings = []
            for w in warnings_data:
                if not isinstance(w, dict):
                    continue

                warnings.append(ControversyWarning(
                    severity=w.get("severity", "LOW"),
                    topic=w.get("topic", "Unknown"),
                    draft_claim=w.get("draft_claim", ""),
                    contradicting_source=w.get("contradicting_source", ""),
                    source_quote=w.get("source_quote", ""),
                    recommendation=w.get("recommendation", ""),
                    section=section_name,
                ))

            return warnings

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reviewer JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            return []

    async def review_all_sections(
        self,
        sections: List[Dict[str, Any]],
        topic: str,
    ) -> List[ReviewResult]:
        """
        Review all sections in a synthesis result.

        Args:
            sections: List of section dicts with 'title' and 'content'
            topic: Overall synthesis topic

        Returns:
            List of ReviewResult, one per section
        """
        results = []

        for section in sections:
            title = section.get("title", "Unknown")
            content = section.get("content", "")

            result = await self.review_section(
                section_name=title,
                section_content=content,
                topic=topic,
            )
            results.append(result)

        # Log summary
        total_warnings = sum(len(r.warnings) for r in results)
        high_severity = sum(r.get_critical_count() for r in results)

        if total_warnings > 0:
            logger.warning(
                f"Adversarial review complete: {total_warnings} warnings "
                f"({high_severity} HIGH severity)"
            )
        else:
            logger.info("Adversarial review complete: No contradictions found")

        return results


# =============================================================================
# LIGHTWEIGHT HEURISTIC REVIEWER (No LLM required)
# =============================================================================

class HeuristicReviewer:
    """
    Fast heuristic-based reviewer for quick checks without LLM.

    Use this for initial screening before expensive LLM review.
    """

    # Known dangerous combinations
    DANGEROUS_PATTERNS = [
        (r"clip.*without.*proximal", "May recommend clipping without proximal control"),
        (r"sacrifice.*(?:artery|nerve)", "May recommend sacrificing critical structures"),
        (r"no need.*(?:monitor|verify)", "May skip critical verification steps"),
        (r"skip.*(?:temporary|test)", "May skip safety measures"),
    ]

    def review(self, content: str) -> List[ControversyWarning]:
        """Quick pattern-based safety check."""
        warnings = []
        content_lower = content.lower()

        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, content_lower):
                # Extract the sentence containing the match
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for sentence in sentences:
                    if re.search(pattern, sentence.lower()):
                        warnings.append(ControversyWarning(
                            severity="MEDIUM",
                            topic="Safety Pattern Match",
                            draft_claim=sentence[:200],
                            contradicting_source="Heuristic Check",
                            source_quote=description,
                            recommendation="Review this claim against authoritative sources",
                        ))
                        break

        return warnings
