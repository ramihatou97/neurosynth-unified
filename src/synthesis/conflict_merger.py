"""
Conflict-Aware Merge System for Internal vs External Data
==========================================================

Intelligently merges internal (corpus) vs external (web) data with
fact-level conflict detection and resolution.

Design Principles:
1. Internal corpus (Rhoton, Youmans, Lawton) = AUTHORITATIVE for established facts
2. External sources = AUTHORITATIVE for recent developments, trials, guidelines
3. Conflicts should be surfaced, not hidden
4. Resolution strategy depends on conflict type
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConflictCategory(Enum):
    """Categories of conflicts between sources."""
    TEMPORAL = "temporal"            # Time-sensitive (guidelines, trials)
    ESTABLISHED_FACT = "established" # Core anatomy/physiology facts
    QUANTITATIVE = "quantitative"    # Numerical disagreements
    APPROACH = "approach"            # Different valid techniques
    RECOMMENDATION = "recommendation" # Different clinical recommendations


class ResolutionStrategy(Enum):
    """How to resolve different conflict types."""
    PREFER_INTERNAL = "prefer_internal"
    PREFER_EXTERNAL = "prefer_external"
    NOTE_BOTH = "note_both"
    FLAG_FOR_REVIEW = "flag_for_review"


# Default resolution by conflict type
DEFAULT_RESOLUTION: Dict[ConflictCategory, ResolutionStrategy] = {
    ConflictCategory.TEMPORAL: ResolutionStrategy.PREFER_EXTERNAL,
    ConflictCategory.ESTABLISHED_FACT: ResolutionStrategy.PREFER_INTERNAL,
    ConflictCategory.QUANTITATIVE: ResolutionStrategy.NOTE_BOTH,
    ConflictCategory.APPROACH: ResolutionStrategy.NOTE_BOTH,
    ConflictCategory.RECOMMENDATION: ResolutionStrategy.FLAG_FOR_REVIEW,
}

# Keywords for classification
TEMPORAL_KEYWORDS = ["2023", "2024", "2025", "recent", "latest", "updated",
                     "current", "guidelines", "trial", "meta-analysis"]
ESTABLISHED_KEYWORDS = ["anatomy", "anatomical", "classical", "standard",
                        "established", "fundamental", "consistently"]


@dataclass
class ExtractedFact:
    """A fact extracted from source material."""
    claim: str
    value: Optional[str] = None
    unit: Optional[str] = None
    source_type: str = "internal"
    is_temporal: bool = False
    is_recommendation: bool = False


@dataclass
class DetectedConflict:
    """A conflict between internal and external sources."""
    category: ConflictCategory
    description: str
    internal_claim: str
    external_claim: str
    resolution_strategy: ResolutionStrategy
    resolved_content: str = ""
    resolution_note: str = ""
    severity: str = "medium"


@dataclass
class MergeResult:
    """Result of conflict-aware merge operation."""
    topic: str
    resolved_content: str
    merge_strategy_used: str
    conflicts: List[DetectedConflict] = field(default_factory=list)
    conflict_count: int = 0
    internal_facts_used: int = 0
    external_facts_used: int = 0
    merge_confidence: float = 0.8
    requires_review: bool = False


class ConflictAwareMerger:
    """
    Merges internal and external content with conflict awareness.

    Usage:
        merger = ConflictAwareMerger(anthropic_client)
        result = await merger.merge(
            topic="MCA aneurysm clipping",
            internal_content="...",
            external_content="...",
        )
    """

    # Regex patterns for quantitative extraction
    PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')
    MEASUREMENT_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*(mm|cm|ml|cc|minutes?|hours?|days?|mg|g)'
    )
    RECOMMENDATION_PATTERNS = [
        r'should\s+(?:be|not)', r'recommended', r'indicated\s+for',
        r'contraindicated', r'first[\s-]line', r'avoid\s+in', r'preferred'
    ]

    def __init__(self, anthropic_client=None):
        self.client = anthropic_client

    async def merge(
        self,
        topic: str,
        internal_content: str,
        external_content: str,
        internal_sources: Optional[List[Dict]] = None,
        external_sources: Optional[List[Dict]] = None,
    ) -> MergeResult:
        """Merge internal and external content with conflict awareness."""

        # Handle edge cases
        if not internal_content and not external_content:
            return MergeResult(topic=topic, resolved_content="No content available.",
                             merge_strategy_used="none")
        if not external_content:
            return MergeResult(topic=topic, resolved_content=internal_content,
                             merge_strategy_used="internal_only")
        if not internal_content:
            return MergeResult(topic=topic, resolved_content=external_content,
                             merge_strategy_used="external_only")

        # Extract facts from both sources
        internal_facts = self._extract_facts(internal_content, "internal")
        external_facts = self._extract_facts(external_content, "external")

        # Detect conflicts
        conflicts = self._detect_conflicts(internal_facts, external_facts)

        # Build merged content
        if conflicts:
            resolved = await self._merge_with_conflicts(
                topic, internal_content, external_content, conflicts
            )
            strategy = f"conflict_aware ({len(conflicts)} conflicts)"
        else:
            resolved = await self._merge_harmonized(
                topic, internal_content, external_content
            )
            strategy = "harmonized_merge"

        return MergeResult(
            topic=topic,
            resolved_content=resolved,
            merge_strategy_used=strategy,
            conflicts=conflicts,
            conflict_count=len(conflicts),
            internal_facts_used=len(internal_facts),
            external_facts_used=len(external_facts),
            merge_confidence=max(0.5, 0.9 - (len(conflicts) * 0.05)),
            requires_review=any(c.severity in ("high", "critical") for c in conflicts),
        )

    def _extract_facts(self, content: str, source_type: str) -> List[ExtractedFact]:
        """Extract factual claims from content."""
        facts = []
        content_lower = content.lower()

        # Extract percentages
        for match in self.PERCENTAGE_PATTERN.finditer(content):
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].strip()
            facts.append(ExtractedFact(
                claim=context, value=match.group(1), unit="%", source_type=source_type
            ))

        # Extract measurements
        for match in self.MEASUREMENT_PATTERN.finditer(content):
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].strip()
            facts.append(ExtractedFact(
                claim=context, value=match.group(1), unit=match.group(2),
                source_type=source_type
            ))

        # Extract recommendation sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sentence in sentences:
            sent_lower = sentence.lower()
            is_rec = any(re.search(p, sent_lower) for p in self.RECOMMENDATION_PATTERNS)
            is_temporal = any(kw in sent_lower for kw in TEMPORAL_KEYWORDS)
            if is_rec:
                facts.append(ExtractedFact(
                    claim=sentence.strip(), source_type=source_type,
                    is_recommendation=True, is_temporal=is_temporal
                ))

        return facts

    def _detect_conflicts(
        self,
        internal_facts: List[ExtractedFact],
        external_facts: List[ExtractedFact],
    ) -> List[DetectedConflict]:
        """Detect conflicts between internal and external facts."""
        conflicts = []

        # Check quantitative conflicts
        int_quant = [f for f in internal_facts if f.value and f.unit]
        ext_quant = [f for f in external_facts if f.value and f.unit]

        for int_fact in int_quant:
            for ext_fact in ext_quant:
                if int_fact.unit != ext_fact.unit:
                    continue

                # Check topic overlap
                int_words = set(int_fact.claim.lower().split())
                ext_words = set(ext_fact.claim.lower().split())
                if len(int_words & ext_words) < 3:
                    continue

                try:
                    int_val = float(int_fact.value)
                    ext_val = float(ext_fact.value)
                    if int_val > 0:
                        diff_pct = abs(int_val - ext_val) / int_val * 100
                        if diff_pct > 15:
                            severity = "low" if diff_pct < 25 else "medium"
                            if diff_pct > 40:
                                severity = "high"

                            conflict = DetectedConflict(
                                category=ConflictCategory.QUANTITATIVE,
                                description=f"{int_val}{int_fact.unit} vs {ext_val}{ext_fact.unit}",
                                internal_claim=int_fact.claim,
                                external_claim=ext_fact.claim,
                                resolution_strategy=ResolutionStrategy.NOTE_BOTH,
                                severity=severity,
                            )
                            self._apply_resolution(conflict)
                            conflicts.append(conflict)
                except ValueError:
                    continue

        # Check recommendation conflicts
        int_recs = [f for f in internal_facts if f.is_recommendation][:5]
        ext_recs = [f for f in external_facts if f.is_recommendation][:5]

        for int_rec in int_recs:
            for ext_rec in ext_recs:
                # Check if they discuss similar topics
                int_words = set(int_rec.claim.lower().split())
                ext_words = set(ext_rec.claim.lower().split())
                overlap = len(int_words & ext_words)

                if overlap >= 5:
                    # Check for contradictory patterns
                    contradictions = [
                        ("recommended", "contraindicated"),
                        ("should", "should not"),
                        ("first-line", "second-line"),
                        ("safe", "avoid"),
                    ]

                    int_lower = int_rec.claim.lower()
                    ext_lower = ext_rec.claim.lower()

                    for pos, neg in contradictions:
                        if (pos in int_lower and neg in ext_lower) or \
                           (neg in int_lower and pos in ext_lower):

                            category = (ConflictCategory.TEMPORAL if ext_rec.is_temporal
                                       else ConflictCategory.RECOMMENDATION)

                            conflict = DetectedConflict(
                                category=category,
                                description=f"Conflicting recommendations",
                                internal_claim=int_rec.claim[:200],
                                external_claim=ext_rec.claim[:200],
                                resolution_strategy=DEFAULT_RESOLUTION[category],
                                severity="high" if category == ConflictCategory.RECOMMENDATION else "medium",
                            )
                            self._apply_resolution(conflict)
                            conflicts.append(conflict)
                            break

        return conflicts[:5]  # Limit to top 5 conflicts

    def _apply_resolution(self, conflict: DetectedConflict) -> None:
        """Apply resolution strategy to conflict."""
        strategy = conflict.resolution_strategy

        if strategy == ResolutionStrategy.PREFER_INTERNAL:
            conflict.resolved_content = conflict.internal_claim
            conflict.resolution_note = f"Using established source. Recent sources differ: {conflict.external_claim[:80]}..."

        elif strategy == ResolutionStrategy.PREFER_EXTERNAL:
            conflict.resolved_content = conflict.external_claim
            conflict.resolution_note = f"Using recent source (updated). Prior: {conflict.internal_claim[:80]}..."

        elif strategy == ResolutionStrategy.NOTE_BOTH:
            conflict.resolved_content = (
                f"Internal: {conflict.internal_claim}\n"
                f"Recent: {conflict.external_claim}\n"
                f"[NOTE: Verify current guidelines]"
            )
            conflict.resolution_note = "Both values presented"

        else:  # FLAG_FOR_REVIEW
            conflict.resolved_content = (
                f"CONFLICTING: Internal: {conflict.internal_claim[:100]}... | "
                f"External: {conflict.external_claim[:100]}... [REQUIRES REVIEW]"
            )
            conflict.resolution_note = "Flagged for expert review"

    async def _merge_with_conflicts(
        self,
        topic: str,
        internal_content: str,
        external_content: str,
        conflicts: List[DetectedConflict],
    ) -> str:
        """Merge content with conflict awareness."""

        if self.client:
            conflict_notes = "\n".join(
                f"- {c.category.value}: {c.description}\n"
                f"  Internal: {c.internal_claim[:100]}...\n"
                f"  External: {c.external_claim[:100]}...\n"
                f"  Resolution: {c.resolution_strategy.value}"
                for c in conflicts[:3]
            )

            prompt = f"""Merge these sources on "{topic}" with conflict awareness.

INTERNAL (established textbooks - authoritative for anatomy/classical techniques):
{internal_content[:3500]}

EXTERNAL (recent sources - authoritative for recent developments):
{external_content[:1500]}

DETECTED CONFLICTS:
{conflict_notes}

INSTRUCTIONS:
1. ESTABLISHED FACTS (anatomy, techniques): Prefer internal
2. TEMPORAL/GUIDELINES (recent trials, updated guidelines): Prefer external
3. QUANTITATIVE conflicts: Note both values with "[Internal: X, Recent: Y]"
4. RECOMMENDATION conflicts: Flag with "Verify current guidelines"
5. Integrate smoothly - don't just concatenate
6. Use [Internal] and [External] citation markers

MERGED CONTENT:"""

            try:
                response = await self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"LLM merge failed: {e}")

        # Fallback: structured merge
        parts = [f"**{topic}**\n"]
        parts.append("**Established Knowledge (Internal):**")
        parts.append(internal_content[:2500])
        parts.append("\n**Recent Developments (External):**")
        parts.append(external_content[:1200])

        if conflicts:
            parts.append("\n**Noted Discrepancies:**")
            for c in conflicts[:3]:
                parts.append(f"- {c.description}: {c.resolution_note}")

        return "\n".join(parts)

    async def _merge_harmonized(
        self,
        topic: str,
        internal_content: str,
        external_content: str,
    ) -> str:
        """Merge content when no conflicts detected."""

        if self.client:
            prompt = f"""Synthesize these sources on "{topic}" into unified content.

INTERNAL (established):
{internal_content[:3500]}

EXTERNAL (recent):
{external_content[:1500]}

Create a smooth synthesis. Use [Internal] and [External] markers. Be concise.

SYNTHESIZED:"""

            try:
                response = await self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"LLM harmonize failed: {e}")

        # Fallback
        return f"""**{topic}**

**From Internal Library:**
{internal_content[:2500]}

**Recent Developments:**
{external_content[:1200]}"""


async def merge_internal_external(
    anthropic_client,
    topic: str,
    internal_content: str,
    external_content: str,
) -> MergeResult:
    """Convenience function for merging."""
    merger = ConflictAwareMerger(anthropic_client)
    return await merger.merge(topic, internal_content, external_content)
