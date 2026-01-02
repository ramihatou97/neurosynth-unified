"""
NeuroSynth Unified - Conflict Detection
========================================

Hybrid conflict detection for synthesis engine.
Default: Heuristic-based (free, fast)
Opt-in: LLM-based (costly, deep)
"""

import re
import logging
from typing import List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """Types of conflicts that can be detected in source material."""
    QUANTITATIVE = "quantitative"      # Numerical disagreements (e.g., 15% vs 25%)
    CONTRADICTORY = "contradictory"    # Direct statement contradictions
    APPROACH = "approach"              # Different recommended techniques
    TEMPORAL = "temporal"              # Outdated vs current information


@dataclass
class Conflict:
    """A detected conflict between sources."""
    type: ConflictType
    description: str
    source_a: str
    source_b: str
    section: str = ""
    severity: str = "medium"  # low, medium, high
    context_a: str = ""
    context_b: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "description": self.description,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "section": self.section,
            "severity": self.severity,
            "context_a": self.context_a,
            "context_b": self.context_b
        }


@dataclass
class ConflictReport:
    """Summary of all detected conflicts."""
    conflicts: List[Conflict] = field(default_factory=list)
    mode: str = "heuristic"
    sections_analyzed: int = 0

    @property
    def count(self) -> int:
        return len(self.conflicts)

    @property
    def by_type(self) -> dict:
        result = {t.value: 0 for t in ConflictType}
        for c in self.conflicts:
            result[c.type.value] += 1
        return result

    @property
    def by_severity(self) -> dict:
        result = {"low": 0, "medium": 0, "high": 0}
        for c in self.conflicts:
            result[c.severity] += 1
        return result

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mode": self.mode,
            "sections_analyzed": self.sections_analyzed,
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "conflicts": [c.to_dict() for c in self.conflicts]
        }


class ConflictHandler:
    """
    Hybrid conflict detection for synthesis.

    Default mode: Heuristic (regex-based, free, fast)
    Opt-in mode: LLM (Claude-based, costly, comprehensive)

    Usage:
        handler = ConflictHandler(llm_client)
        conflicts = await handler.detect_conflicts(sections, mode="heuristic")
    """

    # Patterns for quantitative conflict detection
    PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')
    RANGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*[-–to]\s*(\d+(?:\.\d+)?)\s*%')
    MEASUREMENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(mm|cm|ml|cc|hours?|days?|weeks?|months?|years?)')

    # Keywords indicating contradictory statements
    CONTRADICT_KEYWORDS = [
        "however", "in contrast", "alternatively", "conversely",
        "on the other hand", "unlike", "whereas", "but"
    ]

    # Keywords indicating approach differences
    APPROACH_KEYWORDS = [
        "preferred", "recommended", "standard", "traditional",
        "modern", "alternative", "technique", "approach"
    ]

    def __init__(self, llm_client=None):
        """
        Initialize conflict handler.

        Args:
            llm_client: Optional Anthropic client for LLM-based detection
        """
        self.llm_client = llm_client

    async def detect_conflicts(
        self,
        sections: List[Any],
        mode: str = "heuristic"
    ) -> ConflictReport:
        """
        Detect conflicts across synthesis sections.

        Args:
            sections: List of SynthesisSection or similar objects with
                     source_chunks attribute
            mode: "heuristic" (default, free) or "llm" (opt-in, costly)

        Returns:
            ConflictReport with all detected conflicts
        """
        if mode == "llm" and self.llm_client:
            return await self._detect_with_llm(sections)
        return self._detect_heuristic(sections)

    def _detect_heuristic(self, sections: List[Any]) -> ConflictReport:
        """
        Fast regex-based detection for quantitative disagreements.

        Checks for:
        - Percentage differences > 3%
        - Contradictory keywords in nearby content
        - Different measurements for same concept
        """
        conflicts = []
        sections_analyzed = 0

        for section in sections:
            # Handle both SynthesisSection and dict-like objects
            source_chunks = getattr(section, 'source_chunks', None)
            if source_chunks is None:
                source_chunks = getattr(section, 'sources', [])
            if not source_chunks or len(source_chunks) < 2:
                continue

            sections_analyzed += 1
            section_title = getattr(section, 'title', str(section))

            # Extract all quantitative values with context
            chunk_data = []
            for chunk in source_chunks:
                content = chunk.get('content', '') if isinstance(chunk, dict) else getattr(chunk, 'content', str(chunk))
                source = chunk.get('document_title', 'Unknown') if isinstance(chunk, dict) else getattr(chunk, 'source_title', getattr(chunk, 'document_title', 'Unknown'))
                chunk_data.append({
                    'content': content,
                    'source': source
                })

            # Find quantitative conflicts
            percentage_conflicts = self._find_percentage_conflicts(chunk_data, section_title)
            conflicts.extend(percentage_conflicts)

            # Find measurement conflicts
            measurement_conflicts = self._find_measurement_conflicts(chunk_data, section_title)
            conflicts.extend(measurement_conflicts)

        return ConflictReport(
            conflicts=conflicts,
            mode="heuristic",
            sections_analyzed=sections_analyzed
        )

    def _find_percentage_conflicts(
        self,
        chunk_data: List[dict],
        section_title: str
    ) -> List[Conflict]:
        """Find conflicting percentages across chunks."""
        conflicts = []

        # Extract percentages with context
        percentages = []
        for chunk in chunk_data:
            content = chunk['content']
            source = chunk['source']

            for match in self.PERCENTAGE_PATTERN.finditer(content):
                # Get surrounding context (30 chars before, 30 after)
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                context = content[start:end]

                percentages.append({
                    'value': float(match.group(1)),
                    'context': context.lower(),
                    'source': source,
                    'full_context': context
                })

        # Compare percentages for conflicts
        for i, p1 in enumerate(percentages):
            for p2 in percentages[i+1:]:
                # Skip if same source
                if p1['source'] == p2['source']:
                    continue

                # Check if contexts suggest same topic (word overlap)
                p1_words = set(re.findall(r'\w+', p1['context']))
                p2_words = set(re.findall(r'\w+', p2['context']))
                overlap = len(p1_words & p2_words)

                # If contexts are similar and values differ significantly
                if overlap >= 2 and abs(p1['value'] - p2['value']) > 3:
                    severity = "low" if abs(p1['value'] - p2['value']) < 10 else "medium"
                    if abs(p1['value'] - p2['value']) > 20:
                        severity = "high"

                    conflicts.append(Conflict(
                        type=ConflictType.QUANTITATIVE,
                        description=f"{p1['value']}% vs {p2['value']}%",
                        source_a=p1['source'],
                        source_b=p2['source'],
                        section=section_title,
                        severity=severity,
                        context_a=p1['full_context'],
                        context_b=p2['full_context']
                    ))

        return conflicts

    def _find_measurement_conflicts(
        self,
        chunk_data: List[dict],
        section_title: str
    ) -> List[Conflict]:
        """Find conflicting measurements (mm, cm, etc.) across chunks."""
        conflicts = []

        # Extract measurements with context
        measurements = []
        for chunk in chunk_data:
            content = chunk['content']
            source = chunk['source']

            for match in self.MEASUREMENT_PATTERN.finditer(content):
                start = max(0, match.start() - 40)
                end = min(len(content), match.end() + 20)
                context = content[start:end]

                measurements.append({
                    'value': float(match.group(1)),
                    'unit': match.group(2).lower().rstrip('s'),  # Normalize unit
                    'context': context.lower(),
                    'source': source,
                    'full_context': context
                })

        # Compare measurements with same unit
        for i, m1 in enumerate(measurements):
            for m2 in measurements[i+1:]:
                # Must have same unit and different sources
                if m1['unit'] != m2['unit'] or m1['source'] == m2['source']:
                    continue

                # Check context overlap
                m1_words = set(re.findall(r'\w+', m1['context']))
                m2_words = set(re.findall(r'\w+', m2['context']))
                overlap = len(m1_words & m2_words)

                # If contexts similar and values differ by >20%
                if overlap >= 2:
                    diff_pct = abs(m1['value'] - m2['value']) / max(m1['value'], m2['value']) * 100
                    if diff_pct > 20:
                        severity = "low" if diff_pct < 30 else "medium"
                        if diff_pct > 50:
                            severity = "high"

                        conflicts.append(Conflict(
                            type=ConflictType.QUANTITATIVE,
                            description=f"{m1['value']} {m1['unit']} vs {m2['value']} {m2['unit']}",
                            source_a=m1['source'],
                            source_b=m2['source'],
                            section=section_title,
                            severity=severity,
                            context_a=m1['full_context'],
                            context_b=m2['full_context']
                        ))

        return conflicts

    async def _detect_with_llm(self, sections: List[Any]) -> ConflictReport:
        """
        LLM-based conflict detection - opt-in only.

        Uses Claude to analyze sections for deeper conflicts:
        - Contradictory clinical recommendations
        - Outdated vs current practice
        - Approach disagreements
        """
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to heuristic")
            return self._detect_heuristic(sections)

        conflicts = []
        sections_analyzed = 0

        for section in sections:
            source_chunks = getattr(section, 'source_chunks', None)
            if source_chunks is None:
                source_chunks = getattr(section, 'sources', [])
            if not source_chunks or len(source_chunks) < 2:
                continue

            sections_analyzed += 1
            section_title = getattr(section, 'title', str(section))

            # Build context for LLM
            chunk_texts = []
            for i, chunk in enumerate(source_chunks[:5], 1):  # Limit to 5 chunks
                content = chunk.get('content', '') if isinstance(chunk, dict) else getattr(chunk, 'content', str(chunk))
                source = chunk.get('document_title', 'Unknown') if isinstance(chunk, dict) else getattr(chunk, 'source_title', 'Unknown')
                chunk_texts.append(f"[Source {i}: {source}]\n{content[:500]}")

            prompt = f"""Analyze these neurosurgical text excerpts for conflicts or contradictions.

Section: {section_title}

{chr(10).join(chunk_texts)}

Identify any:
1. QUANTITATIVE conflicts (different numbers for same metric)
2. CONTRADICTORY statements (opposing claims)
3. APPROACH differences (different recommended techniques)
4. TEMPORAL conflicts (outdated vs current practice)

For each conflict found, respond with:
TYPE: [quantitative/contradictory/approach/temporal]
DESCRIPTION: [brief description]
SOURCE_A: [first source name]
SOURCE_B: [second source name]
SEVERITY: [low/medium/high]

If no conflicts found, respond with: NO_CONFLICTS"""

            try:
                response = await self.llm_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text

                if "NO_CONFLICTS" not in response_text:
                    parsed_conflicts = self._parse_llm_conflicts(response_text, section_title)
                    conflicts.extend(parsed_conflicts)

            except Exception as e:
                logger.warning(f"LLM conflict detection failed for {section_title}: {e}")
                # Fall back to heuristic for this section
                heuristic_result = self._detect_heuristic([section])
                conflicts.extend(heuristic_result.conflicts)

        return ConflictReport(
            conflicts=conflicts,
            mode="llm",
            sections_analyzed=sections_analyzed
        )

    def _parse_llm_conflicts(self, response: str, section_title: str) -> List[Conflict]:
        """Parse LLM response into Conflict objects."""
        conflicts = []

        # Split by conflict entries
        entries = re.split(r'\n(?=TYPE:)', response)

        for entry in entries:
            if not entry.strip() or "TYPE:" not in entry:
                continue

            try:
                type_match = re.search(r'TYPE:\s*(\w+)', entry, re.IGNORECASE)
                desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=SOURCE_A:|$)', entry, re.DOTALL)
                src_a_match = re.search(r'SOURCE_A:\s*(.+?)(?=SOURCE_B:|$)', entry, re.DOTALL)
                src_b_match = re.search(r'SOURCE_B:\s*(.+?)(?=SEVERITY:|$)', entry, re.DOTALL)
                sev_match = re.search(r'SEVERITY:\s*(\w+)', entry, re.IGNORECASE)

                if type_match and desc_match:
                    conflict_type_str = type_match.group(1).lower()
                    try:
                        conflict_type = ConflictType(conflict_type_str)
                    except ValueError:
                        conflict_type = ConflictType.CONTRADICTORY

                    conflicts.append(Conflict(
                        type=conflict_type,
                        description=desc_match.group(1).strip(),
                        source_a=src_a_match.group(1).strip() if src_a_match else "Unknown",
                        source_b=src_b_match.group(1).strip() if src_b_match else "Unknown",
                        section=section_title,
                        severity=sev_match.group(1).lower() if sev_match else "medium"
                    ))
            except Exception as e:
                logger.debug(f"Failed to parse conflict entry: {e}")
                continue

        return conflicts
