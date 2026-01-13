"""
Negative Constraint Recognition
================================

P2 Enhancement: Detection and Extraction of Negative Constraints

This module identifies and extracts negative constraints from medical content,
ensuring that explicit exclusions and contraindications are properly highlighted
and not inadvertently overridden by gap-filling.

Clinical Safety Rationale:
- "Avoid excessive traction" → Must be preserved in output
- "Contraindicated in renal failure" → Cannot recommend drug for CKD patient
- "Do not use in pregnancy" → Must flag for pregnant patients
- "Never clamp without proximal control" → Safety-critical instruction

Detection Approach:
1. Parse for negative constraint markers (avoid, never, contraindicated, etc.)
2. Extract the constrained action and context
3. Structure as machine-readable constraints
4. Prevent gap-filling from contradicting explicit constraints
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Type of negative constraint."""

    CONTRAINDICATION = "contraindication"      # Drug/procedure not to be used
    PROHIBITION = "prohibition"                # Action explicitly forbidden
    CAUTION = "caution"                        # Action to minimize/avoid if possible
    CONDITIONAL_AVOID = "conditional_avoid"    # Avoid in specific circumstances
    SEQUENCE_CONSTRAINT = "sequence"           # Must not do before something else
    DOSING_LIMIT = "dosing_limit"              # Maximum dose or rate constraint


class ConstraintSeverity(Enum):
    """Severity/strength of the constraint."""

    ABSOLUTE = "absolute"      # Never do under any circumstances
    STRONG = "strong"          # Avoid unless compelling reason
    MODERATE = "moderate"      # Prefer to avoid, but acceptable in some cases
    ADVISORY = "advisory"      # Best practice to avoid, but not strict


@dataclass
class NegativeConstraint:
    """Represents an extracted negative constraint."""

    constraint_id: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    action: str                           # What is constrained (e.g., "use mannitol")
    reason: str                           # Why (e.g., "causes renal injury")
    context: Optional[str] = None         # When constraint applies
    conditions: List[str] = field(default_factory=list)  # Specific conditions
    source_text: str = ""                 # Original text this was extracted from
    location: str = ""                    # Where in content this was found


@dataclass
class ConstraintViolation:
    """Represents a potential violation of a constraint."""

    constraint: NegativeConstraint
    violating_text: str
    severity: ConstraintSeverity
    recommendation: str


# =============================================================================
# NEGATIVE CONSTRAINT PATTERNS
# =============================================================================

# Patterns for detecting negative constraints
ABSOLUTE_PROHIBITION_PATTERNS = [
    # Never patterns
    r"never\s+(?:use|give|administer|perform|do|attempt)\s+(.+?)(?:\.|,|;|$)",
    r"(?:should|must)\s+never\s+(.+?)(?:\.|,|;|$)",
    r"absolutely\s+contraindicated\s+(.+?)(?:\.|,|;|$)",
    r"is\s+(?:an?\s+)?absolute\s+contraindication(?:\s+to\s+|\s+for\s+)(.+?)(?:\.|,|;|$)",
    # Do not patterns
    r"do\s+not\s+(?:ever\s+)?(.+?)(?:under\s+any\s+circumstances)?(?:\.|,|;|$)",
    r"(?:should|must)\s+not\s+(?:be\s+)?(?:used|given|administered|performed)(.+?)(?:\.|,|;|$)",
]

STRONG_AVOID_PATTERNS = [
    # Avoid patterns
    r"avoid\s+(.+?)(?:\.|,|;|$)",
    r"should\s+(?:be\s+)?avoided\s+(.+?)(?:\.|,|;|$)",
    r"(?:is|are)\s+contraindicated\s+(.+?)(?:\.|,|;|$)",
    r"contraindicated\s+in\s+(.+?)(?:\.|,|;|$)",
    r"not\s+recommended\s+(?:for\s+|in\s+)?(.+?)(?:\.|,|;|$)",
    r"should\s+not\s+be\s+(?:used|given|performed)\s+(.+?)(?:\.|,|;|$)",
]

CAUTION_PATTERNS = [
    # Caution/minimize patterns
    r"minimize\s+(.+?)(?:\.|,|;|$)",
    r"limit\s+(.+?)(?:\.|,|;|$)",
    r"(?:use\s+)?caution\s+(?:with|when|if)\s+(.+?)(?:\.|,|;|$)",
    r"exercise\s+caution\s+(.+?)(?:\.|,|;|$)",
    r"be\s+careful\s+(?:to\s+)?(?:not\s+)?(.+?)(?:\.|,|;|$)",
    r"only\s+if\s+(?:absolutely\s+)?necessary\s+(.+?)(?:\.|,|;|$)",
]

CONDITIONAL_PATTERNS = [
    # Conditional avoid patterns
    r"avoid\s+(.+?)\s+(?:in|when|if|with)\s+(.+?)(?:\.|,|;|$)",
    r"contraindicated\s+(?:in\s+)?(?:patients\s+)?with\s+(.+?)(?:\.|,|;|$)",
    r"do\s+not\s+(.+?)\s+(?:in|when|if)\s+(.+?)(?:\.|,|;|$)",
    r"(?:should|must)\s+not\s+(.+?)\s+(?:in|when|if)\s+(.+?)(?:\.|,|;|$)",
]

SEQUENCE_PATTERNS = [
    # Sequence constraint patterns
    r"(?:do\s+)?not\s+(.+?)\s+(?:before|until|without)\s+(.+?)(?:\.|,|;|$)",
    r"never\s+(.+?)\s+(?:before|until|without)\s+(.+?)(?:\.|,|;|$)",
    r"only\s+after\s+(.+?),?\s+(?:should|can|may)\s+(.+?)(?:\.|,|;|$)",
    r"must\s+(?:first\s+)?(.+?)\s+before\s+(.+?)(?:\.|,|;|$)",
]

DOSING_LIMIT_PATTERNS = [
    # Dosing limit patterns
    r"(?:do\s+)?not\s+exceed\s+(.+?)(?:\.|,|;|$)",
    r"maximum\s+(?:dose|rate|amount)(?:\s+of)?\s+(.+?)(?:\.|,|;|$)",
    r"no\s+more\s+than\s+(.+?)(?:\.|,|;|$)",
    r"limit\s+(?:to\s+|at\s+)?(.+?)(?:\.|,|;|$)",
]

# Known critical constraints (pre-defined)
KNOWN_CRITICAL_CONSTRAINTS = [
    NegativeConstraint(
        constraint_id="STEROIDS_TBI",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="corticosteroids in traumatic brain injury",
        reason="CRASH trial showed increased mortality",
        conditions=["traumatic_brain_injury"],
        source_text="Corticosteroids are contraindicated in TBI per CRASH trial",
    ),
    NegativeConstraint(
        constraint_id="NASCIS_SCI",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="high-dose methylprednisolone for spinal cord injury",
        reason="No benefit, significant complications",
        conditions=["spinal_cord_injury"],
        source_text="NASCIS protocol no longer recommended per CNS 2013 guidelines",
    ),
    NegativeConstraint(
        constraint_id="MANNITOL_RENAL",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="mannitol in renal failure",
        reason="Cannot be excreted, causes volume overload",
        conditions=["renal_failure", "chronic_kidney_disease", "gfr_below_30"],
        source_text="Mannitol contraindicated with GFR < 30",
    ),
    NegativeConstraint(
        constraint_id="HYPERVENTILATION_PROPHYLACTIC",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="prophylactic hyperventilation for ICP",
        reason="Causes cerebral ischemia",
        conditions=["icp_management"],
        source_text="Avoid prophylactic hyperventilation per BTF guidelines",
    ),
    NegativeConstraint(
        constraint_id="VALPROATE_PREGNANCY",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="valproic acid in pregnancy",
        reason="Neural tube defects, developmental delay",
        conditions=["pregnancy"],
        source_text="Valproic acid pregnancy category X",
    ),
    NegativeConstraint(
        constraint_id="TPA_CRANIOTOMY",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="TPA within 3 months of craniotomy",
        reason="Catastrophic intracranial hemorrhage risk",
        conditions=["recent_craniotomy"],
        source_text="TPA contraindicated within 3 months of intracranial surgery",
    ),
    NegativeConstraint(
        constraint_id="GADOLINIUM_RENAL",
        constraint_type=ConstraintType.CONTRAINDICATION,
        severity=ConstraintSeverity.ABSOLUTE,
        action="gadolinium contrast in severe renal failure",
        reason="Nephrogenic systemic fibrosis",
        conditions=["renal_failure", "gfr_below_30"],
        source_text="Gadolinium contraindicated in GFR < 30 due to NSF risk",
    ),
]


class NegativeConstraintExtractor:
    """
    Extracts negative constraints from medical content.

    Usage:
        extractor = NegativeConstraintExtractor()
        constraints = extractor.extract_constraints(content)
        violations = extractor.check_for_violations(constraints, recommendations)
    """

    def __init__(
        self,
        include_known_constraints: bool = True,
        custom_constraints: Optional[List[NegativeConstraint]] = None,
    ):
        self.known_constraints = (
            KNOWN_CRITICAL_CONSTRAINTS.copy() if include_known_constraints else []
        )
        if custom_constraints:
            self.known_constraints.extend(custom_constraints)

        # Compile patterns
        self._absolute_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in ABSOLUTE_PROHIBITION_PATTERNS
        ]
        self._strong_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in STRONG_AVOID_PATTERNS
        ]
        self._caution_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in CAUTION_PATTERNS
        ]
        self._conditional_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in CONDITIONAL_PATTERNS
        ]
        self._sequence_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in SEQUENCE_PATTERNS
        ]
        self._dosing_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in DOSING_LIMIT_PATTERNS
        ]

        self.logger = logging.getLogger(__name__)
        self._constraint_counter = 0

    def _generate_id(self) -> str:
        """Generate unique constraint ID."""
        self._constraint_counter += 1
        return f"NC_{self._constraint_counter:04d}"

    def extract_constraints(
        self,
        content: str,
        include_context: bool = True,
    ) -> List[NegativeConstraint]:
        """
        Extract all negative constraints from content.

        Args:
            content: Medical text to analyze
            include_context: Whether to include surrounding context

        Returns:
            List of extracted negative constraints
        """
        constraints: List[NegativeConstraint] = []

        # Split into sentences for better context extraction
        sentences = re.split(r'(?<=[.!?])\s+', content)

        for i, sentence in enumerate(sentences):
            # Check absolute prohibitions
            for pattern in self._absolute_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    action = match.group(1).strip()
                    context_text = self._get_context(sentences, i, include_context)
                    constraints.append(NegativeConstraint(
                        constraint_id=self._generate_id(),
                        constraint_type=ConstraintType.PROHIBITION,
                        severity=ConstraintSeverity.ABSOLUTE,
                        action=action,
                        reason="Explicitly prohibited in source text",
                        source_text=sentence.strip(),
                        location=context_text,
                    ))

            # Check strong avoid patterns
            for pattern in self._strong_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    action = match.group(1).strip()
                    constraints.append(NegativeConstraint(
                        constraint_id=self._generate_id(),
                        constraint_type=ConstraintType.CONTRAINDICATION,
                        severity=ConstraintSeverity.STRONG,
                        action=action,
                        reason="Strongly discouraged in source text",
                        source_text=sentence.strip(),
                    ))

            # Check caution patterns
            for pattern in self._caution_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    action = match.group(1).strip()
                    constraints.append(NegativeConstraint(
                        constraint_id=self._generate_id(),
                        constraint_type=ConstraintType.CAUTION,
                        severity=ConstraintSeverity.MODERATE,
                        action=action,
                        reason="Caution advised",
                        source_text=sentence.strip(),
                    ))

            # Check conditional avoid patterns
            for pattern in self._conditional_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        action = groups[0].strip()
                        condition = groups[1].strip()
                        constraints.append(NegativeConstraint(
                            constraint_id=self._generate_id(),
                            constraint_type=ConstraintType.CONDITIONAL_AVOID,
                            severity=ConstraintSeverity.STRONG,
                            action=action,
                            reason=f"Avoid in context: {condition}",
                            context=condition,
                            conditions=[condition],
                            source_text=sentence.strip(),
                        ))

            # Check sequence constraints
            for pattern in self._sequence_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        action = groups[0].strip()
                        prerequisite = groups[1].strip()
                        constraints.append(NegativeConstraint(
                            constraint_id=self._generate_id(),
                            constraint_type=ConstraintType.SEQUENCE_CONSTRAINT,
                            severity=ConstraintSeverity.STRONG,
                            action=f"{action} before {prerequisite}",
                            reason="Sequence constraint",
                            source_text=sentence.strip(),
                        ))

            # Check dosing limits
            for pattern in self._dosing_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    limit = match.group(1).strip()
                    constraints.append(NegativeConstraint(
                        constraint_id=self._generate_id(),
                        constraint_type=ConstraintType.DOSING_LIMIT,
                        severity=ConstraintSeverity.STRONG,
                        action=f"exceed {limit}",
                        reason="Dosing/rate limit",
                        source_text=sentence.strip(),
                    ))

        # Deduplicate by action similarity
        constraints = self._deduplicate_constraints(constraints)

        return constraints

    def _get_context(
        self,
        sentences: List[str],
        current_index: int,
        include_context: bool = True,
    ) -> str:
        """Get surrounding context for a constraint."""
        if not include_context:
            return ""

        start = max(0, current_index - 1)
        end = min(len(sentences), current_index + 2)
        return " ".join(sentences[start:end])

    def _deduplicate_constraints(
        self,
        constraints: List[NegativeConstraint]
    ) -> List[NegativeConstraint]:
        """Remove duplicate constraints based on action similarity."""
        seen_actions: Set[str] = set()
        unique: List[NegativeConstraint] = []

        for constraint in constraints:
            # Normalize action for comparison
            normalized_action = constraint.action.lower().strip()
            # Take first few words for comparison
            action_key = " ".join(normalized_action.split()[:5])

            if action_key not in seen_actions:
                seen_actions.add(action_key)
                unique.append(constraint)

        return unique

    def check_for_violations(
        self,
        constraints: List[NegativeConstraint],
        text_to_check: str,
    ) -> List[ConstraintViolation]:
        """
        Check if a piece of text violates any constraints.

        Args:
            constraints: List of constraints to check against
            text_to_check: Text that might violate constraints

        Returns:
            List of potential violations
        """
        violations: List[ConstraintViolation] = []
        text_lower = text_to_check.lower()

        for constraint in constraints:
            # Check if the constrained action is mentioned positively
            action_terms = constraint.action.lower().split()
            key_terms = [t for t in action_terms if len(t) > 3]

            # Look for positive mention of the constrained action
            action_mentioned = all(term in text_lower for term in key_terms[:3])

            if action_mentioned:
                # Check if it's mentioned in a way that seems like a recommendation
                recommendation_indicators = [
                    "recommend",
                    "administer",
                    "give",
                    "use",
                    "initiate",
                    "start",
                    "consider",
                    "prescribe",
                ]

                for indicator in recommendation_indicators:
                    if indicator in text_lower:
                        # Potential violation
                        violations.append(ConstraintViolation(
                            constraint=constraint,
                            violating_text=text_to_check[:200],
                            severity=constraint.severity,
                            recommendation=(
                                f"WARNING: Content may recommend '{constraint.action}' "
                                f"which is constrained. Reason: {constraint.reason}"
                            ),
                        ))
                        break

        return violations

    def check_known_constraints(
        self,
        content: str,
        patient_conditions: Optional[Set[str]] = None,
    ) -> List[NegativeConstraint]:
        """
        Check content against known critical constraints.

        Returns constraints that apply based on content or patient conditions.
        """
        applicable: List[NegativeConstraint] = []
        content_lower = content.lower()
        conditions_normalized = (
            {c.lower().replace(" ", "_") for c in patient_conditions}
            if patient_conditions else set()
        )

        for constraint in self.known_constraints:
            # Check if action is mentioned in content
            action_terms = constraint.action.lower().split()
            action_mentioned = any(
                term in content_lower
                for term in action_terms
                if len(term) > 3
            )

            # Check if patient conditions match constraint conditions
            conditions_match = any(
                cond in conditions_normalized
                for cond in constraint.conditions
            )

            if action_mentioned and (conditions_match or not patient_conditions):
                applicable.append(constraint)

        return applicable

    def get_constraint_summary(
        self,
        constraints: List[NegativeConstraint]
    ) -> Dict[str, Any]:
        """Generate summary of extracted constraints."""
        by_type = {}
        for t in ConstraintType:
            type_constraints = [c for c in constraints if c.constraint_type == t]
            by_type[t.value] = len(type_constraints)

        by_severity = {}
        for s in ConstraintSeverity:
            severity_constraints = [c for c in constraints if c.severity == s]
            by_severity[s.value] = len(severity_constraints)

        return {
            "total_constraints": len(constraints),
            "by_type": by_type,
            "by_severity": by_severity,
            "absolute_constraints": [
                {
                    "action": c.action,
                    "reason": c.reason,
                    "source": c.source_text[:100] if c.source_text else "",
                }
                for c in constraints
                if c.severity == ConstraintSeverity.ABSOLUTE
            ],
            "critical_constraints": [
                {
                    "action": c.action,
                    "reason": c.reason,
                    "conditions": c.conditions,
                }
                for c in constraints
                if c.severity in (ConstraintSeverity.ABSOLUTE, ConstraintSeverity.STRONG)
            ],
        }
