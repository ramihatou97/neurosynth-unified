"""
NeuroSynth Enhanced Relation Extractor
Production-grade entity relation extraction for neurosurgical knowledge graphs.

Enhancements over base extractor:
- Coordination handling: "MCA supplies frontal and temporal lobes" → 2 relations
- Negation detection: "No evidence X supplies Y" → is_negated=True
- Entity-first NER: Find UMLS entities, then classify relations between pairs
- Tiered LLM verification: Route by confidence to LLM verify/complete
- Coreference resolution: "The MCA... It supplies..." → resolves pronouns

Replaces fragile regex with spaCy NLP + multi-strategy extraction.
"""

import re
import json
import hashlib
import logging
from typing import Optional, List, Tuple, Set, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import spacy
from spacy.tokens import Doc, Span, Token

# Local imports
from src.core.relation_config import RelationExtractionConfig, ExtractionStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Data Classes
# =============================================================================

class RelationType(str, Enum):
    """Supported relation types for neurosurgical knowledge graphs."""
    # Vascular
    SUPPLIES = "supplies"
    DRAINS_TO = "drains_to"
    BRANCHES_FROM = "branches_from"
    ANASTOMOSES_WITH = "anastomoses_with"
    
    # Neural
    INNERVATES = "innervates"
    PROJECTS_TO = "projects_to"
    RECEIVES_FROM = "receives_from"
    
    # Spatial
    ADJACENT_TO = "adjacent_to"
    CONTAINED_IN = "contained_in"
    TRAVERSES = "traverses"
    
    # Clinical
    CAUSES = "causes"
    TREATS = "treats"
    INDICATES = "indicates"
    CONTRAINDICATED_FOR = "contraindicated_for"
    
    # Taxonomy (CRITICAL for query expansion)
    IS_A = "is_a"
    HAS_PART = "has_part"


class ExtractionMethod(str, Enum):
    """Tracks how a relation was extracted."""
    DEPENDENCY = "dependency"           # spaCy dependency parsing
    ENTITY_FIRST = "entity_first"       # NER-driven pair classification
    LLM_COMPLETE = "llm_complete"       # LLM full extraction
    LLM_VERIFIED = "llm_verified"       # spaCy extraction verified by LLM
    HYBRID = "hybrid"                   # Multiple methods agreed
    TAXONOMY = "taxonomy"               # From static taxonomy


@dataclass
class ExtractedRelation:
    """A single extracted relation with full metadata."""
    source: str
    target: str
    relation: RelationType
    confidence: float
    context_snippet: str
    source_normalized: Optional[str] = None
    target_normalized: Optional[str] = None
    bidirectional: bool = False
    
    # Enhancement fields
    is_negated: bool = False
    negation_cue: Optional[str] = None
    extraction_method: str = "dependency"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "source_normalized": self.source_normalized or self.source,
            "target_normalized": self.target_normalized or self.target,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "context_snippet": self.context_snippet,
            "bidirectional": self.bidirectional,
            "is_negated": self.is_negated,
            "negation_cue": self.negation_cue,
            "extraction_method": self.extraction_method,
        }
    
    @property
    def hash_id(self) -> str:
        """Unique identifier for deduplication (includes negation state)."""
        data = (
            f"{self.source_normalized or self.source}:"
            f"{self.target_normalized or self.target}:"
            f"{self.relation.value}:"
            f"{self.is_negated}"
        )
        return hashlib.md5(data.encode()).hexdigest()


# =============================================================================
# Abbreviation & Taxonomy Dictionaries
# =============================================================================

NEURO_ABBREVIATIONS = {
    "mca": "middle cerebral artery",
    "aca": "anterior cerebral artery",
    "pca": "posterior cerebral artery",
    "ica": "internal carotid artery",
    "va": "vertebral artery",
    "ba": "basilar artery",
    "sss": "superior sagittal sinus",
    "cn": "cranial nerve",
    "csf": "cerebrospinal fluid",
    "gm": "gray matter",
    "wm": "white matter",
    "sah": "subarachnoid hemorrhage",
    "ich": "intracerebral hemorrhage",
    "sdh": "subdural hematoma",
    "edh": "epidural hematoma",
    "gbm": "glioblastoma",
    "stn": "subthalamic nucleus",
    "gpi": "globus pallidus internus",
    "snr": "substantia nigra pars reticulata",
    "vpl": "ventral posterolateral nucleus",
    "vpm": "ventral posteromedial nucleus",
}

TAXONOMY = {
    # Pathology
    "glioblastoma": ["glioma", "brain tumor", "neoplasm"],
    "meningioma": ["brain tumor", "neoplasm"],
    "schwannoma": ["nerve tumor", "neoplasm"],
    "aneurysm": ["vascular malformation", "cerebrovascular disease"],
    "arteriovenous malformation": ["vascular malformation", "cerebrovascular disease"],
    
    # Hemorrhage types
    "subarachnoid hemorrhage": ["hemorrhage", "stroke"],
    "intracerebral hemorrhage": ["hemorrhage", "stroke"],
    "subdural hematoma": ["hemorrhage", "extra-axial hemorrhage"],
    "epidural hematoma": ["hemorrhage", "extra-axial hemorrhage"],
    "intraventricular hemorrhage": ["hemorrhage"],
    
    # Anatomy - Arteries
    "middle cerebral artery": ["cerebral artery", "anterior circulation"],
    "anterior cerebral artery": ["cerebral artery", "anterior circulation"],
    "posterior cerebral artery": ["cerebral artery", "posterior circulation"],
    "internal carotid artery": ["carotid artery", "anterior circulation"],
    "basilar artery": ["posterior circulation", "vertebrobasilar system"],
    "vertebral artery": ["posterior circulation", "vertebrobasilar system"],
    
    # Anatomy - Nerves
    "cranial nerve": ["peripheral nerve"],
    "olfactory nerve": ["cranial nerve"],
    "optic nerve": ["cranial nerve"],
    "oculomotor nerve": ["cranial nerve"],
    "trigeminal nerve": ["cranial nerve"],
    "facial nerve": ["cranial nerve"],
    "vestibulocochlear nerve": ["cranial nerve"],
    "vagus nerve": ["cranial nerve"],
    
    # Anatomy - Lobes
    "frontal lobe": ["cerebral lobe", "cerebral cortex"],
    "temporal lobe": ["cerebral lobe", "cerebral cortex"],
    "parietal lobe": ["cerebral lobe", "cerebral cortex"],
    "occipital lobe": ["cerebral lobe", "cerebral cortex"],
    "insular cortex": ["cerebral cortex"],
    
    # Spine
    "cervical vertebra": ["vertebra", "cervical spine"],
    "thoracic vertebra": ["vertebra", "thoracic spine"],
    "lumbar vertebra": ["vertebra", "lumbar spine"],
    "sacral vertebra": ["vertebra", "sacrum"],
    "intervertebral disc": ["spinal structure"],
    "spinal cord": ["central nervous system"],
}


# =============================================================================
# Negation Detector
# =============================================================================

class NegationDetector:
    """
    Detects negated relations using negspacy + custom cues.
    
    Handles patterns like:
    - "No evidence that X supplies Y"
    - "X does not supply Y"
    - "Without X supplying Y"
    - "There is no X-Y connection"
    """
    
    # Default negation cues (extended beyond negspacy defaults)
    DEFAULT_CUES = [
        "no evidence",
        "not",
        "without",
        "absence of",
        "negative for",
        "unlikely",
        "failed to",
        "failed to demonstrate",
        "rules out",
        "ruled out",
        "excluded",
        "no",
        "never",
        "neither",
        "cannot",
        "couldn't",
        "won't",
        "wouldn't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    ]
    
    def __init__(self, nlp, config: RelationExtractionConfig):
        """
        Initialize negation detector.
        
        Args:
            nlp: spaCy language model (will add negex pipe if not present)
            config: Extraction configuration
        """
        self.nlp = nlp
        self.config = config
        self.negation_cues = set(self.DEFAULT_CUES)
        
        # Add custom cues from config
        if config.additional_negation_cues:
            self.negation_cues.update(config.additional_negation_cues)
        
        # Initialize negspacy
        self._init_negex()
    
    def _init_negex(self):
        """Add negex pipe to spaCy pipeline if not present."""
        if "negex" not in self.nlp.pipe_names:
            try:
                from negspacy.negation import Negex
                self.nlp.add_pipe("negex", config={"chunk_prefix": ["no"]})
                logger.info("Added negex pipe to spaCy pipeline")
            except ImportError:
                logger.warning(
                    "negspacy not installed. Falling back to pattern-based negation. "
                    "Install with: pip install negspacy"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize negex: {e}")
    
    def check_negation(
        self,
        doc: Doc,
        source_text: str,
        target_text: str,
        trigger_token: Optional[Token] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a relation is negated.
        
        Args:
            doc: Processed spaCy Doc
            source_text: Source entity text
            target_text: Target entity text
            trigger_token: The verb/trigger token for the relation
            
        Returns:
            Tuple of (is_negated, negation_cue)
        """
        # Strategy 1: Check negex entity annotations
        if "negex" in self.nlp.pipe_names:
            for ent in doc.ents:
                if hasattr(ent, "_.negex") and ent._.negex:
                    # Check if negated entity overlaps with source or target
                    ent_text_lower = ent.text.lower()
                    if (source_text.lower() in ent_text_lower or 
                        target_text.lower() in ent_text_lower or
                        ent_text_lower in source_text.lower() or
                        ent_text_lower in target_text.lower()):
                        return True, "negex"
        
        # Strategy 2: Check for negation in trigger verb's dependencies
        if trigger_token:
            for child in trigger_token.children:
                if child.dep_ == "neg":
                    return True, child.text
            
            # Check ancestors for negation
            for ancestor in trigger_token.ancestors:
                for child in ancestor.children:
                    if child.dep_ == "neg":
                        return True, child.text
        
        # Strategy 3: Pattern matching in sentence context
        sentence = trigger_token.sent.text.lower() if trigger_token else doc.text.lower()
        
        for cue in self.negation_cues:
            if cue in sentence:
                # Verify negation applies to this relation (not some other clause)
                cue_pos = sentence.find(cue)
                source_pos = sentence.find(source_text.lower())
                target_pos = sentence.find(target_text.lower())
                
                # Negation should appear before or near the relation elements
                if cue_pos < max(source_pos, target_pos) + 50:
                    return True, cue
        
        return False, None


# =============================================================================
# Coreference Resolver
# =============================================================================

class CoreferenceResolver:
    """
    Resolves pronouns to their antecedents using fastcoref.
    
    Transforms:
        "The middle cerebral artery is critical. It supplies the lateral cortex."
    Into:
        "The middle cerebral artery is critical. The middle cerebral artery supplies the lateral cortex."
    """
    
    PRONOUNS = {"it", "its", "they", "their", "this", "these", "which", "that"}
    
    def __init__(self, config: RelationExtractionConfig):
        """
        Initialize coreference resolver.
        
        Args:
            config: Extraction configuration
        """
        self.config = config
        self._model = None  # Lazy initialization
        self._available = None
    
    def _lazy_init(self) -> bool:
        """
        Lazily initialize the coreference model.
        
        Returns:
            True if model loaded successfully
        """
        if self._available is not None:
            return self._available
        
        try:
            from fastcoref import FCoref
            self._model = FCoref(device="cpu")
            self._available = True
            logger.info("Loaded fastcoref model")
        except ImportError:
            logger.warning(
                "fastcoref not installed. Coreference resolution disabled. "
                "Install with: pip install fastcoref"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to load fastcoref: {e}")
            self._available = False
        
        return self._available
    
    def resolve(self, text: str) -> str:
        """
        Replace pronouns with their resolved antecedents.
        
        Args:
            text: Input text with potential pronouns
            
        Returns:
            Text with pronouns replaced by antecedents
        """
        if not self.config.enable_coreference:
            return text
        
        if not self._lazy_init():
            return text
        
        # Truncate very long texts
        if len(text) > self.config.coref_max_length:
            logger.debug(f"Text too long for coref ({len(text)} chars), truncating")
            text = text[:self.config.coref_max_length]
        
        try:
            preds = self._model.predict(texts=[text])
            
            if not preds or not preds[0].get_clusters():
                return text
            
            # Build replacement map
            replacements = []
            for cluster in preds[0].get_clusters():
                if len(cluster) < 2:
                    continue
                
                # First mention is typically the antecedent
                antecedent = cluster[0]
                antecedent_text = text[antecedent[0]:antecedent[1]]
                
                for mention in cluster[1:]:
                    mention_text = text[mention[0]:mention[1]].lower()
                    
                    # Only replace pronouns
                    if mention_text in self.PRONOUNS:
                        replacements.append((mention[0], mention[1], antecedent_text))
            
            # Apply replacements in reverse order to preserve offsets
            for start, end, replacement in sorted(replacements, reverse=True):
                text = text[:start] + replacement + text[end:]
            
            return text
            
        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}")
            return text


# =============================================================================
# Entity-First Extractor
# =============================================================================

class EntityFirstExtractor:
    """
    Extract relations by first finding entities, then classifying pairs.
    
    Workflow:
    1. Extract UMLS entities from text
    2. Pair entities within proximity threshold
    3. Classify relation type from context between entities
    
    Benefits:
    - Catches implicit relations without explicit trigger verbs
    - Better handles complex sentence structures
    - Leverages UMLS semantic types for relation inference
    """
    
    # Context patterns for relation classification
    CONTEXT_PATTERNS = {
        RelationType.SUPPLIES: [
            r"suppl\w*", r"perfus\w*", r"vasculariz\w*", r"blood\s+to",
            r"feeds?", r"nourish\w*",
        ],
        RelationType.DRAINS_TO: [
            r"drain\w*", r"empties?\s+into", r"flows?\s+to", r"returns?\s+to",
        ],
        RelationType.BRANCHES_FROM: [
            r"branch\w*\s+(?:from|of)", r"originat\w*", r"aris\w*\s+from",
            r"derives?\s+from", r"comes?\s+from",
        ],
        RelationType.INNERVATES: [
            r"innervat\w*", r"nerve\s+(?:to|supply)", r"motor\s+to",
            r"sensory\s+(?:to|from)",
        ],
        RelationType.TRAVERSES: [
            r"travers\w*", r"pass\w*\s+through", r"cross\w*", r"runs?\s+through",
            r"courses?\s+through",
        ],
        RelationType.ADJACENT_TO: [
            r"adjacent\s+to", r"next\s+to", r"beside", r"lateral\s+to",
            r"medial\s+to", r"near",
        ],
        RelationType.CONTAINED_IN: [
            r"(?:with)?in", r"inside", r"contains?", r"houses?",
            r"located\s+in",
        ],
        RelationType.CAUSES: [
            r"caus\w*", r"results?\s+in", r"leads?\s+to", r"produc\w*",
            r"induces?",
        ],
        RelationType.TREATS: [
            r"treats?", r"therap\w*", r"manag\w*", r"indicated\s+for",
        ],
    }
    
    def __init__(
        self, 
        umls_extractor: Optional[Any] = None,
        config: Optional[RelationExtractionConfig] = None,
    ):
        """
        Initialize entity-first extractor.
        
        Args:
            umls_extractor: UMLS entity extractor instance
            config: Extraction configuration
        """
        self.umls = umls_extractor
        self.config = config or RelationExtractionConfig()
        
        # Compile patterns
        self._compiled_patterns = {
            rel_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rel_type, patterns in self.CONTEXT_PATTERNS.items()
        }
    
    def extract_entity_pairs(
        self, 
        text: str,
    ) -> List[Tuple[Any, Any, str, int, int]]:
        """
        Find entity pairs within proximity threshold.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity1, entity2, context, start_idx, end_idx) tuples
        """
        if not self.umls:
            return []
        
        try:
            entities = self.umls.extract(text)
        except Exception as e:
            logger.warning(f"UMLS extraction failed: {e}")
            return []
        
        if not entities:
            return []
        
        # Sort by position
        entities.sort(key=lambda e: e.start_char if hasattr(e, 'start_char') else 0)
        
        pairs = []
        max_dist = self.config.entity_pair_max_distance
        
        for i, e1 in enumerate(entities):
            e1_end = getattr(e1, 'end_char', 0)
            
            for e2 in entities[i + 1:]:
                e2_start = getattr(e2, 'start_char', 0)
                
                # Check distance
                distance = e2_start - e1_end
                if distance > max_dist:
                    break  # Sorted, so no need to check further
                
                if distance < 0:
                    continue  # Overlapping entities
                
                # Extract context between entities
                context = text[e1_end:e2_start]
                pairs.append((e1, e2, context, e1_end, e2_start))
        
        return pairs
    
    def classify_pair(
        self,
        entity1: Any,
        entity2: Any,
        context: str,
        doc: Optional[Doc] = None,
    ) -> Optional[Tuple[RelationType, float]]:
        """
        Classify relation type between entity pair.
        
        Args:
            entity1: First entity
            entity2: Second entity
            context: Text between entities
            doc: Optional spaCy doc for additional analysis
            
        Returns:
            (RelationType, confidence) or None if no relation detected
        """
        context_lower = context.lower()
        
        best_match = None
        best_confidence = 0.0
        
        for rel_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(context_lower):
                    # Calculate confidence based on match quality
                    confidence = 0.65  # Base confidence for pattern match
                    
                    # Boost for shorter context (more direct relationship)
                    if len(context) < 50:
                        confidence += 0.1
                    
                    # Boost if semantic types align with relation
                    confidence += self._semantic_type_boost(
                        entity1, entity2, rel_type
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = rel_type
        
        if best_match:
            return best_match, min(1.0, best_confidence)
        
        return None
    
    def _semantic_type_boost(
        self,
        entity1: Any,
        entity2: Any,
        relation: RelationType,
    ) -> float:
        """
        Calculate confidence boost based on semantic type compatibility.
        
        Args:
            entity1: First entity (with semantic_type attribute)
            entity2: Second entity
            relation: Proposed relation type
            
        Returns:
            Confidence boost (0.0 to 0.15)
        """
        # Define expected semantic type pairs for relations
        EXPECTED_TYPES = {
            RelationType.SUPPLIES: [
                ("T023", "T023"),  # Body Part → Body Part
                ("T023", "T029"),  # Body Part → Body Location
            ],
            RelationType.INNERVATES: [
                ("T023", "T023"),  # Nerve → Body Part
            ],
            RelationType.CAUSES: [
                ("T047", "T047"),  # Disease → Disease
                ("T047", "T184"),  # Disease → Sign/Symptom
            ],
            RelationType.TREATS: [
                ("T121", "T047"),  # Drug → Disease
                ("T061", "T047"),  # Procedure → Disease
            ],
        }
        
        if relation not in EXPECTED_TYPES:
            return 0.0
        
        e1_type = getattr(entity1, 'semantic_type', None)
        e2_type = getattr(entity2, 'semantic_type', None)
        
        if not e1_type or not e2_type:
            return 0.0
        
        expected = EXPECTED_TYPES[relation]
        if (e1_type, e2_type) in expected or (e2_type, e1_type) in expected:
            return 0.15
        
        return 0.0
    
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[ExtractedRelation]:
        """
        Extract relations using entity-first strategy.
        
        Args:
            text: Input text
            doc: Optional pre-processed spaCy Doc
            
        Returns:
            List of extracted relations
        """
        relations = []
        pairs = self.extract_entity_pairs(text)
        
        for e1, e2, context, start_idx, end_idx in pairs:
            result = self.classify_pair(e1, e2, context, doc)
            
            if result:
                rel_type, confidence = result
                
                # Get normalized names
                e1_norm = getattr(e1, 'normalized', str(e1))
                e2_norm = getattr(e2, 'normalized', str(e2))
                e1_text = getattr(e1, 'text', str(e1))
                e2_text = getattr(e2, 'text', str(e2))
                
                relation = ExtractedRelation(
                    source=e1_text,
                    target=e2_text,
                    relation=rel_type,
                    confidence=confidence,
                    context_snippet=f"...{context[:150]}..." if len(context) > 150 else context,
                    source_normalized=e1_norm,
                    target_normalized=e2_norm,
                    extraction_method=ExtractionMethod.ENTITY_FIRST.value,
                )
                relations.append(relation)
        
        return relations


# =============================================================================
# Tiered LLM Verifier
# =============================================================================

class TieredLLMVerifier:
    """
    Route extractions through LLM based on confidence tiers.
    
    Tiers:
    - conf >= 0.9: Pass through (high confidence, no LLM needed)
    - 0.5 <= conf < 0.9: LLM verification (confirm/reject/modify)
    - conf < 0.5 or complex sentence: LLM completion (full extraction)
    """
    
    # Indicators of complex sentence structure requiring LLM
    COMPLEX_INDICATORS = [
        "however", "whereas", "although", "except", "unless",
        "despite", "notwithstanding", "while", "but",
    ]
    
    VERIFY_PROMPT = """Verify these relations extracted from neurosurgical text.
For each relation, respond with: CORRECT, INCORRECT, or MODIFY (with correction).

Text: {text}

Relations to verify:
{relations}

Format response as JSON array:
[{{"relation_idx": 0, "verdict": "CORRECT|INCORRECT|MODIFY", "correction": null|{{"source": "", "target": "", "relation": ""}}}}]"""
    
    COMPLETE_PROMPT = """Extract anatomical and clinical relations from this neurosurgical text.
Return JSON array of relations.

Text: {text}

Valid relation types: {relation_types}

Format:
[{{"source": "entity1", "target": "entity2", "relation": "relation_type", "confidence": 0.0-1.0}}]

Only extract clear, explicit relationships. Return empty array [] if none found."""
    
    def __init__(
        self,
        llm_client: Any,
        config: RelationExtractionConfig,
    ):
        """
        Initialize LLM verifier.
        
        Args:
            llm_client: Anthropic or compatible LLM client
            config: Extraction configuration
        """
        self.llm_client = llm_client
        self.config = config
    
    def _is_complex(self, text: str) -> bool:
        """Check if text has complex structure requiring LLM."""
        text_lower = text.lower()
        return any(ind in text_lower for ind in self.COMPLEX_INDICATORS)
    
    async def process(
        self,
        relations: List[ExtractedRelation],
        text: str,
    ) -> List[ExtractedRelation]:
        """
        Process relations through tiered LLM verification.
        
        Args:
            relations: Initial extractions from spaCy
            text: Original text
            
        Returns:
            Verified/completed relations
        """
        if not self.llm_client:
            return relations
        
        # Tier 1: High confidence - pass through
        high_conf = [
            r for r in relations 
            if r.confidence >= self.config.llm_verify_threshold
        ]
        
        # Tier 2: Medium confidence - verify
        to_verify = [
            r for r in relations
            if self.config.llm_complete_threshold <= r.confidence < self.config.llm_verify_threshold
        ]
        
        # Tier 3: Low confidence or complex - LLM complete
        low_conf = [
            r for r in relations
            if r.confidence < self.config.llm_complete_threshold
        ]
        
        results = list(high_conf)
        
        # Verify medium confidence
        if to_verify:
            verified = await self._batch_verify(to_verify, text)
            results.extend(verified)
        
        # Complete if complex or no extractions
        if self._is_complex(text) or (not relations and self._likely_has_relations(text)):
            completed = await self._llm_complete(text)
            results.extend(completed)
        
        return results
    
    def _likely_has_relations(self, text: str) -> bool:
        """Check if text likely contains extractable relations."""
        indicators = [
            "supplies", "drains", "innervates", "causes", "treats",
            "branches", "traverses", "adjacent", "projects",
            "artery", "nerve", "lobe", "nucleus", "cortex",
        ]
        text_lower = text.lower()
        return sum(1 for ind in indicators if ind in text_lower) >= 2
    
    async def _batch_verify(
        self,
        relations: List[ExtractedRelation],
        text: str,
    ) -> List[ExtractedRelation]:
        """Verify relations in batch."""
        if not relations:
            return []
        
        # Format relations for prompt
        relations_text = "\n".join(
            f"{i}. {r.source} --[{r.relation.value}]--> {r.target} (conf: {r.confidence:.2f})"
            for i, r in enumerate(relations)
        )
        
        prompt = self.VERIFY_PROMPT.format(
            text=text[:1000],  # Truncate long text
            relations=relations_text,
        )
        
        try:
            response = await self._call_llm(prompt)
            verdicts = self._parse_verify_response(response)
            
            verified = []
            for i, relation in enumerate(relations):
                verdict = verdicts.get(i, {"verdict": "CORRECT"})
                
                if verdict["verdict"] == "CORRECT":
                    relation.extraction_method = ExtractionMethod.LLM_VERIFIED.value
                    verified.append(relation)
                elif verdict["verdict"] == "MODIFY" and verdict.get("correction"):
                    # Apply correction
                    corr = verdict["correction"]
                    relation.source = corr.get("source", relation.source)
                    relation.target = corr.get("target", relation.target)
                    if corr.get("relation"):
                        try:
                            relation.relation = RelationType(corr["relation"])
                        except ValueError:
                            pass
                    relation.extraction_method = ExtractionMethod.LLM_VERIFIED.value
                    verified.append(relation)
                # INCORRECT: drop the relation
            
            return verified
            
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            # Fall back to returning unverified
            return relations
    
    async def _llm_complete(self, text: str) -> List[ExtractedRelation]:
        """Use LLM to extract relations from scratch."""
        prompt = self.COMPLETE_PROMPT.format(
            text=text[:1500],
            relation_types=[r.value for r in RelationType],
        )
        
        try:
            response = await self._call_llm(prompt)
            return self._parse_complete_response(response, text)
        except Exception as e:
            logger.warning(f"LLM completion failed: {e}")
            return []
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text."""
        try:
            # Anthropic client
            response = await self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_verify_response(self, response: str) -> Dict[int, dict]:
        """Parse verification response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    item["relation_idx"]: item
                    for item in data
                    if isinstance(item, dict) and "relation_idx" in item
                }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse verify response: {e}")
        
        return {}
    
    def _parse_complete_response(
        self, 
        response: str, 
        original_text: str,
    ) -> List[ExtractedRelation]:
        """Parse completion response into relations."""
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            relations = []
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                rel_type_str = item.get("relation", "")
                try:
                    rel_type = RelationType(rel_type_str)
                except ValueError:
                    continue
                
                relations.append(ExtractedRelation(
                    source=item.get("source", ""),
                    target=item.get("target", ""),
                    relation=rel_type,
                    confidence=float(item.get("confidence", 0.7)),
                    context_snippet=original_text[:200],
                    source_normalized=item.get("source", "").lower(),
                    target_normalized=item.get("target", "").lower(),
                    extraction_method=ExtractionMethod.LLM_COMPLETE.value,
                ))
            
            return relations
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse complete response: {e}")
            return []


# =============================================================================
# Main Extractor
# =============================================================================

class NeuroRelationExtractor:
    """
    Production-grade relation extractor for neurosurgical text.
    
    Multi-strategy extraction with:
    - spaCy dependency parsing (primary)
    - Coordination handling for conjunctions
    - Negation detection
    - Entity-first NER extraction
    - Tiered LLM verification
    - Coreference resolution
    """
    
    # Relation trigger lemmas mapped to (relation_type, subject_is_source)
    TRIGGER_MAP: Dict[str, Tuple[RelationType, bool]] = {
        # Vascular
        "supply": (RelationType.SUPPLIES, True),
        "perfuse": (RelationType.SUPPLIES, True),
        "vascularize": (RelationType.SUPPLIES, True),
        "drain": (RelationType.DRAINS_TO, True),
        "empty": (RelationType.DRAINS_TO, True),
        "branch": (RelationType.BRANCHES_FROM, True),
        "originate": (RelationType.BRANCHES_FROM, True),
        "arise": (RelationType.BRANCHES_FROM, True),
        "anastomose": (RelationType.ANASTOMOSES_WITH, True),
        
        # Neural
        "innervate": (RelationType.INNERVATES, True),
        "project": (RelationType.PROJECTS_TO, True),
        "synapse": (RelationType.PROJECTS_TO, True),
        "receive": (RelationType.RECEIVES_FROM, True),
        
        # Spatial
        "contain": (RelationType.CONTAINED_IN, False),
        "house": (RelationType.CONTAINED_IN, False),
        "traverse": (RelationType.TRAVERSES, True),
        "cross": (RelationType.TRAVERSES, True),
        "pass": (RelationType.TRAVERSES, True),
        "border": (RelationType.ADJACENT_TO, True),
        "abut": (RelationType.ADJACENT_TO, True),
        "adjoin": (RelationType.ADJACENT_TO, True),
        
        # Clinical
        "cause": (RelationType.CAUSES, True),
        "result": (RelationType.CAUSES, False),
        "produce": (RelationType.CAUSES, True),
        "lead": (RelationType.CAUSES, True),
        "treat": (RelationType.TREATS, True),
        "manage": (RelationType.TREATS, True),
        "indicate": (RelationType.INDICATES, True),
        "suggest": (RelationType.INDICATES, True),
        "contraindicate": (RelationType.CONTRAINDICATED_FOR, True),
    }
    
    SYMMETRIC_RELATIONS: Set[RelationType] = {
        RelationType.ADJACENT_TO,
        RelationType.ANASTOMOSES_WITH,
    }
    
    def __init__(
        self,
        model: str = "en_core_web_lg",
        config: Optional[RelationExtractionConfig] = None,
        llm_client: Optional[Any] = None,
        umls_extractor: Optional[Any] = None,
    ):
        """
        Initialize the extractor.
        
        Args:
            model: spaCy model name
            config: Extraction configuration
            llm_client: Optional LLM client for tiered verification
            umls_extractor: Optional UMLS extractor for entity-first strategy
        """
        self.config = config or RelationExtractionConfig()
        self.llm_client = llm_client
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.info(f"Downloading spaCy model: {model}")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
        
        # Build abbreviation pattern
        self._abbrev_pattern = self._build_abbrev_pattern()
        
        # Initialize enhancement components
        self.negation_detector = None
        if self.config.enable_negation:
            self.negation_detector = NegationDetector(self.nlp, self.config)
        
        self.coref_resolver = None
        if self.config.enable_coreference:
            self.coref_resolver = CoreferenceResolver(self.config)
        
        self.entity_first_extractor = None
        if self.config.enable_entity_first_ner and umls_extractor:
            self.entity_first_extractor = EntityFirstExtractor(
                umls_extractor, self.config
            )
        
        self.llm_verifier = None
        if self.config.enable_tiered_llm and llm_client:
            self.llm_verifier = TieredLLMVerifier(llm_client, self.config)
    
    def _build_abbrev_pattern(self) -> re.Pattern:
        """Build regex pattern for abbreviation detection."""
        abbrevs = sorted(NEURO_ABBREVIATIONS.keys(), key=len, reverse=True)
        pattern = r'\b(' + '|'.join(re.escape(a) for a in abbrevs) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def normalize_entity(self, text: str) -> str:
        """
        Normalize an entity string.
        
        - Expands abbreviations
        - Lowercases
        - Strips articles and extra whitespace
        """
        text = text.lower().strip()
        
        # Remove leading articles
        text = re.sub(r'^(the|a|an)\s+', '', text)
        
        # Expand abbreviations
        def replace_abbrev(match: re.Match) -> str:
            return NEURO_ABBREVIATIONS.get(match.group(1).lower(), match.group(1))
        
        text = self._abbrev_pattern.sub(replace_abbrev, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_noun_chunk_for_token(self, token: Token, doc: Doc) -> Optional[Span]:
        """Get the noun chunk containing a token."""
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk
        return None
    
    def _walk_conjunctions(self, token: Token, doc: Doc) -> List[str]:
        """
        Walk conjunction dependencies to find all coordinated entities.
        
        "frontal and temporal lobes" → ["frontal lobes", "temporal lobes"]
        "A, B, and C" → ["A", "B", "C"]
        """
        entities = []
        
        # Get the entity text for this token
        chunk = self._get_noun_chunk_for_token(token, doc)
        if chunk:
            entities.append(chunk.text)
        else:
            entities.append(" ".join([t.text for t in token.subtree]))
        
        # Walk conjunctions
        for child in token.children:
            if child.dep_ == "conj":
                # Get entity for conjunction
                child_chunk = self._get_noun_chunk_for_token(child, doc)
                if child_chunk:
                    entities.append(child_chunk.text)
                else:
                    entities.append(" ".join([t.text for t in child.subtree]))
                
                # Recurse for chains (A, B, and C)
                conj_entities = self._walk_conjunctions(child, doc)
                # Skip first as we already added it
                entities.extend(conj_entities[1:])
        
        return entities
    
    def _find_subject_object(
        self,
        trigger_token: Token,
        doc: Doc,
    ) -> Tuple[List[str], List[str]]:
        """
        Find subject and object noun phrases for a trigger verb.
        
        With coordination enabled, returns lists of all coordinated subjects/objects.
        
        Returns:
            (subjects, objects) - lists of entity strings
        """
        subjects = []
        objects = []
        
        # Find subject (nsubj, nsubjpass)
        for child in trigger_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                if self.config.enable_coordination:
                    subjects.extend(self._walk_conjunctions(child, doc))
                else:
                    chunk = self._get_noun_chunk_for_token(child, doc)
                    if chunk:
                        subjects.append(chunk.text)
                    else:
                        subjects.append(" ".join([t.text for t in child.subtree]))
                break
        
        # Find object (dobj, pobj via prep)
        for child in trigger_token.children:
            if child.dep_ == "dobj":
                if self.config.enable_coordination:
                    objects.extend(self._walk_conjunctions(child, doc))
                else:
                    chunk = self._get_noun_chunk_for_token(child, doc)
                    if chunk:
                        objects.append(chunk.text)
                    else:
                        objects.append(" ".join([t.text for t in child.subtree]))
                break
            
            elif child.dep_ == "prep":
                # Handle prepositional objects: "drains into X"
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        if self.config.enable_coordination:
                            objects.extend(self._walk_conjunctions(pobj, doc))
                        else:
                            chunk = self._get_noun_chunk_for_token(pobj, doc)
                            if chunk:
                                objects.append(chunk.text)
                            else:
                                objects.append(" ".join([t.text for t in pobj.subtree]))
                        break
        
        # Handle passive constructions: "X is supplied by Y"
        if trigger_token.tag_ == "VBN":  # Past participle (passive)
            for child in trigger_token.children:
                if child.dep_ == "agent":  # "by" phrase
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            # In passive, agent is the true subject
                            agent_chunk = self._get_noun_chunk_for_token(pobj, doc)
                            if agent_chunk:
                                # Swap: subjects become objects, agent becomes subject
                                objects = subjects
                                subjects = [agent_chunk.text]
                            break
        
        return subjects, objects
    
    def _get_sentence_context(self, token: Token, max_len: int = 200) -> str:
        """Get the sentence containing the token as context."""
        sent = token.sent
        text = sent.text.strip()
        
        if len(text) > max_len:
            # Truncate around the trigger word
            start = max(0, token.idx - sent.start_char - max_len // 2)
            end = min(len(text), start + max_len)
            text = "..." + text[start:end] + "..."
        
        return text
    
    def _calculate_confidence(
        self, 
        trigger: Token, 
        subj: str, 
        obj: str,
    ) -> float:
        """Calculate extraction confidence based on linguistic features."""
        confidence = 0.7  # Base confidence for spaCy extraction
        
        # Boost for clear dependency structure
        if trigger.dep_ == "ROOT":
            confidence += 0.1
        
        # Boost for longer, more specific noun phrases
        if len(subj.split()) > 1:
            confidence += 0.05
        if len(obj.split()) > 1:
            confidence += 0.05
        
        # Penalty for very short entities (likely extraction errors)
        if len(subj) < 3 or len(obj) < 3:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _dependency_extract(self, text: str, doc: Doc) -> List[ExtractedRelation]:
        """
        Extract relations using dependency parsing.
        
        This is the primary extraction strategy.
        """
        relations = []
        seen_hashes = set()
        
        for token in doc:
            # Check if token is a relation trigger
            if token.lemma_.lower() not in self.TRIGGER_MAP:
                continue
            
            relation_type, subject_is_source = self.TRIGGER_MAP[token.lemma_.lower()]
            
            # Extract subjects and objects (with coordination)
            subjects, objects = self._find_subject_object(token, doc)
            
            if not subjects or not objects:
                continue
            
            # Create Cartesian product of subjects × objects
            for subj in subjects:
                for obj in objects:
                    # Normalize entities
                    subj_norm = self.normalize_entity(subj)
                    obj_norm = self.normalize_entity(obj)
                    
                    # Skip if same entity
                    if subj_norm == obj_norm:
                        continue
                    
                    # Determine source and target based on relation semantics
                    if subject_is_source:
                        source, target = subj, obj
                        source_norm, target_norm = subj_norm, obj_norm
                    else:
                        source, target = obj, subj
                        source_norm, target_norm = obj_norm, subj_norm
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(token, subj, obj)
                    
                    if confidence < self.config.min_confidence:
                        continue
                    
                    # Check negation
                    is_negated = False
                    negation_cue = None
                    if self.config.enable_negation and self.negation_detector:
                        is_negated, negation_cue = self.negation_detector.check_negation(
                            doc, source, target, token
                        )
                    
                    relation = ExtractedRelation(
                        source=source,
                        target=target,
                        relation=relation_type,
                        confidence=confidence,
                        context_snippet=self._get_sentence_context(token),
                        source_normalized=source_norm,
                        target_normalized=target_norm,
                        bidirectional=relation_type in self.SYMMETRIC_RELATIONS,
                        is_negated=is_negated,
                        negation_cue=negation_cue,
                        extraction_method=ExtractionMethod.DEPENDENCY.value,
                    )
                    
                    # Deduplicate
                    if relation.hash_id not in seen_hashes:
                        seen_hashes.add(relation.hash_id)
                        relations.append(relation)
        
        return relations
    
    def _deduplicate(
        self, 
        relations: List[ExtractedRelation],
    ) -> List[ExtractedRelation]:
        """
        Deduplicate relations, keeping highest confidence version.
        """
        seen: Dict[str, ExtractedRelation] = {}
        
        for rel in relations:
            if rel.hash_id in seen:
                # Keep higher confidence
                if rel.confidence > seen[rel.hash_id].confidence:
                    seen[rel.hash_id] = rel
            else:
                seen[rel.hash_id] = rel
        
        return list(seen.values())
    
    def extract_from_text(self, text: str) -> List[ExtractedRelation]:
        """
        Extract relations from text using all enabled strategies.
        
        Args:
            text: Input text (can be multiple sentences)
            
        Returns:
            List of extracted relations
        """
        # Step 0: Coreference resolution (if enabled)
        if self.config.enable_coreference and self.coref_resolver:
            text = self.coref_resolver.resolve(text)
        
        # Process with spaCy
        doc = self.nlp(text)
        relations = []
        
        # Strategy 1: Dependency parsing (always enabled)
        dep_relations = self._dependency_extract(text, doc)
        relations.extend(dep_relations)
        
        # Strategy 2: Entity-first extraction (if enabled)
        if self.config.enable_entity_first_ner and self.entity_first_extractor:
            ef_relations = self.entity_first_extractor.extract(text, doc)
            relations.extend(ef_relations)
        
        # Deduplicate
        relations = self._deduplicate(relations)
        
        return relations
    
    async def extract_with_llm_verification(
        self,
        text: str,
        chunk_id: Optional[str] = None,
    ) -> List[ExtractedRelation]:
        """
        Extract relations with optional LLM verification.
        
        Uses tiered LLM verification if enabled and LLM client available.
        
        Args:
            text: Input text
            chunk_id: Optional chunk ID for tracking
            
        Returns:
            List of extracted (and optionally verified) relations
        """
        # Get initial extractions
        relations = self.extract_from_text(text)
        
        # Apply LLM verification if enabled
        if self.config.enable_tiered_llm and self.llm_verifier:
            relations = await self.llm_verifier.process(relations, text)
        
        return relations
    
    def extract_taxonomy_relations(self, entity: str) -> List[ExtractedRelation]:
        """
        Generate is_a relations from the taxonomy.
        
        Args:
            entity: Entity to look up in taxonomy
            
        Returns:
            List of is_a relations
        """
        entity_norm = self.normalize_entity(entity)
        relations = []
        
        if entity_norm in TAXONOMY:
            for parent in TAXONOMY[entity_norm]:
                relations.append(ExtractedRelation(
                    source=entity_norm,
                    target=parent,
                    relation=RelationType.IS_A,
                    confidence=1.0,  # Taxonomy is ground truth
                    context_snippet=f"Taxonomic relation: {entity_norm} is a type of {parent}",
                    source_normalized=entity_norm,
                    target_normalized=self.normalize_entity(parent),
                    bidirectional=False,
                    extraction_method=ExtractionMethod.TAXONOMY.value,
                ))
        
        return relations


# =============================================================================
# Utility Functions
# =============================================================================

def build_graph_from_relations(
    relations: List[ExtractedRelation],
) -> dict:
    """
    Convert extracted relations to a graph structure.
    
    Returns:
        Dict with 'nodes' and 'edges' ready for graph database insertion
    """
    nodes: Dict[str, dict] = {}
    edges: List[dict] = []
    
    for rel in relations:
        # Add source node
        if rel.source_normalized not in nodes:
            nodes[rel.source_normalized] = {
                "id": rel.source_normalized,
                "label": rel.source_normalized,
                "aliases": {rel.source},
            }
        else:
            nodes[rel.source_normalized]["aliases"].add(rel.source)
        
        # Add target node
        if rel.target_normalized not in nodes:
            nodes[rel.target_normalized] = {
                "id": rel.target_normalized,
                "label": rel.target_normalized,
                "aliases": {rel.target},
            }
        else:
            nodes[rel.target_normalized]["aliases"].add(rel.target)
        
        # Add edge
        edges.append({
            "source": rel.source_normalized,
            "target": rel.target_normalized,
            "relation": rel.relation.value,
            "confidence": rel.confidence,
            "context": rel.context_snippet,
            "bidirectional": rel.bidirectional,
            "is_negated": rel.is_negated,
            "negation_cue": rel.negation_cue,
            "extraction_method": rel.extraction_method,
        })
    
    # Convert sets to lists for JSON serialization
    for node in nodes.values():
        node["aliases"] = list(node["aliases"])
    
    return {"nodes": list(nodes.values()), "edges": edges}


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "NeuroRelationExtractor",
    "ExtractedRelation",
    "RelationType",
    "ExtractionMethod",
    "NegationDetector",
    "CoreferenceResolver",
    "EntityFirstExtractor",
    "TieredLLMVerifier",
    "NEURO_ABBREVIATIONS",
    "TAXONOMY",
    "build_graph_from_relations",
]
