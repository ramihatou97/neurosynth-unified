"""
NeuroSynth Relation Extractor
Production-grade entity relation extraction for neurosurgical knowledge graphs.

Replaces fragile regex with spaCy NLP + LLM fallback for complex sentences.
"""

import re
import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

import spacy
from spacy.tokens import Doc, Span


class RelationType(str, Enum):
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


@dataclass
class ExtractedRelation:
    source: str
    target: str
    relation: RelationType
    confidence: float
    context_snippet: str
    source_normalized: Optional[str] = None
    target_normalized: Optional[str] = None
    bidirectional: bool = False
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "source_normalized": self.source_normalized or self.source,
            "target_normalized": self.target_normalized or self.target,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "context_snippet": self.context_snippet,
            "bidirectional": self.bidirectional,
        }
    
    @property
    def hash_id(self) -> str:
        """Unique identifier for deduplication."""
        key = f"{self.source_normalized}|{self.target_normalized}|{self.relation.value}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


# Neurosurgical abbreviation normalization map
NEURO_ABBREVIATIONS: dict[str, str] = {
    # Arteries
    "mca": "middle cerebral artery",
    "aca": "anterior cerebral artery",
    "pca": "posterior cerebral artery",
    "ica": "internal carotid artery",
    "eca": "external carotid artery",
    "cca": "common carotid artery",
    "acom": "anterior communicating artery",
    "pcom": "posterior communicating artery",
    "pica": "posterior inferior cerebellar artery",
    "aica": "anterior inferior cerebellar artery",
    "sca": "superior cerebellar artery",
    "ba": "basilar artery",
    "va": "vertebral artery",
    "lsa": "lenticulostriate arteries",
    
    # Veins/Sinuses
    "ssv": "superficial sylvian vein",
    "dsv": "deep sylvian vein",
    "smcv": "superficial middle cerebral vein",
    "vog": "vein of galen",
    "vol": "vein of labbe",
    "vot": "vein of trolard",
    "sss": "superior sagittal sinus",
    "iss": "inferior sagittal sinus",
    "ss": "sigmoid sinus",
    "ts": "transverse sinus",
    "cs": "cavernous sinus",
    
    # Structures
    "cn": "cranial nerve",
    "cn i": "olfactory nerve",
    "cn ii": "optic nerve",
    "cn iii": "oculomotor nerve",
    "cn iv": "trochlear nerve",
    "cn v": "trigeminal nerve",
    "cn vi": "abducens nerve",
    "cn vii": "facial nerve",
    "cn viii": "vestibulocochlear nerve",
    "cn ix": "glossopharyngeal nerve",
    "cn x": "vagus nerve",
    "cn xi": "accessory nerve",
    "cn xii": "hypoglossal nerve",
    
    # Brain regions
    "stg": "superior temporal gyrus",
    "mtg": "middle temporal gyrus",
    "itg": "inferior temporal gyrus",
    "sfg": "superior frontal gyrus",
    "mfg": "middle frontal gyrus",
    "ifg": "inferior frontal gyrus",
    "sma": "supplementary motor area",
    "m1": "primary motor cortex",
    "s1": "primary somatosensory cortex",
    "v1": "primary visual cortex",
    "a1": "primary auditory cortex",
    "pfc": "prefrontal cortex",
    "dlpfc": "dorsolateral prefrontal cortex",
    "ofc": "orbitofrontal cortex",
    "acc": "anterior cingulate cortex",
    "pcc": "posterior cingulate cortex",
    "tpj": "temporoparietal junction",
    
    # Deep structures
    "gp": "globus pallidus",
    "gpe": "globus pallidus externa",
    "gpi": "globus pallidus interna",
    "stn": "subthalamic nucleus",
    "snr": "substantia nigra pars reticulata",
    "snc": "substantia nigra pars compacta",
    "vpl": "ventral posterolateral nucleus",
    "vpm": "ventral posteromedial nucleus",
    "vim": "ventral intermediate nucleus",
    "vta": "ventral tegmental area",
    "lc": "locus coeruleus",
    "drn": "dorsal raphe nucleus",
    "pag": "periaqueductal gray",
    "rn": "red nucleus",
    "dn": "dentate nucleus",
    
    # Pathology
    "gbm": "glioblastoma",
    "hgg": "high grade glioma",
    "lgg": "low grade glioma",
    "who": "world health organization",
    "sah": "subarachnoid hemorrhage",
    "ich": "intracerebral hemorrhage",
    "sdh": "subdural hematoma",
    "edh": "epidural hematoma",
    "avm": "arteriovenous malformation",
    "davf": "dural arteriovenous fistula",
    "ccf": "carotid cavernous fistula",
    "tia": "transient ischemic attack",
    "cva": "cerebrovascular accident",
    
    # Procedures
    "evd": "external ventricular drain",
    "vps": "ventriculoperitoneal shunt",
    "dbs": "deep brain stimulation",
    "srs": "stereotactic radiosurgery",
    "gkrs": "gamma knife radiosurgery",
    "acdf": "anterior cervical discectomy and fusion",
    "pcdf": "posterior cervical decompression and fusion",
    "tlif": "transforaminal lumbar interbody fusion",
    "plif": "posterior lumbar interbody fusion",
    "alif": "anterior lumbar interbody fusion",
    "xlif": "extreme lateral interbody fusion",
    "olif": "oblique lumbar interbody fusion",
}

# Taxonomy definitions for is_a relations
TAXONOMY: dict[str, list[str]] = {
    # Tumors
    "tumor": ["neoplasm", "lesion"],
    "glioma": ["tumor", "glial tumor"],
    "glioblastoma": ["glioma", "high grade glioma", "who grade 4 glioma"],
    "astrocytoma": ["glioma"],
    "oligodendroglioma": ["glioma"],
    "meningioma": ["tumor", "extra-axial tumor"],
    "schwannoma": ["tumor", "nerve sheath tumor"],
    "vestibular schwannoma": ["schwannoma", "cerebellopontine angle tumor"],
    "pituitary adenoma": ["tumor", "sellar tumor"],
    "craniopharyngioma": ["tumor", "sellar tumor"],
    "medulloblastoma": ["tumor", "posterior fossa tumor", "embryonal tumor"],
    "hemangioblastoma": ["tumor", "vascular tumor"],
    "metastasis": ["tumor", "secondary tumor"],
    
    # Vascular
    "aneurysm": ["vascular lesion"],
    "arteriovenous malformation": ["vascular malformation", "vascular lesion"],
    "cavernoma": ["vascular malformation", "cavernous malformation"],
    "dural arteriovenous fistula": ["vascular malformation"],
    
    # Hemorrhage
    "subarachnoid hemorrhage": ["hemorrhage", "intracranial hemorrhage"],
    "intracerebral hemorrhage": ["hemorrhage", "intracranial hemorrhage"],
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


class NeuroRelationExtractor:
    """
    Production-grade relation extractor for neurosurgical text.
    
    Uses spaCy dependency parsing for accurate multi-word entity extraction,
    with optional LLM fallback for complex sentences.
    """
    
    # Relation trigger lemmas mapped to relation types
    TRIGGER_MAP: dict[str, tuple[RelationType, bool]] = {
        # (relation_type, subject_is_source)
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
        "contain": (RelationType.CONTAINED_IN, False),  # X contains Y -> Y contained_in X
        "house": (RelationType.CONTAINED_IN, False),
        "traverse": (RelationType.TRAVERSES, True),
        "cross": (RelationType.TRAVERSES, True),
        "pass": (RelationType.TRAVERSES, True),
        "border": (RelationType.ADJACENT_TO, True),
        "abut": (RelationType.ADJACENT_TO, True),
        "adjoin": (RelationType.ADJACENT_TO, True),
        
        # Clinical
        "cause": (RelationType.CAUSES, True),
        "result": (RelationType.CAUSES, False),  # X results from Y -> Y causes X
        "produce": (RelationType.CAUSES, True),
        "lead": (RelationType.CAUSES, True),
        "treat": (RelationType.TREATS, True),
        "manage": (RelationType.TREATS, True),
        "indicate": (RelationType.INDICATES, True),
        "suggest": (RelationType.INDICATES, True),
        "contraindicate": (RelationType.CONTRAINDICATED_FOR, True),
    }
    
    # Symmetric relations (bidirectional edges)
    SYMMETRIC_RELATIONS: set[RelationType] = {
        RelationType.ADJACENT_TO,
        RelationType.ANASTOMOSES_WITH,
    }
    
    def __init__(
        self,
        model: str = "en_core_web_lg",
        min_confidence: float = 0.5,
        llm_client: Optional[object] = None,  # Optional Anthropic/OpenAI client
    ):
        """
        Initialize the extractor.
        
        Args:
            model: spaCy model name (use 'en_core_web_lg' or 'en_core_sci_lg' for medical)
            min_confidence: Minimum confidence threshold for extracted relations
            llm_client: Optional LLM client for complex sentence fallback
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading spaCy model: {model}")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
        
        self.min_confidence = min_confidence
        self.llm_client = llm_client
        
        # Build reverse lookup for normalization
        self._abbrev_pattern = self._build_abbrev_pattern()
    
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
    
    def _get_noun_chunk_for_token(self, token, doc: Doc) -> Optional[Span]:
        """Get the noun chunk containing a token."""
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk
        return None
    
    def _find_subject_object(
        self, 
        trigger_token, 
        doc: Doc
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Find subject and object noun phrases for a trigger verb.
        
        Returns (subject_text, object_text)
        """
        subject = None
        obj = None
        
        # Find subject (nsubj, nsubjpass)
        for child in trigger_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                chunk = self._get_noun_chunk_for_token(child, doc)
                if chunk:
                    subject = chunk.text
                else:
                    # Fallback: get the subtree
                    subject = " ".join([t.text for t in child.subtree])
                break
        
        # Find object (dobj, pobj via prep)
        for child in trigger_token.children:
            if child.dep_ == "dobj":
                chunk = self._get_noun_chunk_for_token(child, doc)
                if chunk:
                    obj = chunk.text
                else:
                    obj = " ".join([t.text for t in child.subtree])
                break
            elif child.dep_ == "prep":
                # Handle prepositional objects: "drains into X"
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        chunk = self._get_noun_chunk_for_token(pobj, doc)
                        if chunk:
                            obj = chunk.text
                        else:
                            obj = " ".join([t.text for t in pobj.subtree])
                        break
        
        # Handle passive constructions: "X is supplied by Y"
        if trigger_token.tag_ == "VBN":  # Past participle (passive)
            for child in trigger_token.children:
                if child.dep_ == "agent":  # "by" phrase
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            chunk = self._get_noun_chunk_for_token(pobj, doc)
                            if chunk:
                                # Swap: in passive, agent is the true subject
                                obj = subject
                                subject = chunk.text
                            break
        
        return subject, obj
    
    def _get_sentence_context(self, token, max_len: int = 200) -> str:
        """Get the sentence containing the token as context."""
        sent = token.sent
        text = sent.text.strip()
        if len(text) > max_len:
            # Truncate around the trigger word
            start = max(0, token.idx - sent.start_char - max_len // 2)
            end = min(len(text), start + max_len)
            text = "..." + text[start:end] + "..."
        return text
    
    def extract_from_text(self, text: str) -> list[ExtractedRelation]:
        """
        Extract relations from text using spaCy dependency parsing.
        
        Args:
            text: Input text (can be multiple sentences)
            
        Returns:
            List of extracted relations
        """
        doc = self.nlp(text)
        relations: list[ExtractedRelation] = []
        seen_hashes: set[str] = set()
        
        for token in doc:
            # Check if token is a relation trigger
            if token.lemma_.lower() not in self.TRIGGER_MAP:
                continue
            
            relation_type, subject_is_source = self.TRIGGER_MAP[token.lemma_.lower()]
            
            # Extract subject and object
            subj, obj = self._find_subject_object(token, doc)
            
            if not subj or not obj:
                continue
            
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
            
            # Calculate confidence based on extraction quality
            confidence = self._calculate_confidence(token, subj, obj)
            
            if confidence < self.min_confidence:
                continue
            
            relation = ExtractedRelation(
                source=source,
                target=target,
                relation=relation_type,
                confidence=confidence,
                context_snippet=self._get_sentence_context(token),
                source_normalized=source_norm,
                target_normalized=target_norm,
                bidirectional=relation_type in self.SYMMETRIC_RELATIONS,
            )
            
            # Deduplicate
            if relation.hash_id not in seen_hashes:
                seen_hashes.add(relation.hash_id)
                relations.append(relation)
        
        return relations
    
    def _calculate_confidence(self, trigger, subj: str, obj: str) -> float:
        """
        Calculate extraction confidence based on linguistic features.
        """
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
        
        # Cap at 1.0
        return min(1.0, max(0.0, confidence))
    
    def extract_taxonomy_relations(self, entity: str) -> list[ExtractedRelation]:
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
                ))
        
        return relations
    
    async def extract_with_llm_fallback(
        self,
        text: str,
        chunk_id: Optional[str] = None,
    ) -> list[ExtractedRelation]:
        """
        Extract relations with LLM fallback for complex sentences.
        
        Uses spaCy first, then falls back to LLM for sentences with
        no extractions but likely relations.
        """
        # First try spaCy
        relations = self.extract_from_text(text)
        
        if relations or not self.llm_client:
            return relations
        
        # Check if text likely contains relations (has medical entities)
        doc = self.nlp(text)
        has_medical_entities = any(
            self.normalize_entity(chunk.text) in NEURO_ABBREVIATIONS.values()
            or any(term in chunk.text.lower() for term in ["artery", "nerve", "cortex", "nucleus"])
            for chunk in doc.noun_chunks
        )
        
        if not has_medical_entities:
            return relations
        
        # LLM extraction fallback
        llm_relations = await self._llm_extract(text)
        relations.extend(llm_relations)
        
        return relations
    
    async def _llm_extract(self, text: str) -> list[ExtractedRelation]:
        """
        Use LLM to extract relations from complex sentences.
        """
        if not self.llm_client:
            return []
        
        prompt = f"""Extract anatomical and clinical relationships from this neurosurgical text.

Text: {text}

Return JSON array of relations with format:
[{{"source": "entity1", "target": "entity2", "relation": "relation_type", "confidence": 0.0-1.0}}]

Valid relation types: {[r.value for r in RelationType]}

Only extract clear, explicit relationships. Return empty array if none found."""

        try:
            # Assuming Anthropic client
            response = await self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            content = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [
                    ExtractedRelation(
                        source=r["source"],
                        target=r["target"],
                        relation=RelationType(r["relation"]),
                        confidence=r.get("confidence", 0.7),
                        context_snippet=text[:200],
                        source_normalized=self.normalize_entity(r["source"]),
                        target_normalized=self.normalize_entity(r["target"]),
                    )
                    for r in data
                    if r.get("relation") in [rt.value for rt in RelationType]
                ]
        except Exception as e:
            print(f"LLM extraction failed: {e}")
        
        return []


def build_graph_from_relations(
    relations: list[ExtractedRelation],
) -> dict:
    """
    Convert extracted relations to a graph structure suitable for NetworkX or Neo4j.
    
    Returns:
        Dict with 'nodes' and 'edges' ready for graph database insertion
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    
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
        })
    
    # Convert sets to lists for JSON serialization
    for node in nodes.values():
        node["aliases"] = list(node["aliases"])
    
    return {"nodes": list(nodes.values()), "edges": edges}


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    # Initialize extractor
    extractor = NeuroRelationExtractor(model="en_core_web_lg")
    
    # Test sentences
    test_texts = [
        "The middle cerebral artery supplies the lateral surface of the cerebral hemisphere.",
        "The MCA supplies the insular cortex and the lateral frontal lobe.",
        "The superior sagittal sinus drains into the confluence of sinuses.",
        "The facial nerve traverses the internal acoustic meatus.",
        "Glioblastoma causes mass effect and surrounding edema.",
        "The subthalamic nucleus receives input from the motor cortex.",
        "The M2 segment branches from the M1 segment at the limen insulae.",
        "Temozolomide treats glioblastoma.",
        "The frontal lobe is adjacent to the parietal lobe.",
        "The thalamus is contained in the diencephalon.",
    ]
    
    print("=" * 80)
    print("NeuroSynth Relation Extractor - Test Results")
    print("=" * 80)
    
    all_relations = []
    
    for text in test_texts:
        print(f"\nInput: {text}")
        relations = extractor.extract_from_text(text)
        
        if relations:
            for rel in relations:
                print(f"  → {rel.source_normalized} --[{rel.relation.value}]--> {rel.target_normalized}")
                print(f"    (confidence: {rel.confidence:.2f})")
                all_relations.append(rel)
        else:
            print("  → No relations extracted")
    
    # Add taxonomy relations for extracted entities
    print("\n" + "=" * 80)
    print("Taxonomy Relations")
    print("=" * 80)
    
    entities_seen = set()
    for rel in all_relations:
        entities_seen.add(rel.source_normalized)
        entities_seen.add(rel.target_normalized)
    
    for entity in entities_seen:
        tax_rels = extractor.extract_taxonomy_relations(entity)
        for rel in tax_rels:
            print(f"  → {rel.source_normalized} --[is_a]--> {rel.target_normalized}")
    
    # Build graph structure
    print("\n" + "=" * 80)
    print("Graph Structure (JSON)")
    print("=" * 80)
    
    graph = build_graph_from_relations(all_relations)
    print(json.dumps(graph, indent=2))
