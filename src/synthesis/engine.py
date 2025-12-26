"""
NeuroSynth Synthesis Engine
Vendored and refactored to use Phase 1 SearchResult model.

Changes from original:
- Removed SourceChunk (uses SearchResult from src.shared.models)
- Updated ContextAdapter to access SearchResult fields directly
- Added RateLimiter for Claude API
- Added graceful Gemini verification degradation
"""

import asyncio
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.shared.models import SearchResult, ExtractedImage

logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for Claude API."""

    def __init__(self, calls_per_minute: int = 50):
        self.calls_per_minute = calls_per_minute
        self.window = 60.0
        self.timestamps: deque = deque()

    async def acquire(self):
        now = time()
        while self.timestamps and now - self.timestamps[0] > self.window:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.calls_per_minute:
            sleep_time = self.timestamps[0] + self.window - now + 0.1
            logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            return await self.acquire()

        self.timestamps.append(now)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class TemplateType(str, Enum):
    """Synthesis template types matching textbook styles."""
    PROCEDURAL = "PROCEDURAL"
    DISORDER = "DISORDER"
    ANATOMY = "ANATOMY"
    ENCYCLOPEDIA = "ENCYCLOPEDIA"


class AuthoritySource(str, Enum):
    """Source authority levels."""
    RHOTON = "RHOTON"
    YOUMANS = "YOUMANS"
    SCHMIDEK = "SCHMIDEK"
    GREENBERG = "GREENBERG"
    JOURNAL = "JOURNAL"
    TEXTBOOK = "TEXTBOOK"
    GENERAL = "GENERAL"


AUTHORITY_SCORES = {
    AuthoritySource.RHOTON: 0.95,
    AuthoritySource.YOUMANS: 0.90,
    AuthoritySource.SCHMIDEK: 0.88,
    AuthoritySource.GREENBERG: 0.85,
    AuthoritySource.JOURNAL: 0.80,
    AuthoritySource.TEXTBOOK: 0.75,
    AuthoritySource.GENERAL: 0.60,
}


@dataclass
class SynthesisSection:
    """A section of synthesized content."""
    title: str
    content: str
    level: int = 2
    sources: List[str] = field(default_factory=list)
    figures: List[Dict] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "sources": self.sources,
            "figures": self.figures,
            "word_count": self.word_count,
        }


@dataclass
class FigureRequest:
    """A figure placeholder to be resolved."""
    placeholder_id: str
    figure_type: str
    topic: str
    context: str
    resolved_id: Optional[str] = None
    resolved_path: Optional[str] = None
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placeholder_id": self.placeholder_id,
            "figure_type": self.figure_type,
            "topic": self.topic,
            "context": self.context,
            "resolved_id": self.resolved_id,
            "resolved_path": self.resolved_path,
            "caption": self.caption,
        }


@dataclass
class SynthesisResult:
    """Complete synthesis output."""
    topic: str
    template_type: TemplateType
    title: str
    abstract: str
    sections: List[SynthesisSection]
    references: List[Dict]
    figure_requests: List[FigureRequest] = field(default_factory=list)
    resolved_figures: List[Dict] = field(default_factory=list)
    
    total_words: int = 0
    total_figures: int = 0
    total_citations: int = 0
    synthesis_time_ms: int = 0
    
    verification_score: Optional[float] = None
    verification_issues: List[str] = field(default_factory=list)
    verified: bool = False

    def __post_init__(self):
        self.total_words = sum(s.word_count for s in self.sections)
        self.total_figures = len(self.resolved_figures)
        self.total_citations = len(self.references)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "template_type": self.template_type.value,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [s.to_dict() for s in self.sections],
            "references": self.references,
            "figure_requests": [f.to_dict() for f in self.figure_requests],
            "resolved_figures": self.resolved_figures,
            "total_words": self.total_words,
            "total_figures": self.total_figures,
            "total_citations": self.total_citations,
            "synthesis_time_ms": self.synthesis_time_ms,
            "verification_score": self.verification_score,
            "verification_issues": self.verification_issues,
            "verified": self.verified,
        }

    def to_markdown(self) -> str:
        md = f"# {self.title}\n\n"
        if self.abstract:
            md += f"*{self.abstract}*\n\n---\n\n"
        for section in self.sections:
            level = "#" * section.level
            md += f"{level} {section.title}\n\n{section.content}\n\n"
        if self.references:
            md += "## References\n\n"
            for i, ref in enumerate(self.references, 1):
                source = ref.get("source", "Unknown")
                md += f"{i}. {source}\n"
        return md


# =============================================================================
# TEMPLATE DEFINITIONS
# =============================================================================

TEMPLATE_SECTIONS = {
    TemplateType.PROCEDURAL: [
        ("Indications", 2),
        ("Preoperative Considerations", 2),
        ("Patient Positioning", 2),
        ("Surgical Approach", 2),
        ("Step-by-Step Technique", 2),
        ("Closure", 2),
        ("Complications and Avoidance", 2),
        ("Outcomes", 2),
    ],
    TemplateType.DISORDER: [
        ("Overview", 2),
        ("Epidemiology", 2),
        ("Pathophysiology", 2),
        ("Clinical Presentation", 2),
        ("Diagnostic Workup", 2),
        ("Imaging Findings", 2),
        ("Differential Diagnosis", 2),
        ("Management", 2),
        ("Prognosis", 2),
    ],
    TemplateType.ANATOMY: [
        ("Boundaries and Relationships", 2),
        ("Surface Anatomy", 2),
        ("Osseous Anatomy", 2),
        ("Dural Relationships", 2),
        ("Arterial Supply", 2),
        ("Venous Drainage", 2),
        ("Neural Structures", 2),
        ("Surgical Corridors", 2),
        ("Key Measurements", 2),
    ],
    TemplateType.ENCYCLOPEDIA: [
        ("Definition and Overview", 2),
        ("Historical Perspective", 2),
        ("Anatomy", 2),
        ("Pathology", 2),
        ("Clinical Features", 2),
        ("Diagnostic Approach", 2),
        ("Treatment Options", 2),
        ("Surgical Technique", 2),
        ("Outcomes and Prognosis", 2),
        ("Future Directions", 2),
    ],
}

TEMPLATE_REQUIREMENTS = {
    TemplateType.PROCEDURAL: {"min_words": 3000, "min_figures": 8},
    TemplateType.DISORDER: {"min_words": 5000, "min_figures": 10},
    TemplateType.ANATOMY: {"min_words": 4000, "min_figures": 15},
    TemplateType.ENCYCLOPEDIA: {"min_words": 15000, "min_figures": 20},
}


# =============================================================================
# CONTEXT ADAPTER (Uses SearchResult directly)
# =============================================================================

class ContextAdapter:
    """
    Adapts Phase 1 SearchResult objects into template-ready context.
    
    NO CONVERSION NEEDED - SearchResult already has:
    - chunk_id, content, document_id, document_title
    - chunk_type, page_start, entity_names, image_ids
    - authority_score, images (List[ExtractedImage])
    """

    def __init__(self):
        self.section_keywords = self._build_section_keywords()

    def _build_section_keywords(self) -> Dict[str, List[str]]:
        return {
            "Indications": ["indication", "recommended", "appropriate", "criteria"],
            "Preoperative": ["preoperative", "pre-op", "planning", "workup"],
            "Positioning": ["position", "prone", "supine", "lateral", "pin"],
            "Approach": ["approach", "craniotomy", "incision", "exposure"],
            "Technique": ["technique", "dissection", "retraction", "resection"],
            "Closure": ["closure", "dural", "cranioplasty", "wound"],
            "Complications": ["complication", "risk", "avoid", "prevent"],
            "Outcomes": ["outcome", "result", "follow-up", "survival"],
            "Epidemiology": ["epidemiology", "incidence", "prevalence"],
            "Pathophysiology": ["pathophysiology", "mechanism", "etiology"],
            "Clinical": ["symptom", "sign", "presentation", "deficit"],
            "Diagnostic": ["diagnosis", "workup", "laboratory"],
            "Imaging": ["MRI", "CT", "imaging", "radiograph"],
            "Differential": ["differential", "versus", "distinguish"],
            "Management": ["management", "treatment", "therapy"],
            "Prognosis": ["prognosis", "survival", "outcome"],
            "Anatomy": ["anatomy", "structure", "relationship"],
            "Arterial": ["artery", "arterial", "blood supply"],
            "Venous": ["vein", "venous", "sinus", "drainage"],
            "Neural": ["nerve", "neural", "cranial nerve"],
            "Measurements": ["measurement", "dimension", "mm", "cm"],
        }

    def detect_authority_from_title(self, document_title: str) -> AuthoritySource:
        """Detect authority level from document title."""
        if not document_title:
            return AuthoritySource.GENERAL
            
        title_lower = document_title.lower()
        
        if "rhoton" in title_lower or "microsurgical anatomy" in title_lower:
            return AuthoritySource.RHOTON
        elif "youmans" in title_lower:
            return AuthoritySource.YOUMANS
        elif "schmidek" in title_lower:
            return AuthoritySource.SCHMIDEK
        elif "greenberg" in title_lower or "handbook" in title_lower:
            return AuthoritySource.GREENBERG
        elif any(x in title_lower for x in ["journal", "j neurosurg"]):
            return AuthoritySource.JOURNAL
        elif "textbook" in title_lower:
            return AuthoritySource.TEXTBOOK
        else:
            return AuthoritySource.GENERAL

    def classify_section(self, content: str, template_type: TemplateType) -> str:
        """Classify content into the most appropriate template section."""
        content_lower = content.lower()
        sections = TEMPLATE_SECTIONS.get(template_type, [])
        
        best_section = sections[0][0] if sections else "General"
        best_score = 0
        
        for section_name, _ in sections:
            keywords = []
            for key, kw_list in self.section_keywords.items():
                if key.lower() in section_name.lower():
                    keywords.extend(kw_list)
            
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_section = section_name
        
        return best_section

    def adapt(
        self,
        topic: str,
        search_results: List["SearchResult"],  # Phase 1 model
        template_type: TemplateType,
    ) -> Dict[str, Any]:
        """
        Adapt SearchResult list into template context.
        
        SearchResult ALREADY has all fields - just reorganize for templates.
        """
        sections_content: Dict[str, List[Dict]] = {
            section[0]: [] for section in TEMPLATE_SECTIONS.get(template_type, [])
        }
        
        all_sources = []
        image_catalog = []
        
        for result in search_results:
            # Access SearchResult fields directly (no conversion!)
            authority = self.detect_authority_from_title(result.document_title)
            
            # Use existing authority_score from SearchResult, or derive from title
            authority_score = result.authority_score
            if authority_score == 1.0:  # Default value, might need override
                authority_score = AUTHORITY_SCORES.get(authority, 0.6)
            
            # Classify into section
            section = self.classify_section(result.content, template_type)
            
            # Build chunk data (all from SearchResult)
            chunk_data = {
                "id": result.chunk_id,
                "content": result.content,
                "document_id": result.document_id,
                "document_title": result.document_title or result.title,
                "page": result.page_start,
                "chunk_type": result.chunk_type.value if hasattr(result.chunk_type, 'value') else str(result.chunk_type),
                "authority": authority.value,
                "authority_score": authority_score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "final_score": result.final_score,
                "combined_score": result.final_score * authority_score,
                "entity_names": result.entity_names,
            }
            
            if section in sections_content:
                sections_content[section].append(chunk_data)
            
            # Build source reference
            doc_title = result.document_title or result.title or "Unknown"
            source_ref = {
                "source": f"{doc_title}, p.{result.page_start}",
                "document_id": result.document_id,
                "authority": authority.value,
                "chunks_used": 1,
            }
            
            # Deduplicate sources
            existing = next((s for s in all_sources if s["document_id"] == result.document_id), None)
            if existing:
                existing["chunks_used"] += 1
            else:
                all_sources.append(source_ref)
            
            # Collect images (already attached to SearchResult!)
            for img in (result.images or []):
                image_catalog.append({
                    "id": img.id if hasattr(img, 'id') else str(img),
                    "caption": getattr(img, 'caption', '') or getattr(img, 'vlm_caption', ''),
                    "path": getattr(img, 'file_path', '') or getattr(img, 'image_path', ''),
                    "page": getattr(img, 'page_number', 0),
                    "document_title": result.document_title,
                    "image_type": getattr(img, 'image_type', 'unknown'),
                })
        
        # Sort chunks within each section by combined score
        for section in sections_content:
            sections_content[section].sort(
                key=lambda x: x["combined_score"],
                reverse=True
            )
        
        # Sort sources by authority
        all_sources.sort(key=lambda x: AUTHORITY_SCORES.get(
            AuthoritySource(x["authority"]) if x["authority"] in [e.value for e in AuthoritySource] else AuthoritySource.GENERAL,
            0.5
        ), reverse=True)
        
        return {
            "topic": topic,
            "template_type": template_type,
            "sections": sections_content,
            "sources": all_sources,
            "image_catalog": image_catalog,
            "requirements": TEMPLATE_REQUIREMENTS.get(template_type, {}),
            "total_chunks": len(search_results),
        }


# =============================================================================
# FIGURE RESOLVER
# =============================================================================

class FigureResolver:
    """Resolves figure placeholders to actual images from SearchResult.images."""

    def __init__(self, min_match_score: float = 0.3):
        self.min_match_score = min_match_score

    def resolve(
        self,
        figure_requests: List[FigureRequest],
        image_catalog: List[Dict],
    ) -> Tuple[List[FigureRequest], List[Dict]]:
        """Match figure requests to images in catalog."""
        resolved_figures = []
        used_images = set()
        
        for request in figure_requests:
            best_match = None
            best_score = 0
            
            for img in image_catalog:
                if img["id"] in used_images:
                    continue
                    
                score = self._calculate_match_score(request, img)
                if score > best_score and score >= self.min_match_score:
                    best_score = score
                    best_match = img
            
            if best_match:
                request.resolved_id = best_match.get("id")
                request.resolved_path = best_match.get("path")
                request.caption = best_match.get("caption") or f"Figure: {request.topic}"
                used_images.add(best_match["id"])
                
                resolved_figures.append({
                    "placeholder_id": request.placeholder_id,
                    "image_id": request.resolved_id,
                    "path": request.resolved_path,
                    "caption": request.caption,
                    "match_score": best_score,
                })
        
        return figure_requests, resolved_figures

    def _calculate_match_score(self, request: FigureRequest, image: Dict) -> float:
        score = 0.0
        
        caption = (image.get("caption") or "").lower()
        topic = request.topic.lower()
        fig_type = request.figure_type.lower()
        
        # Topic word overlap
        topic_words = set(topic.split())
        caption_words = set(caption.split())
        overlap = len(topic_words & caption_words)
        if topic_words:
            score += 0.5 * (overlap / len(topic_words))
        
        # Figure type matching
        type_keywords = {
            "surgical_diagram": ["surgical", "operative", "procedure"],
            "anatomy": ["anatomy", "anatomical", "structure"],
            "imaging": ["mri", "ct", "radiograph", "scan"],
            "illustration": ["illustration", "diagram", "schematic"],
            "intraoperative": ["intraoperative", "surgical field"],
        }
        
        for kw in type_keywords.get(fig_type, []):
            if kw in caption:
                score += 0.25
                break
        
        # Image type matching
        img_type = (image.get("image_type") or "").lower()
        if fig_type in img_type or img_type in fig_type:
            score += 0.25
        
        return min(score, 1.0)


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================

class SynthesisEngine:
    """
    Main synthesis engine using Phase 1 SearchResult.
    
    Flow:
    1. ContextAdapter.adapt() - organize SearchResults by template section
    2. Generate sections with Claude API (rate-limited)
    3. FigureResolver.resolve() - match placeholders to images
    4. Optional: Gemini verification
    """

    def __init__(
        self,
        anthropic_client,
        verification_client=None,
        model: str = "claude-sonnet-4-20250514",
        calls_per_minute: int = 50,
    ):
        self.client = anthropic_client
        self.model = model
        self.rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
        self.verification_client = verification_client
        self.has_verification = verification_client is not None
        
        self.adapter = ContextAdapter()
        self.figure_resolver = FigureResolver()

    async def _call_claude(self, prompt: str, max_tokens: int = 4000) -> str:
        """Rate-limited Claude API call."""
        await self.rate_limiter.acquire()
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def synthesize(
        self,
        topic: str,
        template_type: TemplateType,
        search_results: List["SearchResult"],  # Phase 1 model!
        include_verification: bool = False,
        include_figures: bool = True,
    ) -> SynthesisResult:
        """
        Generate textbook-quality synthesis from SearchResult list.
        
        Args:
            topic: Chapter topic
            template_type: Template style
            search_results: List[SearchResult] from SearchService
            include_verification: Run Gemini verification
            include_figures: Resolve figure placeholders
        """
        start_time = time()
        
        # Stage 1: Adapt context (no conversion - SearchResult has everything)
        logger.info(f"Adapting {len(search_results)} SearchResults for synthesis")
        context = self.adapter.adapt(topic, search_results, template_type)
        
        # Stage 2: Generate title and abstract
        title, abstract = await self._generate_title_abstract(topic, template_type, context)
        
        # Stage 3: Generate sections
        sections = []
        all_figure_requests = []
        
        for section_name, level in TEMPLATE_SECTIONS.get(template_type, []):
            logger.info(f"Generating section: {section_name}")
            
            section_chunks = context["sections"].get(section_name, [])
            section_content, figure_requests = await self._generate_section(
                topic=topic,
                section_name=section_name,
                chunks=section_chunks,
                template_type=template_type,
            )
            
            sections.append(SynthesisSection(
                title=section_name,
                content=section_content,
                level=level,
                sources=[c["id"] for c in section_chunks[:5]],
            ))
            
            all_figure_requests.extend(figure_requests)
        
        # Stage 4: Resolve figures
        resolved_figures = []
        if include_figures and all_figure_requests:
            logger.info(f"Resolving {len(all_figure_requests)} figure requests")
            all_figure_requests, resolved_figures = self.figure_resolver.resolve(
                all_figure_requests,
                context["image_catalog"],
            )
        
        # Build result
        result = SynthesisResult(
            topic=topic,
            template_type=template_type,
            title=title,
            abstract=abstract,
            sections=sections,
            references=context["sources"],
            figure_requests=all_figure_requests,
            resolved_figures=resolved_figures,
            synthesis_time_ms=int((time() - start_time) * 1000),
        )
        
        # Stage 5: Optional verification
        if include_verification:
            if self.has_verification:
                logger.info("Running Gemini verification")
                result = await self._verify(result)
            else:
                logger.warning("Verification requested but Gemini client not available")
        
        logger.info(
            f"Synthesis complete: {result.total_words} words, "
            f"{result.total_figures} figures, {result.synthesis_time_ms}ms"
        )
        
        return result

    async def _generate_title_abstract(
        self,
        topic: str,
        template_type: TemplateType,
        context: Dict,
    ) -> Tuple[str, str]:
        """Generate chapter title and abstract."""
        sources_summary = ", ".join(
            s["source"].split(",")[0] for s in context["sources"][:5]
        )
        
        prompt = f"""Generate a professional title and abstract for a neurosurgical textbook chapter.

Topic: {topic}
Style: {template_type.value}
Primary Sources: {sources_summary}

Requirements:
- Title: Clear, professional, suitable for medical textbook
- Abstract: 150-200 words summarizing scope and key points

Format your response exactly as:
TITLE: [Your title here]
ABSTRACT: [Your abstract here]"""

        response = await self._call_claude(prompt, max_tokens=600)
        
        title = topic
        abstract = ""
        
        if "TITLE:" in response:
            match = re.search(r"TITLE:\s*(.+?)(?=ABSTRACT:|$)", response, re.DOTALL)
            if match:
                title = match.group(1).strip()
        
        if "ABSTRACT:" in response:
            match = re.search(r"ABSTRACT:\s*(.+)", response, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
        
        return title, abstract

    async def _generate_section(
        self,
        topic: str,
        section_name: str,
        chunks: List[Dict],
        template_type: TemplateType,
    ) -> Tuple[str, List[FigureRequest]]:
        """Generate a single section with figure requests."""
        
        if not chunks:
            # Generate minimal content if no source chunks
            prompt = f"""Write a brief "{section_name}" section for a neurosurgical chapter on "{topic}".
Keep it to 100-150 words as no specific source material was found for this section.
Write in formal medical textbook style."""
            
            content = await self._call_claude(prompt, max_tokens=500)
            return content, []
        
        # Build source context from top chunks
        source_context = ""
        for i, chunk in enumerate(chunks[:8], 1):
            doc_title = chunk.get("document_title", "Unknown")
            page = chunk.get("page", "?")
            authority = chunk.get("authority", "GENERAL")
            source_context += f"\n[Source {i}] ({doc_title}, p.{page}, Authority: {authority}):\n"
            source_context += chunk["content"][:1200] + "\n"
        
        prompt = f"""Write the "{section_name}" section for a neurosurgical textbook chapter on "{topic}".

SOURCE MATERIALS:
{source_context}

REQUIREMENTS:
1. Write in formal medical textbook style (Youmans/Rhoton standard)
2. Include specific measurements with Mean±SD where available
3. Cite sources using [Source N] format
4. Include clinical pearls: [PEARL]content[/PEARL]
5. Include hazard warnings: [HAZARD]content[/HAZARD]
6. Request figures: [REQUEST_FIGURE: type="..." topic="..."]
7. Target length: 400-600 words
8. Use evidence grading (Level I/II/III) for clinical claims
9. Prioritize higher-authority sources (Rhoton > Youmans > Greenberg)

Write only the section content, no title."""

        content = await self._call_claude(prompt, max_tokens=2000)
        
        # Extract figure requests
        figure_requests = []
        pattern = r'\[REQUEST_FIGURE:\s*type="([^"]+)"\s*topic="([^"]+)"\]'
        
        for i, match in enumerate(re.finditer(pattern, content)):
            figure_requests.append(FigureRequest(
                placeholder_id=f"{section_name.replace(' ', '_')}_{i}",
                figure_type=match.group(1),
                topic=match.group(2),
                context=section_name,
            ))
        
        return content, figure_requests

    async def _verify(self, result: SynthesisResult) -> SynthesisResult:
        """Run Gemini verification on synthesized content."""
        if not self.has_verification:
            return result
        
        try:
            content_sample = "\n\n".join(
                f"## {s.title}\n{s.content[:800]}..."
                for s in result.sections[:4]
            )
            
            prompt = f"""Verify this neurosurgical textbook content for accuracy.

CONTENT:
{content_sample}

Check for:
1. Factual accuracy (anatomy, procedures)
2. Internal consistency
3. Missing critical information
4. Potential hallucinations

Respond with:
SCORE: [0.0-1.0]
ISSUES:
- [List issues or "None"]"""

            response = await asyncio.to_thread(
                self.verification_client.generate_content,
                prompt
            )
            
            response_text = response.text
            
            score_match = re.search(r"SCORE:\s*([\d.]+)", response_text)
            if score_match:
                result.verification_score = float(score_match.group(1))
                result.verified = True
            
            issues_match = re.search(r"ISSUES:\s*(.+)", response_text, re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1).strip()
                if issues_text.lower() != "none":
                    result.verification_issues = [
                        line.strip("- ").strip()
                        for line in issues_text.split("\n")
                        if line.strip() and line.strip() != "-"
                    ]
            
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            result.verification_score = None
        
        return result
