"""
Content Summarizer - Brief human-readable summaries for chunks and image captions

Generates one-sentence summaries identifying:
- Subject: what topic/procedure/anatomy
- Distinguisher: what specific aspect differentiates this content

Cost: ~$0.005/chunk, ~$0.003/image
"""

import asyncio
import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPTS
# =============================================================================

CHUNK_SUMMARY_PROMPT = """Summarize this medical text chunk in ONE sentence (max 25 words).

Format: [Subject] — [distinguishing aspect]

Examples:
- "Pterional craniotomy — skin incision and myocutaneous flap elevation"
- "Middle cerebral artery — M1 segment anatomical relationships"
- "Vestibular schwannoma — facial nerve preservation outcomes"

Chunk:
{content}

Summary:"""

CHUNK_SUMMARY_DISTINCT_PROMPT = """Summarize this medical text chunk in ONE sentence (max 25 words).

Format: [Subject] — [distinguishing aspect]

IMPORTANT: Make the summary DISTINCT from these neighboring summaries:
{neighbor_summaries}

Focus on what makes THIS chunk unique compared to neighbors.

Examples:
- "Pterional craniotomy — skin incision and myocutaneous flap elevation"
- "Middle cerebral artery — M1 segment anatomical relationships"
- "Vestibular schwannoma — facial nerve preservation outcomes"

Chunk:
{content}

Distinct Summary:"""


IMAGE_SUMMARY_PROMPT = """Summarize this medical image caption in ONE sentence (max 20 words).

Format: [What it shows] — [key detail]

Examples:
- "Pterional craniotomy — dural opening with sylvian fissure exposed"
- "MRI T1 axial — vestibular schwannoma compressing brainstem"
- "Surgical photograph — clip placement on MCA bifurcation aneurysm"
- "Anatomical diagram — circle of Willis arterial relationships"

Caption:
{caption}

Summary:"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChunkSummary:
    chunk_id: str
    summary: str


@dataclass
class ImageSummary:
    image_id: str
    caption_summary: str
    

# =============================================================================
# MAIN SUMMARIZER CLASS
# =============================================================================

class ContentSummarizer:
    """Generates brief summaries for chunks and image captions during ingestion."""
    
    def __init__(self, client, model: str = "claude-sonnet-4-20250514", max_concurrent: int = 10):
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    # -------------------------------------------------------------------------
    # CHUNK SUMMARIZATION
    # -------------------------------------------------------------------------
    
    async def summarize_chunk(
        self,
        content: str,
        neighbor_summaries: Optional[List[str]] = None
    ) -> str:
        """
        Generate summary for a single chunk.

        Args:
            content: The chunk content to summarize
            neighbor_summaries: Optional list of neighboring chunk summaries
                               to ensure distinctiveness (typically prev/next chunks)

        Returns:
            Summary string in format "[Subject] — [distinguishing aspect]"
        """
        async with self.semaphore:
            try:
                # Use distinct prompt if neighbors provided
                if neighbor_summaries and len(neighbor_summaries) > 0:
                    neighbors_text = "\n".join(f"- {s}" for s in neighbor_summaries if s)
                    prompt = CHUNK_SUMMARY_DISTINCT_PROMPT.format(
                        content=content[:2000],
                        neighbor_summaries=neighbors_text
                    )
                else:
                    prompt = CHUNK_SUMMARY_PROMPT.format(content=content[:2000])

                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=60,
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.content[0].text.strip().strip('"\'')
                summary = summary[:150]

                # Validate format
                if not self._validate_format(summary):
                    logger.debug(f"Summary format suboptimal: {summary[:50]}...")

                return summary
            except Exception as e:
                logger.warning(f"Chunk summary failed: {e}")
                return self._fallback_chunk_summary(content)
    
    async def summarize_chunks_batch(self, chunks: List[dict]) -> List[ChunkSummary]:
        """Summarize multiple chunks concurrently (no neighbor context)."""
        tasks = [
            self._summarize_chunk_with_id(chunk['id'], chunk['content'])
            for chunk in chunks
        ]
        return await asyncio.gather(*tasks)

    async def summarize_chunks_with_neighbors(
        self,
        chunks: List[dict],
        window_size: int = 1
    ) -> List[ChunkSummary]:
        """
        Summarize chunks with neighbor-based distinctiveness.

        For each chunk, includes summaries of neighboring chunks (prev/next)
        to ensure the generated summary is distinct from its neighbors.

        Args:
            chunks: List of dicts with 'id' and 'content' keys
            window_size: Number of neighbors on each side to consider (default 1)

        Returns:
            List of ChunkSummary objects with distinct summaries
        """
        if not chunks:
            return []

        # First pass: generate initial summaries (in parallel)
        logger.info(f"Generating initial summaries for {len(chunks)} chunks...")
        initial_summaries = await self.summarize_chunks_batch(chunks)
        summary_map = {s.chunk_id: s.summary for s in initial_summaries}

        # Second pass: regenerate with neighbor context where beneficial
        results = []
        regenerate_indices = []

        # Identify chunks that might benefit from distinctiveness pass
        for i, chunk in enumerate(chunks):
            prev_summaries = [
                summary_map.get(chunks[j]['id'])
                for j in range(max(0, i - window_size), i)
            ]
            next_summaries = [
                summary_map.get(chunks[j]['id'])
                for j in range(i + 1, min(len(chunks), i + 1 + window_size))
            ]

            neighbors = [s for s in prev_summaries + next_summaries if s]

            # Check if current summary is too similar to neighbors
            current_summary = summary_map[chunk['id']]
            if neighbors and self._is_too_similar(current_summary, neighbors):
                regenerate_indices.append((i, neighbors))

        # Regenerate similar summaries with neighbor context
        if regenerate_indices:
            logger.info(f"Regenerating {len(regenerate_indices)} similar summaries...")
            for i, neighbors in regenerate_indices:
                chunk = chunks[i]
                new_summary = await self.summarize_chunk(
                    chunk['content'],
                    neighbor_summaries=neighbors
                )
                summary_map[chunk['id']] = new_summary

        # Build final results
        return [
            ChunkSummary(chunk_id=chunk['id'], summary=summary_map[chunk['id']])
            for chunk in chunks
        ]

    def _is_too_similar(self, summary: str, neighbors: List[str], threshold: float = 0.7) -> bool:
        """
        Check if summary is too similar to any neighbor.

        Uses simple word overlap ratio for efficiency.
        """
        if not summary or not neighbors:
            return False

        summary_words = set(summary.lower().split())

        for neighbor in neighbors:
            if not neighbor:
                continue
            neighbor_words = set(neighbor.lower().split())

            # Jaccard similarity
            intersection = len(summary_words & neighbor_words)
            union = len(summary_words | neighbor_words)

            if union > 0 and intersection / union > threshold:
                return True

        return False

    async def _summarize_chunk_with_id(self, chunk_id: str, content: str) -> ChunkSummary:
        summary = await self.summarize_chunk(content)
        return ChunkSummary(chunk_id=chunk_id, summary=summary)
    
    def _fallback_chunk_summary(self, content: str) -> str:
        """Extract first sentence as fallback."""
        first_sentence = content.split('.')[0]
        if len(first_sentence) > 100:
            return first_sentence[:97] + "..."
        return first_sentence + "." if first_sentence else "Medical content"

    # -------------------------------------------------------------------------
    # FORMAT VALIDATION
    # -------------------------------------------------------------------------

    # Pattern: "[Subject] — [aspect]" with em-dash or regular dash
    SUMMARY_PATTERN = re.compile(r'^.{3,60}\s*[—–-]\s*.{3,80}$')

    def _validate_format(self, summary: str) -> bool:
        """
        Validate summary follows "[Subject] — [distinguishing aspect]" format.

        Returns True if format is valid, False if suboptimal.
        """
        if not summary:
            return False

        # Check for dash separator (em-dash, en-dash, or hyphen)
        has_separator = any(sep in summary for sep in ['—', '–', ' - '])

        # Check reasonable length (15-150 chars)
        valid_length = 15 <= len(summary) <= 150

        # Check word count (5-30 words)
        word_count = len(summary.split())
        valid_words = 5 <= word_count <= 30

        return has_separator and valid_length and valid_words

    def validate_summary(self, summary: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive summary validation with detailed feedback.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not summary:
            return False, ["Empty summary"]

        # Check separator
        if not any(sep in summary for sep in ['—', '–', ' - ']):
            issues.append("Missing dash separator")

        # Check length
        if len(summary) < 15:
            issues.append(f"Too short ({len(summary)} chars, min 15)")
        elif len(summary) > 150:
            issues.append(f"Too long ({len(summary)} chars, max 150)")

        # Check word count
        word_count = len(summary.split())
        if word_count < 5:
            issues.append(f"Too few words ({word_count}, min 5)")
        elif word_count > 30:
            issues.append(f"Too many words ({word_count}, max 30)")

        # Check for common issues
        if summary.startswith('"') or summary.endswith('"'):
            issues.append("Contains quotes (should be stripped)")
        if summary.startswith("Summary:"):
            issues.append("Contains 'Summary:' prefix")
        if "..." in summary and len(summary) < 50:
            issues.append("Appears truncated")

        return len(issues) == 0, issues
    
    # -------------------------------------------------------------------------
    # IMAGE CAPTION SUMMARIZATION
    # -------------------------------------------------------------------------
    
    async def summarize_image_caption(self, caption: str) -> str:
        """Generate summary for an image caption."""
        if not caption or len(caption) < 50:
            return caption or "Medical image"
        
        async with self.semaphore:
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=40,
                    messages=[{
                        "role": "user",
                        "content": IMAGE_SUMMARY_PROMPT.format(caption=caption[:1000])
                    }]
                )
                summary = response.content[0].text.strip().strip('"\'')
                return summary[:100]
            except Exception as e:
                logger.warning(f"Image summary failed: {e}")
                return self._fallback_image_summary(caption)
    
    async def summarize_images_batch(self, images: List[dict]) -> List[ImageSummary]:
        """Summarize multiple image captions concurrently."""
        tasks = [
            self._summarize_image_with_id(img['id'], img.get('vlm_caption', ''))
            for img in images
            if img.get('vlm_caption')
        ]
        return await asyncio.gather(*tasks)
    
    async def _summarize_image_with_id(self, image_id: str, caption: str) -> ImageSummary:
        summary = await self.summarize_image_caption(caption)
        return ImageSummary(image_id=image_id, caption_summary=summary)
    
    def _fallback_image_summary(self, caption: str) -> str:
        """Extract first clause as fallback."""
        first_part = caption.split(',')[0].split('.')[0]
        if len(first_part) > 80:
            return first_part[:77] + "..."
        return first_part


# =============================================================================
# STANDALONE FUNCTIONS (for simple integration)
# =============================================================================

async def generate_chunk_summary(content: str, client) -> str:
    """Quick function to summarize a single chunk."""
    summarizer = ContentSummarizer(client)
    return await summarizer.summarize_chunk(content)


async def generate_image_summary(caption: str, client) -> str:
    """Quick function to summarize a single image caption."""
    summarizer = ContentSummarizer(client)
    return await summarizer.summarize_image_caption(caption)


# =============================================================================
# QUALITY AUDIT UTILITIES
# =============================================================================

@dataclass
class SummaryQualityReport:
    """Quality report for a batch of summaries."""
    total: int
    valid: int
    invalid: int
    issues_breakdown: dict
    sample_issues: List[Tuple[str, str, List[str]]]  # (id, summary, issues)


def audit_summaries(summaries: List[Tuple[str, str]]) -> SummaryQualityReport:
    """
    Audit a batch of summaries for format compliance.

    Args:
        summaries: List of (id, summary) tuples

    Returns:
        SummaryQualityReport with statistics and sample issues
    """
    summarizer = ContentSummarizer(client=None)  # No client needed for validation

    valid_count = 0
    invalid_count = 0
    issues_breakdown = {}
    sample_issues = []

    for chunk_id, summary in summaries:
        is_valid, issues = summarizer.validate_summary(summary)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            # Track issue types
            for issue in issues:
                issue_type = issue.split('(')[0].strip()
                issues_breakdown[issue_type] = issues_breakdown.get(issue_type, 0) + 1

            # Collect samples (up to 10)
            if len(sample_issues) < 10:
                sample_issues.append((chunk_id, summary, issues))

    return SummaryQualityReport(
        total=len(summaries),
        valid=valid_count,
        invalid=invalid_count,
        issues_breakdown=issues_breakdown,
        sample_issues=sample_issues
    )


def print_quality_report(report: SummaryQualityReport):
    """Print a formatted quality report."""
    print("\n" + "=" * 60)
    print("SUMMARY QUALITY REPORT")
    print("=" * 60)
    print(f"Total summaries: {report.total}")
    print(f"Valid: {report.valid} ({100*report.valid/report.total:.1f}%)")
    print(f"Invalid: {report.invalid} ({100*report.invalid/report.total:.1f}%)")

    if report.issues_breakdown:
        print("\nIssue Breakdown:")
        for issue, count in sorted(report.issues_breakdown.items(), key=lambda x: -x[1]):
            print(f"  - {issue}: {count}")

    if report.sample_issues:
        print("\nSample Issues:")
        for chunk_id, summary, issues in report.sample_issues[:5]:
            print(f"\n  ID: {chunk_id}")
            print(f"  Summary: {summary[:80]}...")
            print(f"  Issues: {', '.join(issues)}")

    print("=" * 60 + "\n")


# =============================================================================
# LEGACY ALIAS (backward compatibility)
# =============================================================================

ChunkSummarizer = ContentSummarizer
