"""
Content Summarizer - Brief human-readable summaries for chunks and image captions

Generates one-sentence summaries identifying:
- Subject: what topic/procedure/anatomy
- Distinguisher: what specific aspect differentiates this content

Cost: ~$0.005/chunk, ~$0.003/image
"""

import asyncio
import logging
from typing import List, Optional
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
    
    async def summarize_chunk(self, content: str) -> str:
        """Generate summary for a single chunk."""
        async with self.semaphore:
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=60,
                    messages=[{
                        "role": "user",
                        "content": CHUNK_SUMMARY_PROMPT.format(content=content[:2000])
                    }]
                )
                summary = response.content[0].text.strip().strip('"\'')
                return summary[:150]
            except Exception as e:
                logger.warning(f"Chunk summary failed: {e}")
                return self._fallback_chunk_summary(content)
    
    async def summarize_chunks_batch(self, chunks: List[dict]) -> List[ChunkSummary]:
        """Summarize multiple chunks concurrently."""
        tasks = [
            self._summarize_chunk_with_id(chunk['id'], chunk['content'])
            for chunk in chunks
        ]
        return await asyncio.gather(*tasks)
    
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
# LEGACY ALIAS (backward compatibility)
# =============================================================================

ChunkSummarizer = ContentSummarizer
