"""
IntelligentBookGrouper - Groups scattered PDF chapters into book hierarchies.

Uses fuzzy matching on filenames to identify chapters from the same book,
validates with sequential chapter numbering, and produces parent-child relationships.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from rapidfuzz import fuzz, process
from flashtext import KeywordProcessor

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Minimal document info needed for grouping."""
    id: str
    title: str
    filename: str
    file_path: str
    authority_tier: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BookGroup:
    """A group of documents that belong to the same book."""
    book_signature: str  # Normalized book title/identifier
    book_title: str  # Display title for the book
    chapters: List[DocumentInfo]
    chapter_numbers: List[Optional[int]]  # Extracted chapter numbers
    confidence: float = 0.0
    authority_tier: Optional[str] = None

    @property
    def is_valid_book(self) -> bool:
        """A book needs at least 2 chapters to be considered a valid group."""
        return len(self.chapters) >= 2

    @property
    def has_sequential_chapters(self) -> bool:
        """Check if chapter numbers are reasonably sequential."""
        valid_nums = sorted([n for n in self.chapter_numbers if n is not None])
        if len(valid_nums) < 2:
            return False
        # Check for gaps no larger than 5 (allows for missing chapters)
        for i in range(1, len(valid_nums)):
            if valid_nums[i] - valid_nums[i-1] > 5:
                return False
        return True


@dataclass
class GroupingResult:
    """Result of the grouping operation."""
    books: List[BookGroup]
    ungrouped: List[DocumentInfo]
    total_documents: int
    grouped_count: int
    confidence_stats: Dict[str, float]


class IntelligentBookGrouper:
    """
    Groups scattered PDF chapters into book hierarchies using:
    1. Fuzzy title matching
    2. Authority tier consistency
    3. Sequential chapter number detection
    4. Filename pattern analysis
    """

    # Patterns for extracting chapter numbers
    CHAPTER_PATTERNS = [
        r'[Cc]hapter[\s_-]*(\d+)',
        r'[Cc]h[\s_.-]*(\d+)',
        r'[Pp]art[\s_-]*(\d+)',
        r'[Ss]ection[\s_-]*(\d+)',
        r'[\s_-](\d{1,2})[\s_.-]',  # "Book Title - 01 - Chapter Name"
        r'^(\d{1,2})[\s_.-]',  # "01 - Chapter Name"
        r'_(\d{1,2})_',  # "book_01_chapter"
    ]

    # Common filler words to remove when normalizing
    FILLER_WORDS = {
        'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'with',
        'chapter', 'ch', 'part', 'section', 'vol', 'volume', 'edition',
        'ed', 'pdf', 'scan', 'copy', 'final', 'draft', 'ocr'
    }

    # Authority tiers that should group together
    MASTER_AUTHORITIES = {'RHOTON', 'LAWTON', 'YASARGIL', 'SEKHAR', 'SPETZLER'}

    def __init__(
        self,
        similarity_threshold: float = 70.0,
        confidence_threshold: float = 0.6,
        min_chapters: int = 2
    ):
        """
        Initialize the grouper.

        Args:
            similarity_threshold: Minimum fuzzy match score (0-100) for titles
            confidence_threshold: Minimum confidence to form a group (0.0-1.0)
            min_chapters: Minimum chapters needed to form a book group
        """
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.min_chapters = min_chapters

        # FlashText for fast keyword matching
        self.keyword_processor = KeywordProcessor()
        self._init_authority_keywords()

    def _init_authority_keywords(self):
        """Initialize FlashText with authority keywords."""
        authority_keywords = {
            'RHOTON': ['rhoton', 'cranial anatomy', 'microsurgical anatomy of the brain'],
            'LAWTON': ['lawton', 'seven aneurysms', 'seven avms'],
            'YASARGIL': ['yasargil', 'microneurosurgery'],
            'SEKHAR': ['sekhar', 'atlas of neurosurgical techniques'],
            'GREENBERG': ['greenberg', 'handbook of neurosurgery'],
            'YOUMANS': ['youmans', 'winn', 'neurological surgery'],
            'SPETZLER': ['spetzler', 'barrow neurological'],
        }
        for authority, keywords in authority_keywords.items():
            for kw in keywords:
                self.keyword_processor.add_keyword(kw, authority)

    def extract_chapter_number(self, filename: str, title: str) -> Optional[int]:
        """Extract chapter number from filename or title."""
        text = f"{filename} {title}".lower()

        for pattern in self.CHAPTER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    num = int(match.group(1))
                    if 0 < num < 100:  # Reasonable chapter range
                        return num
                except (ValueError, IndexError):
                    continue
        return None

    def normalize_title(self, text: str) -> str:
        """
        Normalize a title for comparison.
        Removes filler words, punctuation, and normalizes spacing.
        """
        # Lowercase and remove file extension
        text = text.lower()
        text = re.sub(r'\.(pdf|epub|djvu|doc|docx)$', '', text, flags=re.IGNORECASE)

        # Remove chapter/section indicators
        text = re.sub(r'[Cc]hapter[\s_-]*\d+', '', text)
        text = re.sub(r'[Cc]h[\s_.-]*\d+', '', text)
        text = re.sub(r'[Pp]art[\s_-]*\d+', '', text)

        # Remove punctuation except hyphens and underscores
        text = re.sub(r'[^\w\s\-_]', ' ', text)

        # Split into words
        words = text.split()

        # Remove filler words and numbers
        words = [w for w in words if w not in self.FILLER_WORDS and not w.isdigit()]

        # Sort to make comparison order-independent
        words = sorted(set(words))

        return ' '.join(words)

    def extract_book_signature(self, doc: DocumentInfo) -> str:
        """
        Extract a normalized "book signature" from a document.
        This is used to identify chapters from the same book.
        """
        # Combine filename and title for better matching
        combined = f"{doc.filename} {doc.title}"

        # Check for known authority patterns first
        authorities = self.keyword_processor.extract_keywords(combined.lower())
        if authorities:
            # Use authority name as the primary signature
            return f"AUTHORITY_{authorities[0]}"

        # Otherwise normalize the title
        return self.normalize_title(combined)

    def calculate_group_confidence(self, group: BookGroup) -> float:
        """
        Calculate confidence score for a book group.

        Factors:
        - Number of chapters (more = higher confidence)
        - Sequential chapter numbers (if present, increases confidence)
        - Authority tier match (if MASTER tier, higher confidence)
        - Title similarity consistency
        """
        confidence = 0.0

        # Base confidence from chapter count
        chapter_count = len(group.chapters)
        if chapter_count >= 5:
            confidence += 0.4
        elif chapter_count >= 3:
            confidence += 0.3
        else:
            confidence += 0.2

        # Bonus for sequential chapters
        if group.has_sequential_chapters:
            confidence += 0.3

        # Bonus for MASTER authority tier
        if group.authority_tier in self.MASTER_AUTHORITIES:
            confidence += 0.2

        # Bonus for chapter number coverage
        valid_nums = [n for n in group.chapter_numbers if n is not None]
        if valid_nums:
            coverage = len(valid_nums) / chapter_count
            confidence += coverage * 0.1

        return min(confidence, 1.0)

    def group_documents(self, documents: List[DocumentInfo]) -> GroupingResult:
        """
        Group documents into book hierarchies.

        Args:
            documents: List of documents to group

        Returns:
            GroupingResult with books and ungrouped documents
        """
        if not documents:
            return GroupingResult(
                books=[],
                ungrouped=[],
                total_documents=0,
                grouped_count=0,
                confidence_stats={}
            )

        logger.info(f"Grouping {len(documents)} documents...")

        # Step 1: Extract signatures and chapter numbers
        doc_signatures: Dict[str, List[Tuple[DocumentInfo, Optional[int]]]] = defaultdict(list)

        for doc in documents:
            signature = self.extract_book_signature(doc)
            chapter_num = self.extract_chapter_number(doc.filename, doc.title)
            doc_signatures[signature].append((doc, chapter_num))

        logger.info(f"Found {len(doc_signatures)} unique signatures")

        # Step 2: Cluster similar signatures using fuzzy matching
        signature_clusters = self._cluster_signatures(list(doc_signatures.keys()))

        # Step 3: Form book groups from clusters
        books: List[BookGroup] = []
        ungrouped: List[DocumentInfo] = []

        for cluster in signature_clusters:
            # Merge all documents from signatures in this cluster
            cluster_docs: List[Tuple[DocumentInfo, Optional[int]]] = []
            for sig in cluster:
                cluster_docs.extend(doc_signatures[sig])

            if len(cluster_docs) < self.min_chapters:
                # Not enough chapters for a book
                ungrouped.extend([d for d, _ in cluster_docs])
                continue

            # Create book group
            chapters = [d for d, _ in cluster_docs]
            chapter_nums = [n for _, n in cluster_docs]

            # Determine authority tier (use most common among chapters)
            authority_counts = defaultdict(int)
            for ch in chapters:
                if ch.authority_tier:
                    authority_counts[ch.authority_tier] += 1
            authority_tier = max(authority_counts, key=authority_counts.get) if authority_counts else None

            # Generate book title
            book_title = self._generate_book_title(cluster, chapters)

            group = BookGroup(
                book_signature=cluster[0],  # Use first signature as canonical
                book_title=book_title,
                chapters=chapters,
                chapter_numbers=chapter_nums,
                authority_tier=authority_tier
            )

            group.confidence = self.calculate_group_confidence(group)

            if group.confidence >= self.confidence_threshold:
                books.append(group)
            else:
                ungrouped.extend(chapters)

        # Sort books by confidence (highest first)
        books.sort(key=lambda b: (-b.confidence, b.book_title))

        # Sort chapters within each book by chapter number
        for book in books:
            sorted_chapters = sorted(
                zip(book.chapters, book.chapter_numbers),
                key=lambda x: (x[1] or 999, x[0].title)
            )
            book.chapters = [c for c, _ in sorted_chapters]
            book.chapter_numbers = [n for _, n in sorted_chapters]

        grouped_count = sum(len(b.chapters) for b in books)

        # Calculate confidence statistics
        confidence_stats = {}
        if books:
            confidences = [b.confidence for b in books]
            confidence_stats = {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'high_confidence_count': sum(1 for c in confidences if c >= 0.8)
            }

        logger.info(
            f"Grouped {grouped_count}/{len(documents)} documents into {len(books)} books "
            f"({len(ungrouped)} ungrouped)"
        )

        return GroupingResult(
            books=books,
            ungrouped=ungrouped,
            total_documents=len(documents),
            grouped_count=grouped_count,
            confidence_stats=confidence_stats
        )

    def _cluster_signatures(self, signatures: List[str]) -> List[List[str]]:
        """
        Cluster similar signatures together using fuzzy matching.
        Returns list of signature clusters.
        """
        if not signatures:
            return []

        # Track which signatures have been clustered
        clustered = set()
        clusters: List[List[str]] = []

        for sig in signatures:
            if sig in clustered:
                continue

            # Start a new cluster with this signature
            cluster = [sig]
            clustered.add(sig)

            # Find similar signatures
            remaining = [s for s in signatures if s not in clustered]
            if remaining:
                matches = process.extract(
                    sig,
                    remaining,
                    scorer=fuzz.token_sort_ratio,
                    limit=None
                )

                for match_sig, score, _ in matches:
                    if score >= self.similarity_threshold and match_sig not in clustered:
                        cluster.append(match_sig)
                        clustered.add(match_sig)

            clusters.append(cluster)

        return clusters

    def _generate_book_title(
        self,
        signatures: List[str],
        chapters: List[DocumentInfo]
    ) -> str:
        """Generate a display title for a book group."""
        # Check for authority-based signature
        if signatures and signatures[0].startswith('AUTHORITY_'):
            authority = signatures[0].replace('AUTHORITY_', '')

            # Try to find a more descriptive title from chapters
            for ch in chapters:
                title_lower = ch.title.lower()
                if authority.lower() in title_lower:
                    # Extract the book name from chapter title
                    # E.g., "Rhoton's Cranial Anatomy - Chapter 1" -> "Rhoton's Cranial Anatomy"
                    clean_title = re.sub(
                        r'\s*[-–—]\s*(chapter|ch|part|section).*$',
                        '',
                        ch.title,
                        flags=re.IGNORECASE
                    )
                    if clean_title:
                        return clean_title.strip()

            return authority.title()

        # Use the longest common prefix of chapter titles
        if chapters:
            titles = [ch.title for ch in chapters]

            # Find common prefix
            prefix = titles[0]
            for title in titles[1:]:
                while not title.startswith(prefix) and prefix:
                    prefix = prefix[:-1]

            if len(prefix) > 10:
                # Clean up the prefix
                prefix = re.sub(r'\s*[-–—]\s*$', '', prefix).strip()
                return prefix

        # Fallback to first signature (denormalized)
        return signatures[0].replace('_', ' ').title() if signatures else "Unknown Book"


def group_library_documents(
    documents: List[Dict[str, Any]],
    confidence_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Convenience function to group library documents.

    Args:
        documents: List of document dicts with id, title, filename, file_path, etc.
        confidence_threshold: Minimum confidence for grouping

    Returns:
        Dict with 'books' and 'ungrouped' keys
    """
    # Convert dicts to DocumentInfo objects
    doc_infos = []
    for doc in documents:
        doc_info = DocumentInfo(
            id=str(doc.get('id', '')),
            title=doc.get('title', ''),
            filename=doc.get('filename', ''),
            file_path=doc.get('file_path', ''),
            authority_tier=doc.get('authority_tier'),
            metadata=doc.get('metadata', {})
        )
        doc_infos.append(doc_info)

    grouper = IntelligentBookGrouper(confidence_threshold=confidence_threshold)
    result = grouper.group_documents(doc_infos)

    # Convert back to dicts
    return {
        'books': [
            {
                'book_signature': book.book_signature,
                'book_title': book.book_title,
                'confidence': book.confidence,
                'authority_tier': book.authority_tier,
                'chapter_count': len(book.chapters),
                'chapters': [
                    {
                        'id': ch.id,
                        'title': ch.title,
                        'filename': ch.filename,
                        'chapter_number': num
                    }
                    for ch, num in zip(book.chapters, book.chapter_numbers)
                ]
            }
            for book in result.books
        ],
        'ungrouped': [
            {
                'id': doc.id,
                'title': doc.title,
                'filename': doc.filename
            }
            for doc in result.ungrouped
        ],
        'stats': {
            'total': result.total_documents,
            'grouped': result.grouped_count,
            'ungrouped': len(result.ungrouped),
            'book_count': len(result.books),
            'confidence': result.confidence_stats
        }
    }
