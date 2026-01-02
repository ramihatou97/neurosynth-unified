#!/usr/bin/env python3
"""
NeuroSynth Chat Hardening Test Suite
=====================================

Tests the enhanced chat functionality:
1. Basic Q&A with citations
2. Multi-turn conversations
3. Synthesis context linking
4. Streaming responses
5. Citation tracking

Usage:
    # Start server first:
    uvicorn src.api.main:app --reload --port 8000

    # Run tests:
    python scripts/test_chat_hardening.py

    # Quick test (no streaming):
    python scripts/test_chat_hardening.py --quick
"""

import asyncio
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class TestStatus(Enum):
    PASS = "\u2705"
    FAIL = "\u274c"
    SKIP = "\u23ed\ufe0f"
    WARN = "\u26a0\ufe0f"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAIL)

    def print_summary(self):
        print("\n" + "="*70)
        print("CHAT HARDENING TEST RESULTS")
        print("="*70)

        for result in self.results:
            print(f"\n{result.status.value} {result.name} ({result.duration_ms:.0f}ms)")
            print(f"   {result.message}")
            if result.details:
                for k, v in list(result.details.items())[:3]:
                    print(f"   - {k}: {v}")

        print("\n" + "-"*70)
        print(f"TOTAL: {self.passed} passed, {self.failed} failed")
        print("-"*70)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

async def test_basic_question(client, suite: TestSuite):
    """Test basic Q&A functionality."""
    start = time.time()

    try:
        resp = await client.post("/api/v1/chat/ask", json={
            "message": "What is the retrosigmoid approach?",
            "include_citations": True,
            "max_context_chunks": 5
        })

        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()

            has_answer = bool(data.get('answer'))
            has_citations = len(data.get('citations', [])) > 0
            has_conv_id = bool(data.get('conversation_id'))

            if has_answer and has_citations:
                suite.add(TestResult(
                    name="Basic Q&A",
                    status=TestStatus.PASS,
                    duration_ms=duration,
                    message=f"Answer: {len(data['answer'])} chars, {len(data['citations'])} citations",
                    details={
                        'conversation_id': data.get('conversation_id'),
                        'used_citations': data.get('used_citation_indices', []),
                        'search_time_ms': data.get('search_time_ms')
                    }
                ))
                return data.get('conversation_id')
            else:
                suite.add(TestResult(
                    name="Basic Q&A",
                    status=TestStatus.WARN,
                    duration_ms=duration,
                    message=f"Missing: answer={has_answer}, citations={has_citations}"
                ))
        else:
            suite.add(TestResult(
                name="Basic Q&A",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}: {resp.text[:100]}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Basic Q&A",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))

    return None


async def test_multi_turn(client, suite: TestSuite, conv_id: str):
    """Test multi-turn conversation."""
    if not conv_id:
        suite.add(TestResult(
            name="Multi-turn Conversation",
            status=TestStatus.SKIP,
            duration_ms=0,
            message="No conversation ID from previous test"
        ))
        return

    start = time.time()

    try:
        # Follow-up question
        resp = await client.post("/api/v1/chat/ask", json={
            "message": "What are the complications of this approach?",
            "conversation_id": conv_id,
            "include_citations": True
        })

        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()

            # Check conversation continuity
            same_conv = data.get('conversation_id') == conv_id
            has_answer = bool(data.get('answer'))

            if same_conv and has_answer:
                suite.add(TestResult(
                    name="Multi-turn Conversation",
                    status=TestStatus.PASS,
                    duration_ms=duration,
                    message=f"Follow-up answered, same conversation",
                    details={
                        'answer_length': len(data.get('answer', '')),
                        'citations_used': len(data.get('used_citation_indices', []))
                    }
                ))
            else:
                suite.add(TestResult(
                    name="Multi-turn Conversation",
                    status=TestStatus.WARN,
                    duration_ms=duration,
                    message=f"Continuity issue: same_conv={same_conv}"
                ))
        else:
            suite.add(TestResult(
                name="Multi-turn Conversation",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Multi-turn Conversation",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


async def test_conversation_history(client, suite: TestSuite, conv_id: str):
    """Test conversation history retrieval."""
    if not conv_id:
        suite.add(TestResult(
            name="Conversation History",
            status=TestStatus.SKIP,
            duration_ms=0,
            message="No conversation ID"
        ))
        return

    start = time.time()

    try:
        resp = await client.get(f"/api/v1/chat/conversations/{conv_id}")
        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()

            turn_count = len(data.get('turns', []))
            citation_count = len(data.get('all_citations', []))

            suite.add(TestResult(
                name="Conversation History",
                status=TestStatus.PASS if turn_count >= 2 else TestStatus.WARN,
                duration_ms=duration,
                message=f"Retrieved {turn_count} turns, {citation_count} unique citations",
                details={
                    'turns': turn_count,
                    'citations': citation_count
                }
            ))
        else:
            suite.add(TestResult(
                name="Conversation History",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Conversation History",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


async def test_citation_quality(client, suite: TestSuite):
    """Test citation quality and richness."""
    start = time.time()

    try:
        resp = await client.post("/api/v1/chat/ask", json={
            "message": "Describe the facial nerve anatomy in the cerebellopontine angle",
            "include_citations": True,
            "max_context_chunks": 10
        })

        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            citations = data.get('citations', [])

            if citations:
                # Check citation richness
                sample = citations[0]
                has_title = bool(sample.get('document_title'))
                has_page = sample.get('page_number') is not None
                has_type = bool(sample.get('chunk_type'))
                has_authority = sample.get('authority_score') is not None

                richness = sum([has_title, has_page, has_type, has_authority])

                suite.add(TestResult(
                    name="Citation Quality",
                    status=TestStatus.PASS if richness >= 3 else TestStatus.WARN,
                    duration_ms=duration,
                    message=f"{len(citations)} citations, richness score: {richness}/4",
                    details={
                        'has_title': has_title,
                        'has_page': has_page,
                        'has_type': has_type,
                        'has_authority': has_authority,
                        'sample_title': sample.get('document_title', '')[:50]
                    }
                ))
            else:
                suite.add(TestResult(
                    name="Citation Quality",
                    status=TestStatus.WARN,
                    duration_ms=duration,
                    message="No citations returned"
                ))
        else:
            suite.add(TestResult(
                name="Citation Quality",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Citation Quality",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


async def test_follow_up_suggestions(client, suite: TestSuite):
    """Test follow-up question generation."""
    start = time.time()

    try:
        resp = await client.post("/api/v1/chat/ask", json={
            "message": "What surgical approaches are used for acoustic neuroma?",
            "include_citations": True
        })

        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            follow_ups = data.get('follow_up_questions', [])

            suite.add(TestResult(
                name="Follow-up Suggestions",
                status=TestStatus.PASS if follow_ups else TestStatus.WARN,
                duration_ms=duration,
                message=f"Generated {len(follow_ups)} follow-up questions",
                details={'suggestions': follow_ups[:3]}
            ))
        else:
            suite.add(TestResult(
                name="Follow-up Suggestions",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Follow-up Suggestions",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


async def test_streaming(client, suite: TestSuite, base_url: str, skip: bool = False):
    """Test streaming response."""
    if skip:
        suite.add(TestResult(
            name="Streaming Response",
            status=TestStatus.SKIP,
            duration_ms=0,
            message="Skipped (--quick mode)"
        ))
        return

    start = time.time()

    try:
        # Use regular requests for SSE
        import httpx

        chunks = []
        metadata = None

        async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as stream_client:
            async with stream_client.stream(
                "POST",
                "/api/v1/chat/ask/stream",
                json={
                    "message": "Explain the retrosigmoid approach briefly",
                    "max_context_chunks": 5
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get('type') == 'text':
                                chunks.append(data.get('content', ''))
                            elif data.get('type') == 'complete':
                                metadata = data
                        except:
                            pass

        duration = (time.time() - start) * 1000

        full_text = ''.join(chunks)

        if full_text and metadata:
            suite.add(TestResult(
                name="Streaming Response",
                status=TestStatus.PASS,
                duration_ms=duration,
                message=f"Streamed {len(chunks)} chunks, {len(full_text)} chars total",
                details={
                    'chunk_count': len(chunks),
                    'text_length': len(full_text),
                    'has_citations': bool(metadata.get('citations'))
                }
            ))
        elif full_text:
            suite.add(TestResult(
                name="Streaming Response",
                status=TestStatus.WARN,
                duration_ms=duration,
                message=f"Got text but no metadata"
            ))
        else:
            suite.add(TestResult(
                name="Streaming Response",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message="No content streamed"
            ))

    except Exception as e:
        suite.add(TestResult(
            name="Streaming Response",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


async def test_list_conversations(client, suite: TestSuite):
    """Test conversation listing."""
    start = time.time()

    try:
        resp = await client.get("/api/v1/chat/conversations")
        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            convs = resp.json()

            suite.add(TestResult(
                name="List Conversations",
                status=TestStatus.PASS,
                duration_ms=duration,
                message=f"Listed {len(convs)} conversations",
                details={
                    'count': len(convs),
                    'sample_id': convs[0]['conversation_id'] if convs else None
                }
            ))
        else:
            suite.add(TestResult(
                name="List Conversations",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="List Conversations",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


async def test_existing_rag_endpoint(client, suite: TestSuite):
    """Test existing /api/v1/rag/ask endpoint still works."""
    start = time.time()

    try:
        resp = await client.post("/api/v1/rag/ask", json={
            "question": "What is the pterional approach?",
            "max_context_chunks": 5
        })

        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            has_answer = bool(data.get('answer') or data.get('response'))

            suite.add(TestResult(
                name="Existing RAG Endpoint",
                status=TestStatus.PASS if has_answer else TestStatus.WARN,
                duration_ms=duration,
                message="Legacy endpoint working" if has_answer else "No answer returned"
            ))
        else:
            suite.add(TestResult(
                name="Existing RAG Endpoint",
                status=TestStatus.WARN,
                duration_ms=duration,
                message=f"HTTP {resp.status_code} (may not be implemented)"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Existing RAG Endpoint",
            status=TestStatus.WARN,
            duration_ms=(time.time() - start) * 1000,
            message=f"Endpoint may not exist: {str(e)[:50]}"
        ))


async def test_chat_health(client, suite: TestSuite):
    """Test chat health endpoint."""
    start = time.time()

    try:
        resp = await client.get("/api/v1/chat/health")
        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            suite.add(TestResult(
                name="Chat Health",
                status=TestStatus.PASS,
                duration_ms=duration,
                message=f"Status: {data.get('status', 'unknown')}",
                details={
                    'storage_backend': data.get('storage_backend'),
                    'engine_ready': data.get('engine_ready')
                }
            ))
        else:
            suite.add(TestResult(
                name="Chat Health",
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=f"HTTP {resp.status_code}"
            ))
    except Exception as e:
        suite.add(TestResult(
            name="Chat Health",
            status=TestStatus.FAIL,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        ))


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Test chat hardening")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Skip streaming test")

    args = parser.parse_args()

    print("="*70)
    print("NEUROSYNTH CHAT HARDENING TESTS")
    print("="*70)
    print(f"\nTarget: {args.url}")

    try:
        import httpx
    except ImportError:
        print("\n\u274c httpx not installed. Run: pip install httpx")
        sys.exit(1)

    # Check server
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{args.url}/health", timeout=5.0)
            if resp.status_code != 200:
                # Try /api/v1/health
                resp = await client.get(f"{args.url}/api/v1/health", timeout=5.0)
                if resp.status_code != 200:
                    print(f"\n\u274c Server unhealthy: {resp.status_code}")
                    sys.exit(1)
    except httpx.ConnectError:
        print(f"\n\u274c Cannot connect to {args.url}")
        print("   Start server with: uvicorn src.api.main:app --port 8000")
        sys.exit(1)

    print("\n\u2705 Server running")

    suite = TestSuite()

    async with httpx.AsyncClient(base_url=args.url, timeout=60.0) as client:
        # Test sequence
        print("\n[1] Testing chat health...")
        await test_chat_health(client, suite)

        print("[2] Testing basic Q&A...")
        conv_id = await test_basic_question(client, suite)

        print("[3] Testing multi-turn conversation...")
        await test_multi_turn(client, suite, conv_id)

        print("[4] Testing conversation history...")
        await test_conversation_history(client, suite, conv_id)

        print("[5] Testing citation quality...")
        await test_citation_quality(client, suite)

        print("[6] Testing follow-up suggestions...")
        await test_follow_up_suggestions(client, suite)

        print("[7] Testing streaming response...")
        await test_streaming(client, suite, args.url, skip=args.quick)

        print("[8] Testing conversation listing...")
        await test_list_conversations(client, suite)

        print("[9] Testing existing RAG endpoint...")
        await test_existing_rag_endpoint(client, suite)

    suite.print_summary()

    sys.exit(1 if suite.failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
