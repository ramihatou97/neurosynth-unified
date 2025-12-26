"""
NeuroSynth Unified - Test Suite
================================

Comprehensive tests for the NeuroSynth system.

Structure:
    tests/
    ├── conftest.py          - Shared fixtures
    ├── unit/                - Unit tests (fast, no dependencies)
    │   ├── test_context.py  - Context assembly tests
    │   ├── test_faiss.py    - FAISS manager tests
    │   ├── test_models.py   - API model tests
    │   └── test_prompts.py  - Prompt library tests
    └── integration/         - Integration tests
        └── test_api.py      - API endpoint tests

Running Tests:
    # All tests
    pytest
    
    # Unit tests only
    pytest tests/unit/
    
    # With coverage
    pytest --cov=src --cov-report=html
    
    # Specific test file
    pytest tests/unit/test_context.py
    
    # Verbose output
    pytest -v
    
    # Stop on first failure
    pytest -x

Markers:
    @pytest.mark.unit       - Fast unit tests
    @pytest.mark.integration - Integration tests
    @pytest.mark.slow       - Slow tests (API calls)
    @pytest.mark.api        - API endpoint tests
"""
