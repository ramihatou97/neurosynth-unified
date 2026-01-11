"""
NeuroSynth - Robust LLM Output Parsing
======================================

Handles markdown-wrapped JSON, preambles, and validation.
Prevents V3 pipeline crashes from LLM response format variations.
"""

import json
import re
import logging
from typing import Type, TypeVar, Optional, List
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMParsingError(Exception):
    """Raised when LLM output cannot be parsed."""
    pass


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    # Remove ```json or ``` at start of lines
    text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text, flags=re.MULTILINE)
    # Remove trailing ```
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text


def find_json_boundaries(text: str) -> tuple[int, int]:
    """
    Find the start and end indices of JSON in text.

    Handles both objects {} and arrays [].
    """
    # Try object first
    obj_start = text.find('{')
    obj_end = text.rfind('}')

    # Try array
    arr_start = text.find('[')
    arr_end = text.rfind(']')

    # Determine which to use
    if obj_start == -1 and arr_start == -1:
        return -1, -1

    if obj_start == -1:
        return arr_start, arr_end

    if arr_start == -1:
        return obj_start, obj_end

    # Both exist - use whichever comes first
    if obj_start < arr_start:
        return obj_start, obj_end
    else:
        return arr_start, arr_end


def extract_json_string(text: str) -> str:
    """
    Extract JSON string from LLM output.

    Handles:
    - ```json ... ``` blocks
    - Preambles ("Here is the JSON: {...}")
    - Postscripts ("Let me know if...")
    """
    # Step 1: Strip markdown fences
    text = strip_markdown_fences(text)

    # Step 2: Find JSON boundaries
    start_idx, end_idx = find_json_boundaries(text)

    if start_idx == -1 or end_idx == -1:
        raise LLMParsingError("No JSON object/array found in text")

    if end_idx < start_idx:
        raise LLMParsingError("Malformed JSON: end before start")

    return text[start_idx : end_idx + 1]


def extract_and_parse_json(text: str, model_class: Type[T]) -> T:
    """
    Robustly extracts JSON from LLM output and validates against Pydantic model.

    Args:
        text: Raw LLM response
        model_class: Pydantic model to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        LLMParsingError: If parsing or validation fails
    """
    try:
        # Extract JSON string
        json_str = extract_json_string(text)

        # Parse JSON
        data = json.loads(json_str)

        # Validate with Pydantic
        return model_class.model_validate(data)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        raise LLMParsingError(f"Invalid JSON syntax: {str(e)}")
    except ValidationError as e:
        logger.warning(f"Pydantic validation error: {e}")
        raise LLMParsingError(f"Schema validation failed: {str(e)}")
    except LLMParsingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected parsing error: {e}")
        raise LLMParsingError(f"Parsing failed: {str(e)}")


async def parse_json_with_retry(
    generate_func,
    model_class: Type[T],
    max_retries: int = 2,
    **generate_kwargs
) -> Optional[T]:
    """
    Attempt to generate and parse JSON with retries.

    Args:
        generate_func: Async function that generates LLM response
        model_class: Pydantic model to validate against
        max_retries: Number of retry attempts
        **generate_kwargs: Arguments to pass to generate_func

    Returns:
        Validated model instance or None if all retries fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await generate_func(**generate_kwargs)
            result = extract_and_parse_json(response, model_class)
            return result
        except LLMParsingError as e:
            last_error = e
            logger.warning(f"Parse attempt {attempt + 1}/{max_retries} failed: {e}")
            continue

    logger.error(f"All {max_retries} parse attempts failed. Last error: {last_error}")
    return None


def extract_list_items(text: str, item_pattern: str = r'[-•*]\s*(.+)') -> List[str]:
    """
    Extract list items from markdown-style lists.

    Useful for parsing bullet point responses.
    """
    items = []
    for match in re.finditer(item_pattern, text, re.MULTILINE):
        item = match.group(1).strip()
        if item:
            items.append(item)
    return items


def clean_llm_text(text: str) -> str:
    """
    Clean up LLM response text for display.

    - Removes excessive whitespace
    - Normalizes line endings
    - Strips markdown artifacts
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
