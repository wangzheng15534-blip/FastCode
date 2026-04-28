"""Domain-independent JSON utilities."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import ast
import json
import re
from typing import Any

_MAX_SAFE_JSONABLE_DEPTH = 12


def safe_jsonable(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert objects to JSON-serializable structures.

    Handles dicts, lists/tuples/sets, objects with ``to_dict()``, and
    arbitrary objects via ``vars()``.  Non-serializable values fall back
    to ``repr()``.  Depth is capped to prevent infinite recursion on
    circular references.
    """
    if _depth > _MAX_SAFE_JSONABLE_DEPTH:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        safe_dict = {}
        for k, v in obj.items():
            try:
                safe_dict[str(k)] = safe_jsonable(v, _depth=_depth + 1)
            except Exception:
                safe_dict[str(k)] = repr(v)
        return safe_dict
    if isinstance(obj, (list, tuple, set)):
        return [safe_jsonable(v, _depth=_depth + 1) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return safe_jsonable(obj.to_dict(), _depth=_depth + 1)
        except Exception:
            return {"repr": repr(obj)}
    if hasattr(obj, "__dict__"):
        try:
            return safe_jsonable(vars(obj), _depth=_depth + 1)
        except Exception:
            return {"repr": repr(obj)}
    return repr(obj)


def extract_json_from_response(response: str) -> str:
    """Extract JSON string from LLM response, handling markdown blocks and reasoning text.

    More robust for small models that may generate malformed JSON.

    Args:
        response: Raw LLM response text.

    Returns:
        Extracted JSON string, or the original response if no JSON found.
    """
    response = response.strip()

    # Remove any leading/trailing non-JSON text that small models sometimes add
    # e.g., "Here's the JSON:", "The response is:", etc.
    prefixes_to_remove = [
        "here's the json:",
        "here is the json:",
        "the json is:",
        "json:",
        "response:",
        "output:",
        "result:",
        "here's the response:",
        "here is the response:",
    ]
    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix) :].strip()
            break

    # 1. Try to find markdown code blocks (non-greedy for nested braces)
    json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 2. Try to find raw JSON object
        # Find first '{' and matching '}'
        start = response.find("{")
        if start == -1:
            return response

        # Find matching closing brace
        brace_count = 0
        end = -1
        for i in range(start, len(response)):
            if response[i] == "{":
                brace_count += 1
            elif response[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if end == -1:
            # No matching brace found, use rfind as fallback
            end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            json_str = response[start : end + 1]
        else:
            return response

    # 3. Clean up common issues for small models
    return sanitize_json_string(json_str)


def sanitize_json_string(json_str: str) -> str:
    """Sanitize JSON string to fix common issues from small models.

    Handles:
    - Remove/escape control characters in strings
    - Fix trailing commas
    - Fix missing commas between elements

    Args:
        json_str: Potentially malformed JSON string.

    Returns:
        Sanitized JSON string.
    """
    # Remove or escape control characters (except in already escaped sequences)
    # This is a simplified approach - replace literal newlines/tabs in strings
    cleaned: list[str] = []
    in_string = False
    escape_next = False

    for i, char in enumerate(json_str):
        if escape_next:
            cleaned.append(char)
            escape_next = False
            continue

        if char == "\\" and not escape_next:
            # Check if next character forms valid escape sequence
            if i + 1 < len(json_str):
                next_char = json_str[i + 1]
                # Valid JSON escape sequences: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
                if next_char in r'"\\/bfnrtu':
                    cleaned.append(char)
                    escape_next = True
                else:
                    # Invalid escape, add the backslash as-is
                    cleaned.append(char)
            else:
                # Backslash at end of string
                cleaned.append(char)
            continue

        if char == '"':
            in_string = not in_string
            cleaned.append(char)
            continue

        # If we're inside a string and hit a control character, escape it
        if in_string:
            if char == "\n":
                cleaned.append("\\n")
            elif char == "\r":
                cleaned.append("\\r")
            elif char == "\t":
                cleaned.append("\\t")
            elif ord(char) < 32:
                # Skip or replace with space
                cleaned.append(" ")
            else:
                cleaned.append(char)
        else:
            cleaned.append(char)

    result = "".join(cleaned)

    # Remove inline comments (# or //) outside of strings
    result = remove_json_comments(result)

    # Fix trailing commas before closing braces/brackets
    result = re.sub(r",(\s*[}\]])", r"\1", result)

    # Fix missing commas between } and {, ] and [, etc.
    result = re.sub(r"\}(\s*)\{", r"},\1{", result)
    result = re.sub(r"\](\s*)\[", r"],\1[", result)
    result = re.sub(r"\}(\s*)\[", r"},\1[", result)
    result = re.sub(r"\](\s*)\{", r"],\1{", result)

    # Fix missing commas between JSON values
    # Only add comma between closing quote/bracket/brace and opening quote
    result = re.sub(r'(["}\]])(\s*)(")', r"\1,\2\3", result)
    # Fix missing comma after boolean/null followed by quote or opening brace/bracket
    return re.sub(r"\b(true|false|null)(\s*)([\"{\[])", r"\1,\2\3", result)


def remove_json_comments(json_str: str) -> str:
    """Remove inline comments from JSON string (# or // style).

    Tracks in_string state to avoid stripping URLs like "http://example.com".

    Args:
        json_str: JSON string that may contain comments.

    Returns:
        JSON string with comments removed.
    """
    lines = json_str.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        # Track if we're inside a string
        in_string = False
        escape_next = False
        cleaned_line: list[str] = []

        i = 0
        while i < len(line):
            char = line[i]

            if escape_next:
                cleaned_line.append(char)
                escape_next = False
                i += 1
                continue

            if char == "\\":
                cleaned_line.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                cleaned_line.append(char)
                i += 1
                continue

            # If not in string, check for comments
            if not in_string:
                # Check for # comment
                if char == "#":
                    # Remove everything after # on this line
                    break
                # Check for // comment
                if char == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    # Remove everything after // on this line
                    break

            cleaned_line.append(char)
            i += 1

        cleaned_lines.append("".join(cleaned_line).rstrip())

    return "\n".join(cleaned_lines)


def robust_json_parse(json_str: str) -> Any:
    """Robustly parse JSON with multiple fallback strategies for small model outputs.

    Strategies (tried in order):
    1. Direct json.loads
    2. Sanitize then parse
    3. Fix unquoted keys then parse
    4. ast.literal_eval (handles Python-style dicts)
    5. Extract first complete object with brace matching

    Args:
        json_str: JSON string to parse.

    Returns:
        Parsed JSON object (dict or list).

    Raises:
        json.JSONDecodeError: If all parsing strategies fail.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Parse with sanitization
    try:
        sanitized = sanitize_json_string(json_str)
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Try to fix common JSON errors with regex
    try:
        # Fix unquoted keys
        fixed = re.sub(
            r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', json_str
        )
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 4: Use ast.literal_eval as safer alternative (can handle Python-style dicts)
    try:
        # ast.literal_eval is safer than eval - only evaluates literals
        result = ast.literal_eval(json_str)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError, Exception):
        pass

    # Strategy 5: Try to extract and parse just the first complete object
    try:
        # Find first { and try to parse incrementally
        start = json_str.find("{")
        if start != -1:
            for end in range(len(json_str), start, -1):
                try:
                    subset = json_str[start:end]
                    if subset.count("{") == subset.count("}"):
                        return json.loads(subset)
                except Exception:
                    continue
    except Exception:
        pass

    # All strategies failed
    raise json.JSONDecodeError("All parsing strategies failed", json_str, 0)
