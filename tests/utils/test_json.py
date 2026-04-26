"""Tests for pure JSON parsing functions."""

import json

import pytest

from fastcode.utils.json import (
    extract_json_from_response,
    remove_json_comments,
    robust_json_parse,
    sanitize_json_string,
)


class TestExtractJsonFromResponse:
    def test_plain_json(self):
        response = '{"confidence": 80, "reasoning": "test"}'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["confidence"] == 80

    def test_json_in_markdown_block(self):
        response = '```json\n{"confidence": 90}\n```'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["confidence"] == 90

    def test_json_with_prefix(self):
        response = 'Here is the JSON:\n{"confidence": 70}'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["confidence"] == 70

    def test_no_json(self):
        response = "No JSON here"
        result = extract_json_from_response(response)
        assert result == response

    def test_embedded_json(self):
        response = 'Some text before {"key": "value"} and after'
        result = extract_json_from_response(response)
        data = json.loads(result)
        assert data["key"] == "value"


class TestSanitizeJsonString:
    def test_trailing_comma_in_object(self):
        result = sanitize_json_string('{"a": 1,}')
        assert json.loads(result) == {"a": 1}

    def test_trailing_comma_in_array(self):
        result = sanitize_json_string("[1, 2,]")
        assert json.loads(result) == [1, 2]

    def test_control_chars_in_strings(self):
        result = sanitize_json_string('{"text": "line1\nline2"}')
        data = json.loads(result)
        assert "line1" in data["text"]

    def test_missing_comma_between_nested_objects(self):
        # Two adjacent objects inside an array become valid with comma insertion
        result = sanitize_json_string('[{"a": 1}{"b": 2}]')
        data = json.loads(result)
        assert data[0]["a"] == 1
        assert data[1]["b"] == 2

    def test_missing_comma_between_nested_arrays(self):
        result = sanitize_json_string("[[1][2]]")
        data = json.loads(result)
        assert data == [[1], [2]]


class TestRemoveJsonComments:
    def test_hash_comment(self):
        result = remove_json_comments('{"a": 1} # comment')
        assert "# comment" not in result
        assert '"a": 1' in result

    def test_double_slash_comment(self):
        result = remove_json_comments('{"a": 1} // comment')
        assert "// comment" not in result

    def test_preserves_comments_in_strings(self):
        result = remove_json_comments('{"url": "http://example.com"}')
        assert "http://example.com" in result

    def test_hash_url_in_string_preserved(self):
        result = remove_json_comments('{"url": "http://example.com/path#anchor"}')
        assert "#anchor" in result


class TestRobustJsonParse:
    def test_valid_json(self):
        result = robust_json_parse('{"a": 1}')
        assert result == {"a": 1}

    def test_trailing_comma(self):
        result = robust_json_parse('{"a": 1,}')
        assert result == {"a": 1}

    def test_unquoted_keys(self):
        result = robust_json_parse("{a: 1}")
        assert result == {"a": 1}

    def test_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            robust_json_parse("not json at all {{{")

    def test_comment_stripped(self):
        result = robust_json_parse('{"a": 1} # comment')
        assert result == {"a": 1}

    def test_extract_first_object(self):
        result = robust_json_parse('prefix {"a": 1} suffix')
        assert result == {"a": 1}
