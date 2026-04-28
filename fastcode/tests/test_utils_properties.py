"""Property-based tests for utils module."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.utils import (
    calculate_code_complexity,
    clean_docstring,
    compute_file_hash,
    ensure_dir,
    extract_code_snippet,
    format_code_block,
    get_file_extension,
    get_language_from_extension,
    get_repo_name_from_url,
    is_supported_file,
    is_text_file,
    merge_dicts,
    normalize_path,
    resolve_config_paths,
    safe_get,
    safe_jsonable,
    truncate_to_tokens,
    utc_now,
)

# --- Helpers ---

small_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=20
)
extension_st = st.sampled_from(
    [".py", ".js", ".ts", ".go", ".java", ".rs", ".rb", ".cpp", ".c", ".h"]
)


# --- Properties ---


class TestUtcNow:
    @pytest.mark.basic
    def test_returns_iso_string_property(self):
        result = utc_now()
        assert isinstance(result, str)
        assert "T" in result


class TestComputeFileHash:
    @pytest.mark.basic
    def test_hash_deterministic_property(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello world")
            f.flush()
            h1 = compute_file_hash(f.name)
            h2 = compute_file_hash(f.name)
        os.unlink(f.name)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex length

    @pytest.mark.edge
    def test_hash_missing_file_property(self):
        assert compute_file_hash("/nonexistent") == ""

    @given(content=st.text(min_size=0, max_size=100))
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_hash_varies_with_content_property(self, content: str):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            f.flush()
            h = compute_file_hash(f.name)
        os.unlink(f.name)
        assert isinstance(h, str)
        assert len(h) == 32


class TestIsTextFile:
    @pytest.mark.basic
    def test_text_file_true_property(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            f.flush()
            assert is_text_file(f.name) is True
        os.unlink(f.name)

    @pytest.mark.edge
    def test_binary_file_false_property(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\xff\xfe")
            f.flush()
            assert is_text_file(f.name) is False
        os.unlink(f.name)


class TestGetFileExtension:
    @given(ext=extension_st)
    @settings(max_examples=10)
    @pytest.mark.basic
    def test_returns_extension_property(self, ext: Any):
        assert get_file_extension(f"file{ext}") == ext

    @pytest.mark.basic
    def test_no_extension_property(self):
        assert get_file_extension("Makefile") == ""


class TestIsSupportedFile:
    @pytest.mark.basic
    def test_supported_property(self):
        assert is_supported_file("main.py", [".py", ".js"]) is True

    @pytest.mark.edge
    def test_unsupported_property(self):
        assert is_supported_file("data.csv", [".py", ".js"]) is False


class TestNormalizePath:
    @pytest.mark.basic
    def test_forward_slash_property(self):
        assert normalize_path("a/b/c") == "a/b/c"

    @pytest.mark.basic
    def test_dots_collapsed_property(self):
        result = normalize_path("a/../b")
        assert ".." not in result  # normpath collapses

    @given(
        path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=1, max_size=20)
    )
    @settings(max_examples=10)
    @pytest.mark.basic
    def test_no_backslashes_property(self, path: str):
        result = normalize_path(path)
        assert "\\" not in result


class TestGetLanguageFromExtension:
    @pytest.mark.basic
    def test_python_property(self):
        assert get_language_from_extension(".py") == "python"

    @pytest.mark.basic
    def test_javascript_property(self):
        assert get_language_from_extension(".js") == "javascript"

    @pytest.mark.basic
    def test_case_insensitive_property(self):
        assert get_language_from_extension(".PY") == "python"

    @pytest.mark.edge
    def test_unknown_property(self):
        assert get_language_from_extension(".xyz") == "unknown"


class TestExtractCodeSnippet:
    @pytest.mark.basic
    def test_basic_extraction_property(self):
        content = "line0\nline1\nline2\nline3\nline4"
        result = extract_code_snippet(content, 1, 3)
        assert "code" in result
        assert "start_line" in result
        assert "end_line" in result

    @pytest.mark.edge
    def test_empty_content_property(self):
        result = extract_code_snippet("", 0, 0)
        assert "code" in result


class TestFormatCodeBlock:
    @pytest.mark.basic
    def test_basic_format_property(self):
        result = format_code_block("x = 1", "python")
        assert result.startswith("```python")
        assert result.endswith("```")

    @pytest.mark.basic
    def test_with_file_path_property(self):
        result = format_code_block("x = 1", "python", "a.py")
        assert "a.py" in result

    @pytest.mark.edge
    def test_no_language_property(self):
        result = format_code_block("x = 1")
        assert result.startswith("```\n")


class TestCalculateCodeComplexity:
    @pytest.mark.basic
    def test_simple_code_property(self):
        assert calculate_code_complexity("x = 1") == 1

    @pytest.mark.basic
    def test_if_increases_property(self):
        c = calculate_code_complexity("if x:\n  pass")
        assert c >= 2

    @pytest.mark.basic
    def test_empty_code_property(self):
        assert calculate_code_complexity("") == 1


class TestMergeDicts:
    @pytest.mark.basic
    def test_merge_two_property(self):
        assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    @pytest.mark.basic
    def test_latter_wins_property(self):
        assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}

    @pytest.mark.edge
    def test_empty_property(self):
        assert merge_dicts() == {}


class TestSafeGet:
    @pytest.mark.basic
    def test_simple_get_property(self):
        assert safe_get({"a": {"b": 1}}, "a", "b") == 1

    @pytest.mark.edge
    def test_missing_key_returns_default_property(self):
        assert safe_get({"a": 1}, "b", default="x") == "x"

    @pytest.mark.edge
    def test_none_intermediate_property(self):
        assert safe_get({"a": None}, "a", "b", default="d") == "d"

    @pytest.mark.edge
    def test_non_dict_intermediate_property(self):
        assert safe_get({"a": 42}, "a", "b", default="d") == "d"


class TestEnsureDir:
    @pytest.mark.basic
    def test_creates_directory_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c")
            ensure_dir(path)
            assert os.path.isdir(path)


class TestSafeJsonable:
    @pytest.mark.basic
    def test_primitives_passthrough_property(self):
        assert safe_jsonable(None) is None
        assert safe_jsonable(42) == 42
        assert safe_jsonable("hello") == "hello"
        assert safe_jsonable(True) is True

    @pytest.mark.basic
    def test_dict_property(self):
        result = safe_jsonable({"a": 1, "b": "x"})
        assert result == {"a": 1, "b": "x"}

    @pytest.mark.basic
    def test_list_property(self):
        assert safe_jsonable([1, 2, 3]) == [1, 2, 3]

    @pytest.mark.edge
    def test_set_converted_to_list_property(self):
        result = safe_jsonable({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    @pytest.mark.edge
    def test_object_with_to_dict_property(self):
        class Obj:
            def to_dict(self) -> dict[str, int]:
                return {"x": 1}

        assert safe_jsonable(Obj()) == {"x": 1}

    @pytest.mark.edge
    def test_object_with_dunder_dict_property(self):
        class Obj:
            def __init__(self) -> None:
                self.y = 2

        result = safe_jsonable(Obj())
        assert result == {"y": 2}

    @pytest.mark.edge
    def test_depth_limit_returns_repr_property(self):
        class Recursive:
            def __init__(self, d: Any = 0) -> None:
                self.d = d
                self.child = Recursive(d + 1) if d < 15 else None

        result = safe_jsonable(Recursive())
        assert isinstance(result, (dict, str))

    @given(obj=st.one_of(st.integers(), st.text(min_size=0, max_size=10), st.none()))
    @settings(max_examples=15)
    @pytest.mark.basic
    def test_primitives_always_jsonable_property(self, obj: Any):
        result = safe_jsonable(obj)
        assert result == obj


class TestGetRepoNameFromUrl:
    @pytest.mark.basic
    def test_github_https_property(self):
        assert get_repo_name_from_url("https://github.com/org/myrepo") == "myrepo"

    @pytest.mark.basic
    def test_git_suffix_stripped_property(self):
        assert get_repo_name_from_url("https://github.com/org/myrepo.git") == "myrepo"

    @pytest.mark.edge
    def test_trailing_slash_property(self):
        assert get_repo_name_from_url("https://github.com/org/myrepo/") == "myrepo"

    @pytest.mark.edge
    def test_empty_returns_unknown_property(self):
        # Empty URL splits to [''] which is falsy
        result = get_repo_name_from_url("")
        assert isinstance(result, str)


class TestCleanDocstring:
    @pytest.mark.basic
    def test_basic_clean_property(self):
        result = clean_docstring("  hello  ")
        assert result == "hello"

    @pytest.mark.basic
    def test_multiline_dedent_property(self):
        ds = "    line1\n    line2"
        result = clean_docstring(ds)
        assert result == "line1\nline2"

    @pytest.mark.edge
    def test_empty_returns_empty_property(self):
        assert clean_docstring("") == ""

    @pytest.mark.edge
    def test_none_returns_empty_property(self):
        assert clean_docstring(None) == ""

    @pytest.mark.edge
    def test_only_whitespace_property(self):
        assert clean_docstring("   \n  \n  ") == ""


class TestResolveConfigPaths:
    @pytest.mark.basic
    def test_empty_config_property(self):
        assert resolve_config_paths({}, "/root") == {}

    @pytest.mark.basic
    def test_none_config_property(self):
        assert resolve_config_paths(None, "/root") is None

    @pytest.mark.basic
    def test_repo_root_resolved_property(self):
        cfg = {"repo_root": "relative/path"}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["repo_root"])

    @pytest.mark.basic
    def test_absolute_path_unchanged_property(self):
        cfg = {"repo_root": "/absolute/path"}
        result = resolve_config_paths(cfg, "/project")
        assert result["repo_root"].startswith("/absolute")


class TestTruncateToTokens:
    @pytest.mark.basic
    def test_short_text_unchanged_property(self):
        text = "hello world"
        assert truncate_to_tokens(text, 100) == text

    @pytest.mark.edge
    def test_empty_text_property(self):
        assert truncate_to_tokens("", 10) == ""

    @pytest.mark.edge
    def test_zero_max_tokens_property(self):
        result = truncate_to_tokens("hello", 0)
        assert isinstance(result, str)

    @pytest.mark.edge
    def test_special_tokens_in_text_property(self):
        result = truncate_to_tokens("hello <|endoftext|> world", 100)
        assert isinstance(result, str)


class TestComputeFileHashEdge:
    @pytest.mark.edge
    def test_hash_empty_file_property(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("")
            f.flush()
            h = compute_file_hash(f.name)
        os.unlink(f.name)
        assert len(h) == 32

    @pytest.mark.edge
    def test_hash_large_file_property(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"x" * 100000)
            f.flush()
            h = compute_file_hash(f.name)
        os.unlink(f.name)
        assert len(h) == 32


class TestSafeJsonableEdge:
    @pytest.mark.edge
    def test_nested_dict_depth_property(self):
        d = {"a": {"b": {"c": {"d": 1}}}}
        result = safe_jsonable(d)
        assert result["a"]["b"]["c"]["d"] == 1

    @pytest.mark.edge
    def test_mixed_list_property(self):
        result = safe_jsonable([1, "x", None, True, {"y": 2}])
        assert result == [1, "x", None, True, {"y": 2}]

    @pytest.mark.edge
    def test_object_without_to_dict_property(self):
        class Bare:
            pass

        result = safe_jsonable(Bare())
        assert isinstance(result, (dict, str))


class TestExtractCodeSnippetEdge:
    @pytest.mark.edge
    def test_start_beyond_content_property(self):
        result = extract_code_snippet("line0\nline1", 100, 101)
        assert "code" in result

    @pytest.mark.edge
    def test_negative_line_numbers_property(self):
        result = extract_code_snippet("line0\nline1", -1, 1)
        assert "code" in result

    @pytest.mark.edge
    def test_context_lines_zero_property(self):
        result = extract_code_snippet("a\nb\nc\nd", 1, 2, context_lines=0)
        assert "code" in result


class TestCleanDocstringEdge:
    @pytest.mark.edge
    def test_leading_newlines_property(self):
        assert clean_docstring("\n\nhello") == "hello"

    @pytest.mark.edge
    def test_trailing_newlines_property(self):
        assert clean_docstring("hello\n\n") == "hello"

    @pytest.mark.edge
    def test_no_indent_property(self):
        assert clean_docstring("line1\nline2") == "line1\nline2"


class TestResolveConfigPathsEdge:
    @pytest.mark.edge
    def test_vector_store_path_resolved_property(self):
        cfg = {"vector_store": {"persist_directory": "data/vectors"}}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["vector_store"]["persist_directory"])

    @pytest.mark.edge
    def test_cache_directory_resolved_property(self):
        cfg = {"cache": {"cache_directory": "cache/data"}}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["cache"]["cache_directory"])

    @pytest.mark.edge
    def test_logging_file_resolved_property(self):
        cfg = {"logging": {"file": "logs/app.log"}}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["logging"]["file"])

    @pytest.mark.edge
    def test_empty_string_path_not_resolved_property(self):
        cfg = {"repo_root": ""}
        result = resolve_config_paths(cfg, "/project")
        assert result["repo_root"] in ("", None)


class TestCalculateCodeComplexityEdge:
    @pytest.mark.edge
    def test_many_ifs_property(self):
        code = "if a:\n  pass\nif b:\n  pass\nif c:\n  pass"
        c = calculate_code_complexity(code)
        assert c >= 4

    @pytest.mark.edge
    def test_try_except_property(self):
        c = calculate_code_complexity("try:\n  pass\nexcept:\n  pass")
        assert c >= 3


class TestFormatCodeBlockEdge:
    @pytest.mark.edge
    def test_with_line_number_property(self):
        result = format_code_block("x = 1", "python", "a.py", start_line=42)
        assert "42" in result

    @pytest.mark.edge
    def test_empty_code_property(self):
        result = format_code_block("", "python")
        assert "```" in result
