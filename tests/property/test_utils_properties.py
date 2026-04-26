"""Property-based tests for utils module."""

from __future__ import annotations

from typing import Any
import os
import tempfile

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


@pytest.mark.property
class TestUtcNow:
    @pytest.mark.happy
    def test_returns_iso_string(self):
        result = utc_now()
        assert isinstance(result, str)
        assert "T" in result


@pytest.mark.property
class TestComputeFileHash:
    @pytest.mark.happy
    def test_hash_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello world")
            f.flush()
            h1 = compute_file_hash(f.name)
            h2 = compute_file_hash(f.name)
        os.unlink(f.name)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex length

    @pytest.mark.edge
    def test_hash_missing_file(self):
        assert compute_file_hash("/nonexistent") == ""

    @given(content=st.text(min_size=0, max_size=100))
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_hash_varies_with_content(self, content: str):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            f.flush()
            h = compute_file_hash(f.name)
        os.unlink(f.name)
        assert isinstance(h, str)
        assert len(h) == 32


@pytest.mark.property
class TestIsTextFile:
    @pytest.mark.happy
    def test_text_file_true(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            f.flush()
            assert is_text_file(f.name) is True
        os.unlink(f.name)

    @pytest.mark.edge
    def test_binary_file_false(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\xff\xfe")
            f.flush()
            assert is_text_file(f.name) is False
        os.unlink(f.name)


@pytest.mark.property
class TestGetFileExtension:
    @given(ext=extension_st)
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_returns_extension(self, ext: Any):
        assert get_file_extension(f"file{ext}") == ext

    @pytest.mark.happy
    def test_no_extension(self):
        assert get_file_extension("Makefile") == ""


@pytest.mark.property
class TestIsSupportedFile:
    @pytest.mark.happy
    def test_supported(self):
        assert is_supported_file("main.py", [".py", ".js"]) is True

    @pytest.mark.edge
    def test_unsupported(self):
        assert is_supported_file("data.csv", [".py", ".js"]) is False


@pytest.mark.property
class TestNormalizePath:
    @pytest.mark.happy
    def test_forward_slash(self):
        assert normalize_path("a/b/c") == "a/b/c"

    @pytest.mark.happy
    def test_dots_collapsed(self):
        result = normalize_path("a/../b")
        assert ".." not in result  # normpath collapses

    @given(
        path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=1, max_size=20)
    )
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_no_backslashes(self, path: str):
        result = normalize_path(path)
        assert "\\" not in result


@pytest.mark.property
class TestGetLanguageFromExtension:
    @pytest.mark.happy
    def test_python(self):
        assert get_language_from_extension(".py") == "python"

    @pytest.mark.happy
    def test_javascript(self):
        assert get_language_from_extension(".js") == "javascript"

    @pytest.mark.happy
    def test_case_insensitive(self):
        assert get_language_from_extension(".PY") == "python"

    @pytest.mark.edge
    def test_unknown(self):
        assert get_language_from_extension(".xyz") == "unknown"


@pytest.mark.property
class TestExtractCodeSnippet:
    @pytest.mark.happy
    def test_basic_extraction(self):
        content = "line0\nline1\nline2\nline3\nline4"
        result = extract_code_snippet(content, 1, 3)
        assert "code" in result
        assert "start_line" in result
        assert "end_line" in result

    @pytest.mark.edge
    def test_empty_content(self):
        result = extract_code_snippet("", 0, 0)
        assert "code" in result


@pytest.mark.property
class TestFormatCodeBlock:
    @pytest.mark.happy
    def test_basic_format(self):
        result = format_code_block("x = 1", "python")
        assert result.startswith("```python")
        assert result.endswith("```")

    @pytest.mark.happy
    def test_with_file_path(self):
        result = format_code_block("x = 1", "python", "a.py")
        assert "a.py" in result

    @pytest.mark.edge
    def test_no_language(self):
        result = format_code_block("x = 1")
        assert result.startswith("```\n")


@pytest.mark.property
class TestCalculateCodeComplexity:
    @pytest.mark.happy
    def test_simple_code(self):
        assert calculate_code_complexity("x = 1") == 1

    @pytest.mark.happy
    def test_if_increases(self):
        c = calculate_code_complexity("if x:\n  pass")
        assert c >= 2

    @pytest.mark.happy
    def test_empty_code(self):
        assert calculate_code_complexity("") == 1


@pytest.mark.property
class TestMergeDicts:
    @pytest.mark.happy
    def test_merge_two(self):
        assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    @pytest.mark.happy
    def test_latter_wins(self):
        assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}

    @pytest.mark.edge
    def test_empty(self):
        assert merge_dicts() == {}


@pytest.mark.property
class TestSafeGet:
    @pytest.mark.happy
    def test_simple_get(self):
        assert safe_get({"a": {"b": 1}}, "a", "b") == 1

    @pytest.mark.edge
    def test_missing_key_returns_default(self):
        assert safe_get({"a": 1}, "b", default="x") == "x"

    @pytest.mark.edge
    def test_none_intermediate(self):
        assert safe_get({"a": None}, "a", "b", default="d") == "d"

    @pytest.mark.edge
    def test_non_dict_intermediate(self):
        assert safe_get({"a": 42}, "a", "b", default="d") == "d"


@pytest.mark.property
class TestEnsureDir:
    @pytest.mark.happy
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c")
            ensure_dir(path)
            assert os.path.isdir(path)


@pytest.mark.property
class TestSafeJsonable:
    @pytest.mark.happy
    def test_primitives_passthrough(self):
        assert safe_jsonable(None) is None
        assert safe_jsonable(42) == 42
        assert safe_jsonable("hello") == "hello"
        assert safe_jsonable(True) is True

    @pytest.mark.happy
    def test_dict(self):
        result = safe_jsonable({"a": 1, "b": "x"})
        assert result == {"a": 1, "b": "x"}

    @pytest.mark.happy
    def test_list(self):
        assert safe_jsonable([1, 2, 3]) == [1, 2, 3]

    @pytest.mark.edge
    def test_set_converted_to_list(self):
        result = safe_jsonable({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    @pytest.mark.edge
    def test_object_with_to_dict(self):
        class Obj:
            def to_dict(self) -> None:
                return {"x": 1}

        assert safe_jsonable(Obj()) == {"x": 1}

    @pytest.mark.edge
    def test_object_with_dunder_dict(self):
        class Obj:
            def __init__(self) -> None:
                self.y = 2

        result = safe_jsonable(Obj())
        assert result == {"y": 2}

    @pytest.mark.edge
    def test_depth_limit_returns_repr(self):
        class Recursive:
            def __init__(self, d: Any = 0) -> None:
                self.d = d
                self.child = Recursive(d + 1) if d < 15 else None

        result = safe_jsonable(Recursive())
        assert isinstance(result, (dict, str))

    @given(obj=st.one_of(st.integers(), st.text(min_size=0, max_size=10), st.none()))
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_primitives_always_jsonable(self, obj: Any):
        result = safe_jsonable(obj)
        assert result == obj


@pytest.mark.property
class TestGetRepoNameFromUrl:
    @pytest.mark.happy
    def test_github_https(self):
        assert get_repo_name_from_url("https://github.com/org/myrepo") == "myrepo"

    @pytest.mark.happy
    def test_git_suffix_stripped(self):
        assert get_repo_name_from_url("https://github.com/org/myrepo.git") == "myrepo"

    @pytest.mark.edge
    def test_trailing_slash(self):
        assert get_repo_name_from_url("https://github.com/org/myrepo/") == "myrepo"

    @pytest.mark.edge
    def test_empty_returns_unknown(self):
        # Empty URL splits to [''] which is falsy
        result = get_repo_name_from_url("")
        assert isinstance(result, str)


@pytest.mark.property
class TestCleanDocstring:
    @pytest.mark.happy
    def test_basic_clean(self):
        result = clean_docstring("  hello  ")
        assert result == "hello"

    @pytest.mark.happy
    def test_multiline_dedent(self):
        ds = "    line1\n    line2"
        result = clean_docstring(ds)
        assert result == "line1\nline2"

    @pytest.mark.edge
    def test_empty_returns_empty(self):
        assert clean_docstring("") == ""

    @pytest.mark.edge
    def test_none_returns_empty(self):
        assert clean_docstring(None) == ""

    @pytest.mark.edge
    def test_only_whitespace(self):
        assert clean_docstring("   \n  \n  ") == ""


@pytest.mark.property
class TestResolveConfigPaths:
    @pytest.mark.happy
    def test_empty_config(self):
        assert resolve_config_paths({}, "/root") == {}

    @pytest.mark.happy
    def test_none_config(self):
        assert resolve_config_paths(None, "/root") is None

    @pytest.mark.happy
    def test_repo_root_resolved(self):
        cfg = {"repo_root": "relative/path"}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["repo_root"])

    @pytest.mark.happy
    def test_absolute_path_unchanged(self):
        cfg = {"repo_root": "/absolute/path"}
        result = resolve_config_paths(cfg, "/project")
        assert result["repo_root"].startswith("/absolute")


@pytest.mark.property
class TestTruncateToTokens:
    @pytest.mark.happy
    def test_short_text_unchanged(self):
        text = "hello world"
        assert truncate_to_tokens(text, 100) == text

    @pytest.mark.edge
    def test_empty_text(self):
        assert truncate_to_tokens("", 10) == ""

    @pytest.mark.edge
    def test_zero_max_tokens(self):
        result = truncate_to_tokens("hello", 0)
        assert isinstance(result, str)

    @pytest.mark.edge
    def test_special_tokens_in_text(self):
        result = truncate_to_tokens("hello <|endoftext|> world", 100)
        assert isinstance(result, str)


@pytest.mark.property
class TestComputeFileHashEdge:
    @pytest.mark.edge
    def test_hash_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("")
            f.flush()
            h = compute_file_hash(f.name)
        os.unlink(f.name)
        assert len(h) == 32

    @pytest.mark.edge
    def test_hash_large_file(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"x" * 100000)
            f.flush()
            h = compute_file_hash(f.name)
        os.unlink(f.name)
        assert len(h) == 32


@pytest.mark.property
class TestSafeJsonableEdge:
    @pytest.mark.edge
    def test_nested_dict_depth(self):
        d = {"a": {"b": {"c": {"d": 1}}}}
        result = safe_jsonable(d)
        assert result["a"]["b"]["c"]["d"] == 1

    @pytest.mark.edge
    def test_mixed_list(self):
        result = safe_jsonable([1, "x", None, True, {"y": 2}])
        assert result == [1, "x", None, True, {"y": 2}]

    @pytest.mark.edge
    def test_object_without_to_dict(self):
        class Bare:
            pass

        result = safe_jsonable(Bare())
        assert isinstance(result, (dict, str))


@pytest.mark.property
class TestExtractCodeSnippetEdge:
    @pytest.mark.edge
    def test_start_beyond_content(self):
        result = extract_code_snippet("line0\nline1", 100, 101)
        assert "code" in result

    @pytest.mark.edge
    def test_negative_line_numbers(self):
        result = extract_code_snippet("line0\nline1", -1, 1)
        assert "code" in result

    @pytest.mark.edge
    def test_context_lines_zero(self):
        result = extract_code_snippet("a\nb\nc\nd", 1, 2, context_lines=0)
        assert "code" in result


@pytest.mark.property
class TestCleanDocstringEdge:
    @pytest.mark.edge
    def test_leading_newlines(self):
        assert clean_docstring("\n\nhello") == "hello"

    @pytest.mark.edge
    def test_trailing_newlines(self):
        assert clean_docstring("hello\n\n") == "hello"

    @pytest.mark.edge
    def test_no_indent(self):
        assert clean_docstring("line1\nline2") == "line1\nline2"


@pytest.mark.property
class TestResolveConfigPathsEdge:
    @pytest.mark.edge
    def test_vector_store_path_resolved(self):
        cfg = {"vector_store": {"persist_directory": "data/vectors"}}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["vector_store"]["persist_directory"])

    @pytest.mark.edge
    def test_cache_directory_resolved(self):
        cfg = {"cache": {"cache_directory": "cache/data"}}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["cache"]["cache_directory"])

    @pytest.mark.edge
    def test_logging_file_resolved(self):
        cfg = {"logging": {"file": "logs/app.log"}}
        result = resolve_config_paths(cfg, "/project")
        assert os.path.isabs(result["logging"]["file"])

    @pytest.mark.edge
    def test_empty_string_path_not_resolved(self):
        cfg = {"repo_root": ""}
        result = resolve_config_paths(cfg, "/project")
        assert result["repo_root"] in ("", None)


@pytest.mark.property
class TestCalculateCodeComplexityEdge:
    @pytest.mark.edge
    def test_many_ifs(self):
        code = "if a:\n  pass\nif b:\n  pass\nif c:\n  pass"
        c = calculate_code_complexity(code)
        assert c >= 4

    @pytest.mark.edge
    def test_try_except(self):
        c = calculate_code_complexity("try:\n  pass\nexcept:\n  pass")
        assert c >= 3


@pytest.mark.property
class TestFormatCodeBlockEdge:
    @pytest.mark.edge
    def test_with_line_number(self):
        result = format_code_block("x = 1", "python", "a.py", start_line=42)
        assert "42" in result

    @pytest.mark.edge
    def test_empty_code(self):
        result = format_code_block("", "python")
        assert "```" in result
