"""Tests for pure snapshot logic."""

from fastcode.retrieval.contracts import Hit, SourceCitation
from fastcode.retrieval.context.snapshot import extract_sources_from_elements
from fastcode.utils.hashing import projection_params_hash
from fastcode.utils.paths import projection_scope_key


class TestProjectionScopeKey:
    def test_deterministic(self):
        key1 = projection_scope_key("global", "snap:test:abc", None, None, None)
        key2 = projection_scope_key("global", "snap:test:abc", None, None, None)
        assert key1 == key2

    def test_different_scope(self):
        key1 = projection_scope_key("global", "snap:test:abc", None, None, None)
        key2 = projection_scope_key("targeted", "snap:test:abc", None, None, None)
        assert key1 != key2

    def test_with_query(self):
        key_no_query = projection_scope_key("global", "snap:test:abc", None, None, None)
        key_with_query = projection_scope_key(
            "global", "snap:test:abc", "how does X work?", None, None
        )
        assert key_no_query != key_with_query

    def test_with_filters(self):
        key_no_filter = projection_scope_key(
            "global", "snap:test:abc", None, None, None
        )
        key_with_filter = projection_scope_key(
            "global", "snap:test:abc", None, None, {"language": "python"}
        )
        assert key_no_filter != key_with_filter


class TestProjectionParamsHash:
    def test_produces_hex_string(self):
        scope_dict = {"scope_kind": "global", "snapshot_id": "snap:test"}
        result = projection_params_hash(scope_dict, "v1")
        assert len(result) == 40  # SHA-1 hex digest
        assert all(c in "0123456789abcdef" for c in result)

    def test_version_affects_hash(self):
        scope_dict = {"scope_kind": "global", "snapshot_id": "snap:test"}
        v1 = projection_params_hash(scope_dict, "v1")
        v2 = projection_params_hash(scope_dict, "v2")
        assert v1 != v2


class TestExtractSourcesFromElements:
    def test_basic(self):
        elements = [
            Hit(
                element_id="func:1",
                element_type="function",
                element_name="func1",
                score=0.0,
                relative_path="src/main.py",
                repo_name="myrepo",
                start_line=10,
                end_line=20,
                total_score=0.0,
            ),
        ]
        sources = extract_sources_from_elements(elements)
        assert sources == (
            SourceCitation(
                repository="myrepo",
                file="src/main.py",
                name="func1",
                element_type="function",
                lines="10-20",
                score=0.0,
            ),
        )

    def test_empty(self):
        assert extract_sources_from_elements([]) == ()
