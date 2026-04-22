"""Snapshot tests for IR data model serialization contracts."""

from __future__ import annotations

import pytest

from fastcode.semantic_ir import IRDocument, IRAttachment, IREdge, IROccurrence, IRSnapshot, IRSymbol


@pytest.mark.snapshot
@pytest.mark.happy
class TestIRSnapshotContract:

    def test_ir_document_serialization(self, snapshot):
        doc = IRDocument(
            doc_id="doc:abc123",
            path="src/main.py",
            language="python",
            blob_oid="a" * 40,
            content_hash="b" * 40,
            source_set={"ast", "scip"},
        )
        snapshot.assert_match(doc.to_dict())

    def test_ir_symbol_serialization(self, snapshot):
        sym = IRSymbol(
            symbol_id="scip:snap:1:pkg.main.MyClass.my_method",
            external_symbol_id="pkg.main.MyClass.my_method",
            path="src/main.py",
            display_name="my_method",
            kind="method",
            language="python",
            qualified_name="pkg.main.MyClass.my_method",
            signature="def my_method(self, x: int) -> str",
            start_line=42,
            start_col=8,
            end_line=50,
            end_col=20,
            source_priority=100,
            source_set={"scip"},
            metadata={"scip": True, "source": "scip"},
        )
        snapshot.assert_match(sym.to_dict())

    def test_ir_occurrence_serialization(self, snapshot):
        occ = IROccurrence(
            occurrence_id="occ:abc123",
            symbol_id="scip:snap:1:pkg.main.foo",
            doc_id="doc:abc123",
            role="definition",
            start_line=10,
            start_col=0,
            end_line=20,
            end_col=0,
            source="scip",
            metadata={"source": "scip", "confidence": "precise"},
        )
        snapshot.assert_match(occ.to_dict())

    def test_ir_edge_serialization(self, snapshot):
        edge = IREdge(
            edge_id="edge:abc123",
            src_id="doc:abc123",
            dst_id="scip:snap:1:pkg.main.MyClass",
            edge_type="contain",
            source="scip",
            confidence="precise",
            doc_id="doc:abc123",
            metadata={"extractor": "fastcode.adapters.scip_to_ir"},
        )
        snapshot.assert_match(edge.to_dict())

    def test_ir_attachment_serialization(self, snapshot):
        attachment = IRAttachment(
            attachment_id="att:abc123",
            target_id="sym:abc123",
            target_type="symbol",
            attachment_type="embedding",
            source="fc_embedding",
            confidence="derived",
            payload={"vector": [0.1, 0.2], "text": "Function main"},
            metadata={"ast_element_id": "elem_main"},
        )
        snapshot.assert_match(attachment.to_dict())

    def test_ir_snapshot_full_serialization(self, snapshot):
        doc = IRDocument(
            doc_id="doc:1",
            path="app.py",
            language="python",
            source_set={"ast"},
        )
        sym = IRSymbol(
            symbol_id="sym:1",
            external_symbol_id=None,
            path="app.py",
            display_name="main",
            kind="function",
            language="python",
            start_line=1,
            source_priority=10,
            source_set={"ast"},
            metadata={"source": "ast"},
        )
        occ = IROccurrence(
            occurrence_id="occ:1",
            symbol_id="sym:1",
            doc_id="doc:1",
            role="definition",
            start_line=1,
            start_col=0,
            end_line=5,
            end_col=0,
            source="ast",
            metadata={},
        )
        edge = IREdge(
            edge_id="e:1",
            src_id="doc:1",
            dst_id="sym:1",
            edge_type="contain",
            source="ast",
            confidence="resolved",
        )
        attachment = IRAttachment(
            attachment_id="att:1",
            target_id="sym:1",
            target_type="symbol",
            attachment_type="summary",
            source="fc_structure",
            confidence="derived",
            payload={"text": "Entry point"},
            metadata={"ast_element_id": "elem_main"},
        )
        snap = IRSnapshot(
            repo_name="my-repo",
            snapshot_id="snap:my-repo:abc123",
            branch="main",
            commit_id="abc123",
            documents=[doc],
            symbols=[sym],
            occurrences=[occ],
            edges=[edge],
            attachments=[attachment],
            metadata={"source_modes": ["ast"]},
        )
        snapshot.assert_match(snap.to_dict())

    @pytest.mark.edge
    def test_ir_snapshot_empty_collections_serialization(self, snapshot):
        """EDGE: empty snapshot with no documents, symbols, occurrences, edges."""
        snap = IRSnapshot(
            repo_name="empty-repo",
            snapshot_id="snap:empty:000",
        )
        snapshot.assert_match(snap.to_dict())
