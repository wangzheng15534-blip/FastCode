"""Regression tests for .h header file classification as C vs C++.

The C/C++ resolver must correctly classify .h files based on their
content and context (e.g., adjacent .cpp files, C++ keywords in source).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastcode.indexer import CodeElement
from fastcode.parser import CodeParser
from fastcode.semantic_ir import IRCodeUnit, IRSnapshot
from fastcode.semantic_resolvers.c_family import CppSemanticResolver, CSemanticResolver
from fastcode.utils.paths import (
    get_language_from_extension,
    infer_language_from_file_context,
)


class TestHeaderClassification:
    """Verify that .h files are handled correctly by both C and C++ resolvers."""

    def test_c_resolver_accepts_dot_h_files(self) -> None:
        """CSemanticResolver must accept .h extension files."""
        resolver = CSemanticResolver()
        assert "c" in resolver.language
        # .h is ambiguous but the C resolver should handle it
        assert ".h" not in getattr(resolver, "_excluded_extensions", set())

    def test_cpp_resolver_accepts_dot_hpp_files(self) -> None:
        """CppSemanticResolver must accept .hpp extension files."""
        resolver = CppSemanticResolver()
        assert resolver.language == "cpp"

    def test_c_and_cpp_resolvers_have_different_languages(self) -> None:
        """C and C++ resolvers must have distinct language identifiers."""
        c_resolver = CSemanticResolver()
        cpp_resolver = CppSemanticResolver()
        assert c_resolver.language != cpp_resolver.language
        assert c_resolver.language == "c"
        assert cpp_resolver.language == "cpp"

    def test_c_resolver_spec_requires_clang(self) -> None:
        """C resolver spec must list clang as required tool."""
        resolver = CSemanticResolver()
        assert "clang" in resolver.spec.required_tools

    def test_cpp_resolver_spec_requires_clangpp(self) -> None:
        """C++ resolver spec must list clang++ as required tool."""
        resolver = CppSemanticResolver()
        assert "clang++" in resolver.spec.required_tools

    def test_c_resolver_include_resolution_uses_relative_paths(self) -> None:
        """C resolver include resolution must support relative paths like
        ../include/util.h resolved against the source file location.
        """
        # This test exercises the same path as test_c_resolver_emits_relative_include_relation
        # but from the header classification perspective

        file_main = IRCodeUnit(
            unit_id="doc:snap:1:src/main.c",
            kind="file",
            path="src/main.c",
            language="c",
            display_name="src/main.c",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        file_header = IRCodeUnit(
            unit_id="doc:snap:1:include/util.h",
            kind="file",
            path="include/util.h",
            language="c",
            display_name="include/util.h",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            units=[file_main, file_header],
            supports=[],
            relations=[],
        )
        elements = [
            CodeElement(
                id="file:main",
                type="file",
                name="src/main.c",
                file_path="src/main.c",
                relative_path="src/main.c",
                language="c",
                start_line=1,
                end_line=10,
                code='#include "../include/util.h"',
                signature=None,
                docstring=None,
                summary=None,
                metadata={"imports": [{"module": "../include/util.h", "level": 0}]},
            )
        ]

        resolver = CSemanticResolver()
        patch = resolver.resolve(
            snapshot=snapshot,
            elements=elements,
            target_paths={"src/main.c"},
            legacy_graph_builder=None,
        )

        assert len(patch.relations) == 1
        assert patch.relations[0].dst_unit_id == file_header.unit_id


class TestHeaderExtensionMapping:
    """Test the extension-to-language mapping for header files."""

    @pytest.mark.regression
    def test_dot_h_is_currently_classified_as_c(self) -> None:
        """Baseline: get_language_from_extension('.h') returns 'c'.

        This documents CURRENT behavior. When context-aware classification
        is implemented, update this test or replace with context-parameterized
        tests.
        """
        assert get_language_from_extension(".h") == "c"

    @pytest.mark.regression
    def test_dot_hpp_is_classified_as_cpp(self) -> None:
        """Unambiguous: .hpp must always be cpp."""
        assert get_language_from_extension(".hpp") == "cpp"

    def test_dot_hh_is_classified_as_cpp(self) -> None:
        """Unambiguous: .hh must always be cpp."""
        assert get_language_from_extension(".hh") == "cpp"

    def test_dot_hxx_is_classified_as_cpp(self) -> None:
        """Unambiguous: .hxx must always be cpp."""
        assert get_language_from_extension(".hxx") == "cpp"


class TestHeaderResolverDispatch:
    """Test that resolvers correctly filter elements by language for .h files."""

    @pytest.mark.regression
    def test_c_resolver_applicable_when_dot_h_element_has_language_c(self) -> None:
        """CSemanticResolver must be applicable when a .h element has language='c'."""
        resolver = CSemanticResolver()
        file_unit = IRCodeUnit(
            unit_id="doc:snap:1:include/util.h",
            kind="file",
            path="include/util.h",
            language="c",
            display_name="include/util.h",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            units=[file_unit],
            supports=[],
            relations=[],
        )
        element = CodeElement(
            id="file:util_h",
            type="file",
            name="include/util.h",
            file_path="include/util.h",
            relative_path="include/util.h",
            language="c",
            start_line=1,
            end_line=10,
            code="#ifndef UTIL_H\n#define UTIL_H\nvoid util(void);\n#endif",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
        )
        assert resolver.applicable(
            snapshot=snapshot,
            elements=[element],
            target_paths={"include/util.h"},
        )

    @pytest.mark.regression
    def test_cpp_resolver_not_applicable_for_plain_c_style_dot_h(self) -> None:
        """CppSemanticResolver must stay off plain C-style `.h` headers."""
        resolver = CppSemanticResolver()
        file_unit = IRCodeUnit(
            unit_id="doc:snap:1:include/util.h",
            kind="file",
            path="include/util.h",
            language="c",
            display_name="include/util.h",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            units=[file_unit],
            supports=[],
            relations=[],
        )
        element = CodeElement(
            id="file:util_h",
            type="file",
            name="include/util.h",
            file_path="include/util.h",
            relative_path="include/util.h",
            language="c",
            start_line=1,
            end_line=10,
            code="#ifndef UTIL_H\n#define UTIL_H\nvoid util(void);\n#endif",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
        )
        assert not resolver.applicable(
            snapshot=snapshot,
            elements=[element],
            target_paths={"include/util.h"},
        )


class TestCppIncludeOfDotHResolution:
    """Test include resolution when a .cpp file includes a .h file."""

    @pytest.mark.regression
    def test_cpp_resolver_resolves_dot_h_include_from_cpp_file(self) -> None:
        """CppSemanticResolver must resolve #include 'util.h' from main.cpp
        to a .h unit, even when the .h unit is tagged language='c'.
        """
        resolver = CppSemanticResolver()
        file_cpp = IRCodeUnit(
            unit_id="doc:snap:1:src/main.cpp",
            kind="file",
            path="src/main.cpp",
            language="cpp",
            display_name="src/main.cpp",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        file_header = IRCodeUnit(
            unit_id="doc:snap:1:include/util.h",
            kind="file",
            path="include/util.h",
            language="c",
            display_name="include/util.h",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            units=[file_cpp, file_header],
            supports=[],
            relations=[],
        )
        element = CodeElement(
            id="file:main_cpp",
            type="file",
            name="src/main.cpp",
            file_path="src/main.cpp",
            relative_path="src/main.cpp",
            language="cpp",
            start_line=1,
            end_line=10,
            code='#include "../include/util.h"',
            signature=None,
            docstring=None,
            summary=None,
            metadata={"imports": [{"module": "../include/util.h", "level": 0}]},
        )
        patch = resolver.resolve(
            snapshot=snapshot,
            elements=[element],
            target_paths={"src/main.cpp"},
            legacy_graph_builder=None,
        )
        assert len(patch.relations) == 1
        assert patch.relations[0].dst_unit_id == file_header.unit_id
        assert patch.relations[0].relation_type == "import"


class TestHeaderAmbiguityContract:
    """Context-aware .h classification must promote C++ headers when warranted."""

    def test_dot_h_with_adjacent_cpp_classified_as_cpp(self) -> None:
        """When a .h file has adjacent .cpp files in the same directory,
        get_language_from_extension (or a context-aware successor) must
        classify it as C++.
        """
        # Snapshot with src/main.cpp alongside src/util.h — util.h should be cpp
        file_cpp = IRCodeUnit(
            unit_id="doc:snap:1:src/main.cpp",
            kind="file",
            path="src/main.cpp",
            language="cpp",
            display_name="src/main.cpp",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        file_header = IRCodeUnit(
            unit_id="doc:snap:1:src/util.h",
            kind="file",
            path="src/util.h",
            language="c",
            display_name="src/util.h",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            units=[file_cpp, file_header],
            supports=[],
            relations=[],
        )
        resolver = CppSemanticResolver()
        element = CodeElement(
            id="file:util_h",
            type="file",
            name="src/util.h",
            file_path="src/util.h",
            relative_path="src/util.h",
            language="c",
            start_line=1,
            end_line=10,
            code="#ifndef UTIL_H\n#define UTIL_H\nvoid util();\n#endif",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
        )
        assert resolver.applicable(
            snapshot=snapshot,
            elements=[element],
            target_paths={"src/util.h"},
        )

    def test_dot_h_with_cpp_keywords_classified_as_cpp(self) -> None:
        """When a .h file contains C++ keywords (class, namespace, template),
        it must be classified as C++ regardless of adjacent files.
        """
        resolver = CppSemanticResolver()
        file_header = IRCodeUnit(
            unit_id="doc:snap:1:include/widget.h",
            kind="file",
            path="include/widget.h",
            language="c",
            display_name="include/widget.h",
            source_set={"fc_structure"},
            metadata={"source": "fc_structure"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:1",
            units=[file_header],
            supports=[],
            relations=[],
        )
        element = CodeElement(
            id="file:widget_h",
            type="file",
            name="include/widget.h",
            file_path="include/widget.h",
            relative_path="include/widget.h",
            language="c",
            start_line=1,
            end_line=10,
            code="#ifndef WIDGET_H\n#define WIDGET_H\nclass Widget {\npublic:\n    virtual void run() = 0;\n};\n#endif",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
        )
        assert resolver.applicable(
            snapshot=snapshot,
            elements=[element],
            target_paths={"include/widget.h"},
        )


class TestHeaderParserClassification:
    """Header classification must apply before indexing emits element language."""

    def test_infer_language_from_file_context_uses_adjacent_cpp(
        self, tmp_path: Path
    ) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        header_path = src_dir / "util.h"
        header_path.write_text("#ifndef UTIL_H\n#define UTIL_H\nvoid util();\n#endif")
        (src_dir / "main.cpp").write_text('#include "util.h"\n')

        assert infer_language_from_file_context(str(header_path)) == "cpp"

    def test_parser_classifies_dot_h_with_cpp_keywords_as_cpp(
        self, tmp_path: Path
    ) -> None:
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        header_path = include_dir / "widget.h"
        header_path.write_text(
            "#ifndef WIDGET_H\n#define WIDGET_H\n"
            "class Widget {\npublic:\n    virtual void run() = 0;\n};\n#endif"
        )

        parser = CodeParser(config={})
        parsed = parser.parse_file(str(header_path), header_path.read_text())

        assert parsed is not None
        assert parsed.language == "cpp"


class TestDuplicateExtensionMaps:
    """Verify that all .h extension maps in the codebase agree."""

    @pytest.mark.regression
    def test_all_language_maps_agree_on_dot_h(self) -> None:
        """The .h mapping in paths.py and _compat.py must return the same language."""
        from fastcode.utils._compat import get_language_from_extension as compat_fn

        assert get_language_from_extension(".h") == compat_fn(".h")
