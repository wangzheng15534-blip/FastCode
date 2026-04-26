"""Tests for pure repo analysis functions."""

from fastcode.core.repo_analysis import (
    format_file_structure,
    generate_structure_based_overview,
    infer_project_type,
    is_key_file,
)
from fastcode.utils.paths import get_language_from_extension


class TestGetLanguageFromExtension:
    def test_python(self):
        assert get_language_from_extension(".py") == "python"

    def test_typescript(self):
        assert get_language_from_extension(".ts") == "typescript"

    def test_go(self):
        assert get_language_from_extension(".go") == "go"

    def test_unknown(self):
        assert get_language_from_extension(".xyz") == "unknown"

    def test_case_insensitive(self):
        assert get_language_from_extension(".PY") == "python"

    def test_jsx(self):
        assert get_language_from_extension(".jsx") == "javascript"

    def test_tsx(self):
        assert get_language_from_extension(".tsx") == "typescript"

    def test_rust(self):
        assert get_language_from_extension(".rs") == "rust"

    def test_csharp(self):
        assert get_language_from_extension(".cs") == "csharp"


class TestIsKeyFile:
    def test_main(self):
        assert is_key_file("src/main.py")

    def test_package_json(self):
        assert is_key_file("package.json")

    def test_dockerfile(self):
        assert is_key_file("Dockerfile")

    def test_regular_file(self):
        assert not is_key_file("src/utils/helper.py")

    def test_index(self):
        assert is_key_file("src/index.ts")

    def test_go_mod(self):
        assert is_key_file("go.mod")

    def test_cargo_toml(self):
        assert is_key_file("Cargo.toml")

    def test_makefile(self):
        assert is_key_file("Makefile")

    def test_nested_config(self):
        assert is_key_file("config/config.yaml")


class TestInferProjectType:
    def test_react(self):
        assert "React" in infer_project_type(["package.json"], {"tsx": 10})

    def test_react_in_key_files(self):
        assert "React" in infer_project_type(["package.json", "src/react-app.tsx"], {})

    def test_vue(self):
        assert "Vue" in infer_project_type(["package.json", "vue.config.js"], {})

    def test_nodejs(self):
        assert "Node.js" in infer_project_type(["package.json"], {})

    def test_python_project(self):
        assert "Python" in infer_project_type(["requirements.txt"], {"python": 10})

    def test_setup_py(self):
        assert "Python" in infer_project_type(["setup.py"], {})

    def test_django(self):
        assert "Django" in infer_project_type(
            ["requirements.txt", "django_settings.py"], {}
        )

    def test_flask(self):
        assert "Flask" in infer_project_type(["requirements.txt", "flask_app.py"], {})

    def test_android(self):
        assert "Android" in infer_project_type(["build.gradle"], {"java": 5})

    def test_ios(self):
        # Actual impl checks "swift" in key_files_str, so key file must contain "swift"
        assert "iOS" in infer_project_type(
            ["Podfile", "ios/SwiftAppDelegate.swift"], {}
        )

    def test_ios_via_swift_in_key_files(self):
        # Also triggers when a file named with "swift" is in key files
        assert "iOS" in infer_project_type(["swift_main.swift"], {})

    def test_dockerfile(self):
        assert "containerized" in infer_project_type(["Dockerfile"], {})

    def test_default(self):
        assert "software project" in infer_project_type([], {})


class TestGenerateStructureBasedOverview:
    def test_basic(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 8, "TypeScript": 2},
            "key_files": ["main.py", "setup.py"],
        }
        overview = generate_structure_based_overview("myrepo", structure)
        assert "myrepo" in overview
        assert "10 files" in overview

    def test_primary_language(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 8, "TypeScript": 2},
            "key_files": [],
        }
        overview = generate_structure_based_overview("repo", structure)
        assert "Python" in overview

    def test_no_languages(self):
        structure = {"total_files": 5, "languages": {}, "key_files": []}
        overview = generate_structure_based_overview("repo", structure)
        assert "unknown" in overview

    def test_multiple_languages(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 5, "JavaScript": 3, "Rust": 2},
            "key_files": [],
        }
        overview = generate_structure_based_overview("repo", structure)
        assert "multiple languages" in overview

    def test_key_entry_points(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 10},
            "key_files": ["main.py", "config.py"],
        }
        overview = generate_structure_based_overview("repo", structure)
        assert "Key entry points" in overview
        assert "main.py" in overview

    def test_empty_structure(self):
        structure = {"total_files": 0, "languages": {}, "key_files": []}
        overview = generate_structure_based_overview("repo", structure)
        assert "repo" in overview
        assert "0 files" in overview


class TestFormatFileStructure:
    def test_basic(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 8},
            "all_files": ["main.py", "utils.py"],
            "directories": {"src": ["main.py"]},
        }
        text = format_file_structure(structure)
        assert "Total Files: 10" in text
        assert "Python: 8 files" in text

    def test_languages_sorted_by_count(self):
        structure = {
            "total_files": 10,
            "languages": {"Python": 3, "JavaScript": 7},
            "all_files": [],
            "directories": {},
        }
        text = format_file_structure(structure)
        js_pos = text.index("JavaScript")
        py_pos = text.index("Python")
        assert js_pos < py_pos  # JavaScript (7) listed before Python (3)

    def test_files_sorted(self):
        structure = {
            "total_files": 3,
            "languages": {},
            "all_files": ["zebra.py", "alpha.py", "beta.py"],
            "directories": {},
        }
        text = format_file_structure(structure)
        alpha_pos = text.index("alpha.py")
        beta_pos = text.index("beta.py")
        zebra_pos = text.index("zebra.py")
        assert alpha_pos < beta_pos < zebra_pos

    def test_top_level_directories(self):
        structure = {
            "total_files": 5,
            "languages": {},
            "all_files": [],
            "directories": {"src": ["main.py", "utils.py"], "tests": ["test.py"]},
        }
        text = format_file_structure(structure)
        assert "src/" in text
        assert "2 files" in text
        assert "tests/" in text

    def test_empty_structure(self):
        structure = {"total_files": 0}
        text = format_file_structure(structure)
        assert "Total Files: 0" in text

    def test_no_directories_section_when_empty(self):
        structure = {
            "total_files": 1,
            "languages": {},
            "all_files": ["readme.txt"],
            "directories": {},
        }
        text = format_file_structure(structure)
        assert "Top-Level Directories" not in text
