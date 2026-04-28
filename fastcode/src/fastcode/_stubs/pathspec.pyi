"""Minimal type stubs for pathspec."""

class PathSpec:
    @staticmethod
    def from_lines(pattern_class: type, lines: list[str]) -> PathSpec: ...
    def match_file(self, path: str) -> bool: ...

class GitWildMatchPattern:
    @staticmethod
    def pattern_to_regex(pattern: str) -> tuple[str, bool]: ...
