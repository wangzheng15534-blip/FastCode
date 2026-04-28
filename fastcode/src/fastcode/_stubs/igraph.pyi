"""Minimal type stubs for igraph."""

from typing import Any

class Graph:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def List(**kwargs: Any) -> Graph: ...
