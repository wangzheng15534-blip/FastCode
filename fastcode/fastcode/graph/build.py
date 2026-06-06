"""
Graph Builder - Build code relationship graphs
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from typing import Any

import networkx as nx
import tqdm

from fastcode.graph.call_extractor import CallExtractor
from fastcode.graph.contracts import ModuleImportResolver, SymbolNameResolver
from fastcode.ir.element import CodeElement
from fastcode.utils.path_utils import file_path_to_module_path


class CodeGraphBuilder:
    """Build various code relationship graphs"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.graph_config = config.get("graph", {})
        self.logger = logging.getLogger(__name__)
        self.logger.warning(
            "CodeGraphBuilder is in compatibility mode; IRGraphBuilder is the primary backend."
        )

        self.build_call_graph = self.graph_config.get("build_call_graph", True)
        self.build_dependency_graph = self.graph_config.get(
            "build_dependency_graph", True
        )
        self.build_inheritance_graph = self.graph_config.get(
            "build_inheritance_graph", True
        )
        self.max_depth = self.graph_config.get("max_depth", 5)

        # Graphs
        self.call_graph: nx.DiGraph[str] = nx.DiGraph()
        self.dependency_graph: nx.DiGraph[str] = nx.DiGraph()
        self.inheritance_graph: nx.DiGraph[str] = nx.DiGraph()
        self._lazy_graph_payloads: dict[str, dict[str, list[Any]]] | None = None
        self._lazy_successors: dict[str, dict[str, set[str]]] | None = None
        self._lazy_predecessors: dict[str, dict[str, set[str]]] | None = None

        # Maps for quick lookup
        self.element_by_name: dict[str, CodeElement] = {}
        self.element_by_id: dict[str, CodeElement] = {}
        self.imports_by_file: dict[str, list[dict[str, Any]]] = {}

    def build_graphs(
        self,
        elements: list[CodeElement],
        module_resolver: ModuleImportResolver | None = None,
        symbol_resolver: SymbolNameResolver | None = None,
    ) -> None:
        """
        Build all configured graphs

        Args:
            elements: List of code elements
            module_resolver: Optional ModuleResolver for precise dependency resolution
            symbol_resolver: Optional SymbolResolver for precise inheritance resolution
        """
        self.logger.info("Building code relationship graphs")
        self._clear_lazy_graph_state()

        # --- OPTIMIZATION: Pre-compute Lookup Maps ---
        # 1. Scope Lookup for Call Graph (replaces O(N) scan in _get_caller_id_from_scope)
        # Key: (file_path, type, name) -> element_id
        self.scope_lookup: dict[tuple[str, str, str], str] = {}

        # 2. Class Lookup for Inheritance (replaces map building in fallback)
        # Key: class_name -> List[CodeElement]
        self.classes_by_name_lookup: dict[str, list[CodeElement]] = {}

        # Index elements by name
        for elem in elements:
            self.element_by_name[elem.name] = elem
            self.element_by_id[elem.id] = elem

            # Populate Scope Lookup
            if elem.type in ["function", "class", "method"]:
                key = (elem.file_path, elem.type, elem.name)
                self.scope_lookup[key] = elem.id

            # Populate Class Lookup
            if elem.type == "class":
                if elem.name not in self.classes_by_name_lookup:
                    self.classes_by_name_lookup[elem.name] = []
                self.classes_by_name_lookup[elem.name].append(elem)

            # --- FIX: Selective Node Addition (Typing Check) ---
            # Only add nodes to graphs where they semantically belong.
            # This prevents "bloat" (irrelevant nodes) while ensuring
            # valid isolated nodes still exist to prevent NetworkX crashes.

            # 1. Dependency Graph: Only files
            if self.build_dependency_graph and elem.type == "file":
                self.dependency_graph.add_node(elem.id)

            # 2. Inheritance Graph: Only classes
            if self.build_inheritance_graph and elem.type == "class":
                self.inheritance_graph.add_node(elem.id)

            # 3. Call Graph: Functions, Methods, Classes
            # (Files are excluded as nodes unless they are added later as explicit callers)
            if self.build_call_graph and elem.type in ["function", "method", "class"]:
                self.call_graph.add_node(elem.id)

            # Track imports
            if elem.type == "file":
                imports = elem.metadata.get("imports", [])
                if imports:
                    self.imports_by_file[elem.file_path] = imports

            # --- ADD LOGGING HERE ---
            ## <debug> with verify_shadowing.py
            if elem.name == "action":
                preprocessed_metadata = {
                    k: v for k, v in elem.metadata.items() if k != "embedding"
                }
                preprocessed_metadata["embedding"] = (
                    " array([...])" if "embedding" in elem.metadata else "None"
                )
                self.logger.debug(
                    f"[DEBUG GRAPH] Found 'action' element. ID: {elem.id}, Type: {elem.type}"
                )
                self.logger.debug(f"              Metadata: {preprocessed_metadata}")

        # Build graphs
        if self.build_dependency_graph:
            self._build_dependency_graph(elements, module_resolver)

        if self.build_inheritance_graph:
            self._build_inheritance_graph(elements, symbol_resolver)

        if self.build_call_graph:
            self._build_call_graph(elements, symbol_resolver)

        self.logger.info(
            f"Built graphs: "
            f"dependency ({self.dependency_graph.number_of_nodes()} nodes), "
            f"inheritance ({self.inheritance_graph.number_of_nodes()} nodes), "
            f"call ({self.call_graph.number_of_nodes()} nodes)"
        )

    def _clear_lazy_graph_state(self) -> None:
        self._lazy_graph_payloads = None
        self._lazy_successors = None
        self._lazy_predecessors = None

    @staticmethod
    def _empty_graph_payloads() -> dict[str, dict[str, list[Any]]]:
        return {
            "call": {"nodes": [], "edges": []},
            "dependency": {"nodes": [], "edges": []},
            "inheritance": {"nodes": [], "edges": []},
        }

    @staticmethod
    def _build_lazy_adjacency(
        graph_payloads: dict[str, dict[str, list[Any]]],
    ) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, set[str]]]]:
        successors: dict[str, dict[str, set[str]]] = {}
        predecessors: dict[str, dict[str, set[str]]] = {}
        for graph_name, payload in graph_payloads.items():
            succ: dict[str, set[str]] = {}
            pred: dict[str, set[str]] = {}
            for raw_node in payload.get("nodes", []):
                node_id = str(raw_node)
                succ.setdefault(node_id, set())
                pred.setdefault(node_id, set())
            for edge in payload.get("edges", []):
                if not isinstance(edge, dict):
                    continue
                source = edge.get("source")
                target = edge.get("target")
                if source is None or target is None:
                    continue
                source_id = str(source)
                target_id = str(target)
                succ.setdefault(source_id, set()).add(target_id)
                succ.setdefault(target_id, set())
                pred.setdefault(target_id, set()).add(source_id)
                pred.setdefault(source_id, set())
            successors[graph_name] = succ
            predecessors[graph_name] = pred
        return successors, predecessors

    @staticmethod
    def _materialize_graph_payload(
        payload: dict[str, list[Any]],
    ) -> nx.DiGraph[str]:
        graph: nx.DiGraph[str] = nx.DiGraph()
        for raw_node in payload.get("nodes", []):
            graph.add_node(str(raw_node))
        for edge in payload.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source is None or target is None:
                continue
            attrs = edge.get("attrs")
            graph.add_edge(
                str(source),
                str(target),
                **(dict(attrs) if isinstance(attrs, dict) else {}),
            )
        return graph

    @staticmethod
    def graph_payload_from_object(graph: Any) -> dict[str, list[Any]]:
        """Convert a graph-like object into the plain graph payload contract."""
        payload: dict[str, list[Any]] = {"nodes": [], "edges": []}
        if graph is None:
            return payload

        nodes_attr = getattr(graph, "nodes", ())
        raw_nodes = nodes_attr() if callable(nodes_attr) else nodes_attr
        if isinstance(raw_nodes, Iterable):
            for raw_node in raw_nodes:
                payload["nodes"].append(str(raw_node))

        edges_attr = getattr(graph, "edges", ())
        raw_edges = edges_attr(data=True) if callable(edges_attr) else edges_attr
        if not isinstance(raw_edges, Iterable):
            return payload
        for edge in raw_edges:
            if isinstance(edge, dict):
                source = edge.get("source")
                target = edge.get("target")
                attrs = edge.get("attrs")
            elif isinstance(edge, tuple) and len(edge) >= 2:
                source = edge[0]
                target = edge[1]
                attrs = edge[2] if len(edge) > 2 else {}
            else:
                continue
            if source is None or target is None:
                continue
            payload["edges"].append(
                {
                    "source": str(source),
                    "target": str(target),
                    "attrs": dict(attrs) if isinstance(attrs, dict) else {},
                }
            )
        return payload

    def _ensure_materialized_graphs(self) -> None:
        if self._lazy_graph_payloads is None:
            return
        self.call_graph = self._materialize_graph_payload(
            self._lazy_graph_payloads["call"]
        )
        self.dependency_graph = self._materialize_graph_payload(
            self._lazy_graph_payloads["dependency"]
        )
        self.inheritance_graph = self._materialize_graph_payload(
            self._lazy_graph_payloads["inheritance"]
        )
        self._clear_lazy_graph_state()

    def _lazy_neighbors(
        self,
        graph_name: str,
        element_id: str,
        *,
        reverse: bool = False,
    ) -> list[str]:
        views = self._lazy_predecessors if reverse else self._lazy_successors
        if views is None:
            return []
        return list(views.get(graph_name, {}).get(element_id, ()))

    def _lazy_reachable(
        self,
        graph_name: str,
        element_id: str,
        *,
        max_hops: int,
        reverse: bool = False,
    ) -> set[str]:
        views = self._lazy_predecessors if reverse else self._lazy_successors
        if views is None:
            return set()
        adjacency = views.get(graph_name, {})
        if element_id not in adjacency:
            return set()
        visited = {element_id}
        frontier = deque([(element_id, 0)])
        while frontier:
            node_id, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for neighbor in adjacency.get(node_id, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
        return visited

    def _build_dependency_graph(
        self,
        elements: list[CodeElement],
        module_resolver: ModuleImportResolver | None = None,
    ) -> None:
        """
        Build file dependency graph based on imports using precise module resolution.

        Args:
            elements: List of code elements
            module_resolver: ModuleResolver for precise dependency resolution
        """
        repo_root = self.config.get("repo_root", "")

        for elem in elements:
            if elem.type == "file":
                imports = elem.metadata.get("imports", [])

                # Get current file's module path
                current_module_path = self._get_module_path_from_file_id(
                    elem.id, elements, repo_root
                )
                if not current_module_path:
                    continue

                # --- NEW: Check if this is a package file ---
                is_package = elem.file_path.endswith("__init__.py")

                for imp in imports:
                    module: str = imp.get("module", "")
                    names: list[str] = imp.get("names", [])
                    level: int = imp.get("level", 0)

                    # Determine which modules to resolve
                    modules_to_resolve: list[str] = []

                    if module:
                        # Case 1: Standard import - from module import names
                        modules_to_resolve.append(module)
                    elif level > 0 and names:
                        # Case 2: Relative import - from . import X, Y
                        # Handle "from . import X" where module is empty but names contains imports
                        modules_to_resolve.extend(names)

                    # Skip if nothing to resolve
                    if not modules_to_resolve:
                        continue

                    # Use ModuleResolver for precise resolution if available
                    if module_resolver:
                        for target_module in modules_to_resolve:
                            # --- UPDATE THE CALL HERE ---
                            target_file_id = module_resolver.resolve_import(
                                current_module_path=current_module_path,
                                import_name=target_module,
                                level=level,
                                is_package=is_package,  # Pass the flag
                            )

                            # Add edge if resolution succeeded and it's not a third-party library
                            if target_file_id:
                                if target_file_id == elem.id:
                                    continue
                                # [FIX] Ensure target file belongs to the same repo (Multi-Repo Collision Fix)
                                target_elem = self.element_by_id.get(target_file_id)
                                if (
                                    target_elem
                                    and target_elem.repo_name != elem.repo_name
                                ):
                                    self.logger.debug(
                                        f"Skipping cross-repo dependency: {elem.id} -> {target_file_id} "
                                        f"(Repos: {elem.repo_name} vs {target_elem.repo_name})"
                                    )
                                    continue

                                self.dependency_graph.add_edge(
                                    elem.id,
                                    target_file_id,
                                    type="imports",  # Use "imports" for consistency
                                    module=target_module,  # Use the actual resolved module name
                                    level=level,
                                    resolution_method="AST ModuleResolver",
                                )
                    else:
                        # Fallback to original logic (for backward compatibility)
                        # NOTE: This is the flawed string matching approach
                        self.logger.warning(
                            f"ModuleResolver not provided for {elem.id}. "
                            f"Using fallback string matching - may produce false positives."
                        )
                        for target_module in modules_to_resolve:
                            for other_elem in elements:
                                # [FIX] Ensure we only link files within the same repository to avoid multi-repo collisions
                                if other_elem.repo_name != elem.repo_name:
                                    continue

                                # [FIX] Prevent self-imports (e.g. file linking to itself because import name is substring of filename)
                                if elem.id == other_elem.id:
                                    continue

                                if (
                                    other_elem.type == "file"
                                    and target_module in other_elem.relative_path
                                ):
                                    self.dependency_graph.add_edge(
                                        elem.id,
                                        other_elem.id,
                                        type="imports",
                                        module=target_module,
                                        level=level,
                                        resolution_method="fallback_string_matching",
                                    )

    def _build_inheritance_graph(
        self,
        elements: list[CodeElement],
        symbol_resolver: SymbolNameResolver | None = None,
    ) -> None:
        """
        Build class inheritance graph using precise symbol resolution

        Args:
            elements: List of code elements
            symbol_resolver: Optional SymbolResolver for cross-file inheritance resolution
        """
        # Build inheritance relationships
        for elem in elements:
            if elem.type == "class":
                bases = elem.metadata.get("bases", [])

                for base_name in bases:
                    # Use SymbolResolver for precise resolution if available
                    if symbol_resolver:
                        # Get context needed for SymbolResolver
                        file_imports = self.imports_by_file.get(elem.file_path, [])
                        current_file_id = self._get_file_id_for_class_element(
                            elem, elements
                        )

                        if current_file_id:
                            # Resolve parent class using SymbolResolver
                            parent_class_id = symbol_resolver.resolve_symbol(
                                symbol_name=base_name,
                                current_file_id=current_file_id,
                                imports=file_imports,
                            )

                            # Add edge if resolution succeeded
                            if parent_class_id:
                                self.inheritance_graph.add_edge(
                                    elem.id,
                                    parent_class_id,
                                    type="inherits",
                                    base_name=base_name,  # Store the original base name for debugging
                                )
                                self.logger.debug(
                                    f"Added inheritance edge: {elem.id} -> {parent_class_id} "
                                    f"(resolved from '{base_name}')"
                                )
                            else:
                                self.logger.debug(
                                    f"Could not resolve parent class '{base_name}' for {elem.id}"
                                )
                        else:
                            self.logger.warning(
                                f"Could not determine file_id for class {elem.id}, "
                                f"falling back to local resolution"
                            )
                            # Fall back to local resolution
                            self._fallback_to_local_inheritance_resolution(
                                elem, base_name, elements
                            )
                    else:
                        # Fallback to original logic (for backward compatibility)
                        self.logger.warning(
                            f"SymbolResolver not provided for {elem.id}. "
                            f"Using fallback local name matching - may miss cross-file inheritance."
                        )
                        self._fallback_to_local_inheritance_resolution(
                            elem, base_name, elements
                        )

    def _get_file_id_for_class_element(
        self, class_elem: CodeElement, elements: list[CodeElement]
    ) -> str | None:
        """
        Get file ID for a class element by finding the corresponding file element

        Args:
            class_elem: Class CodeElement
            elements: List of all code elements

        Returns:
            File ID if found, None otherwise
        """
        # Find the file element that contains this class
        for elem in elements:
            if elem.type == "file" and elem.file_path == class_elem.file_path:
                return elem.id

        # Try to construct file ID from file path if we can't find the element
        # This is a fallback approach
        if class_elem.file_path:
            # Generate file ID similar to how indexer would do it
            import os

            filename = os.path.basename(class_elem.file_path)
            name, _ = os.path.splitext(filename)
            return f"file_{name}"

        return None

    def _fallback_to_local_inheritance_resolution(
        self, elem: CodeElement, base_name: str, elements: list[CodeElement]
    ) -> None:
        """
        Fallback method for local-only inheritance resolution.
        [FIXED] Now repo-aware to prevent multi-repo collisions.
        OPTIMIZED: Uses pre-computed class lookup instead of rebuilding map

        Args:
            elem: Class CodeElement
            base_name: Name of the base class to resolve
            elements: List of all code elements
        """
        # OPTIMIZED: Use pre-computed lookup instead of building map every time
        candidates = self.classes_by_name_lookup.get(base_name, [])

        if not candidates:
            return

        # Find base class using name matching, prioritizing SAME REPOSITORY
        best_match = None

        # Try to find a match in the same repository
        for candidate in candidates:
            if candidate.repo_name == elem.repo_name:
                best_match = candidate
                break

        # Only link if it's in the same repo to prevent contamination
        if best_match:
            self.inheritance_graph.add_edge(
                elem.id,
                best_match.id,
                type="inherits",
                base_name=base_name,
                resolution_method="fallback_local_matching_repo_aware",
            )
            self.logger.debug(
                f"Added inheritance edge via fallback: {elem.id} -> {best_match.id} "
                f"(matched '{base_name}' in same repo '{elem.repo_name}')"
            )
        else:
            self.logger.debug(
                f"Ignored potential base class '{base_name}' for {elem.id} "
                f"because it belongs to a different repository."
            )

    def _build_call_graph(
        self,
        elements: list[CodeElement],
        symbol_resolver: SymbolNameResolver | None = None,
    ) -> None:
        """
        Build function call graph using CallExtractor and SymbolResolver (Task 4.4)

        Args:
            elements: List of code elements
            symbol_resolver: SymbolResolver for resolving callee definitions
        """
        if not symbol_resolver:
            self.logger.warning(
                "SymbolResolver not provided for call graph building. "
                "Skipping call graph construction."
            )
            return

        # Initialize CallExtractor
        call_extractor = CallExtractor()

        # Statistics
        total_calls = 0
        linked_calls = 0

        pbar_elements = tqdm.tqdm(elements, desc="Building call graph")
        for elem in pbar_elements:
            if elem.type == "file":
                # Extract calls from this file (Task 4.3)
                calls = call_extractor.extract_calls(elem.code, elem.file_path)
                total_calls += len(calls)

                # Extract instance variable types for enhanced resolution (Phase 1)
                file_instance_types = call_extractor.extract_instance_types(elem.code)

                # --- ADD DEBUG LOGGING HERE ---
                if file_instance_types:
                    self.logger.debug(
                        f"[EXTRACT] Instance types found in {elem.file_path}:"
                    )
                    for scope, vars_map in file_instance_types.items():
                        self.logger.debug(
                            f"    Scope '{scope}': {list(vars_map.keys())}"
                        )
                        # Detailed log for __init__ methods to debug Bug #8
                        if "init" in scope or "class" in scope:
                            self.logger.debug(f"        -> Vars: {vars_map}")
                # -----------------------------

                # Get imports context for this file
                file_imports = self.imports_by_file.get(elem.file_path, [])

                for call in calls:
                    # Determine caller ID (using scope_id from Task 4.3)
                    caller_id = self._get_caller_id_from_scope(call, elem, elements)

                    # Retrieve the actual caller element to get its class context
                    caller_elem = (
                        self.element_by_id.get(caller_id) if caller_id else None
                    )

                    # Determine callee ID using SymbolResolver with instance type inference (Phase 2)
                    callee_ids = self._resolve_callee_with_symbol_resolver(
                        call,
                        elem.id,
                        file_imports,
                        symbol_resolver,
                        file_instance_types,
                        caller_elem=caller_elem,
                    )

                    # Add edge(s) for each resolved callee (now supports one-to-many)
                    if caller_id and callee_ids:
                        for callee_id in callee_ids:
                            self.call_graph.add_edge(
                                caller_id,
                                callee_id,
                                type="calls",
                                call_name=call["call_name"],
                                call_type=call.get("call_type", "unknown"),
                                file_path=call["file_path"],
                                node_text=call.get(
                                    "node_text", ""
                                ),  # <--- ADD THIS LINE
                            )
                            linked_calls += 1

                            self.logger.debug(
                                f"Added call edge: {caller_id} -> {callee_id} "
                                f"('{call['call_name']}' in {call.get('call_type', 'unknown')} call)"
                            )
                    else:
                        self.logger.debug(
                            f"Could not link call: '{call['call_name']}' "
                            f"(caller: {caller_id}, callee: {callee_ids})"
                        )

        self.logger.info(
            f"Call graph built: {linked_calls}/{total_calls} calls successfully linked "
            f"({linked_calls / total_calls * 100 if total_calls > 0 else 0:.1f}% success rate)"
        )

    def get_related_elements(self, element_id: str, max_hops: int = 2) -> set[str]:
        """
        Get related elements within max_hops distance

        Args:
            element_id: Starting element ID
            max_hops: Maximum distance to traverse

        Returns:
            Set of related element IDs
        """
        related: set[str] = set()

        if self._lazy_graph_payloads is not None:
            for graph_name in ("dependency", "inheritance", "call"):
                related.update(
                    self._lazy_reachable(
                        graph_name,
                        element_id,
                        max_hops=max_hops,
                    )
                )
                related.update(
                    self._lazy_reachable(
                        graph_name,
                        element_id,
                        max_hops=max_hops,
                        reverse=True,
                    )
                )
            return related

        # Check all graphs
        for graph in [self.dependency_graph, self.inheritance_graph, self.call_graph]:
            if element_id in graph:
                # Get predecessors and successors within max_hops
                for node in nx.single_source_shortest_path_length(
                    graph, element_id, cutoff=max_hops
                ):
                    related.add(node)

                # Also check reverse direction
                for node in nx.single_source_shortest_path_length(
                    graph.reverse(), element_id, cutoff=max_hops
                ):
                    related.add(node)

        return related

    def _get_module_path_from_file_id(
        self, file_id: str, elements: list[CodeElement], repo_root: str
    ) -> str | None:
        """
        Get module path from file ID by looking up the corresponding CodeElement.

        Args:
            file_id: File element ID
            elements: List of all code elements for lookup
            repo_root: Repository root path for module path conversion

        Returns:
            Module path (e.g., "app.services.auth") or None if not found
        """
        # Find the file element by ID
        for elem in elements:
            if elem.id == file_id and elem.type == "file":
                # Use the element's file_path to generate module path
                return file_path_to_module_path(elem.file_path, repo_root)

        return None

    def get_dependencies(self, element_id: str) -> list[str]:
        """Get direct dependencies of an element"""
        if self._lazy_graph_payloads is not None:
            return self._lazy_neighbors("dependency", element_id)
        if element_id in self.dependency_graph:
            return list(self.dependency_graph.successors(element_id))
        return []

    def get_dependents(self, element_id: str) -> list[str]:
        """Get elements that depend on this element"""
        if self._lazy_graph_payloads is not None:
            return self._lazy_neighbors("dependency", element_id, reverse=True)
        if element_id in self.dependency_graph:
            return list(self.dependency_graph.predecessors(element_id))
        return []

    def get_subclasses(self, element_id: str) -> list[str]:
        """Get subclasses of a class"""
        if self._lazy_graph_payloads is not None:
            return self._lazy_neighbors("inheritance", element_id, reverse=True)
        if element_id in self.inheritance_graph:
            return list(self.inheritance_graph.predecessors(element_id))
        return []

    def get_superclasses(self, element_id: str) -> list[str]:
        """Get superclasses of a class"""
        if self._lazy_graph_payloads is not None:
            return self._lazy_neighbors("inheritance", element_id)
        if element_id in self.inheritance_graph:
            return list(self.inheritance_graph.successors(element_id))
        return []

    def get_callers(self, element_id: str) -> list[str]:
        """Get functions that call this function"""
        if self._lazy_graph_payloads is not None:
            return self._lazy_neighbors("call", element_id, reverse=True)
        if element_id in self.call_graph:
            return list(self.call_graph.predecessors(element_id))
        return []

    def get_callees(self, element_id: str) -> list[str]:
        """Get functions called by this function"""
        if self._lazy_graph_payloads is not None:
            return self._lazy_neighbors("call", element_id)
        if element_id in self.call_graph:
            return list(self.call_graph.successors(element_id))
        return []

    def find_path(
        self, source_id: str, target_id: str, graph_type: str = "dependency"
    ) -> list[str] | None:
        """
        Find path between two elements

        Args:
            source_id: Source element ID
            target_id: Target element ID
            graph_type: Type of graph to use (dependency, inheritance, call)

        Returns:
            List of element IDs forming the path, or None if no path
        """
        self._ensure_materialized_graphs()
        graph_map: dict[str, nx.DiGraph[str]] = {
            "dependency": self.dependency_graph,
            "inheritance": self.inheritance_graph,
            "call": self.call_graph,
        }

        graph = graph_map.get(graph_type)
        if graph is None:
            return None

        try:
            _sp = nx.shortest_path
            result: Any = _sp(graph, source_id, target_id)
            return list(result)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the graphs"""
        self._ensure_materialized_graphs()
        stats: dict[str, dict[str, Any]] = {}

        for name, graph in [
            ("dependency", self.dependency_graph),
            ("inheritance", self.inheritance_graph),
            ("call", self.call_graph),
        ]:
            stats[name] = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(graph),
            }

            if graph.number_of_nodes() > 0:
                try:
                    stats[name]["avg_degree"] = (
                        sum(d for _, d in graph.degree()) / graph.number_of_nodes()
                    )
                except Exception:
                    stats[name]["avg_degree"] = 0

        return stats

    def materialize_graphs(self) -> None:
        """Materialize any lazy graph payloads into NetworkX graph objects."""
        self._ensure_materialized_graphs()

    def graph_payloads(self) -> dict[str, dict[str, list[Any]]]:
        """Return plain graph payloads for persistence boundaries."""
        self._ensure_materialized_graphs()
        return {
            "call": self.graph_payload_from_object(self.call_graph),
            "dependency": self.graph_payload_from_object(self.dependency_graph),
            "inheritance": self.graph_payload_from_object(self.inheritance_graph),
        }

    def persisted_elements(self) -> list[CodeElement]:
        """Return the unique elements currently attached to this graph state."""
        if self.element_by_id:
            return list(self.element_by_id.values())
        seen_ids: set[str] = set()
        elements: list[CodeElement] = []
        for elem in self.element_by_name.values():
            if elem.id in seen_ids:
                continue
            seen_ids.add(elem.id)
            elements.append(elem)
        return elements

    def restore_graph_state(
        self,
        *,
        imports_by_file: dict[str, list[dict[str, Any]]],
        elements: list[CodeElement],
        graph_payloads: dict[str, dict[str, list[Any]]] | None = None,
        call_graph: nx.DiGraph[str] | None = None,
        dependency_graph: nx.DiGraph[str] | None = None,
        inheritance_graph: nx.DiGraph[str] | None = None,
    ) -> None:
        """Restore in-memory graph state from explicit graph-domain values."""
        if graph_payloads is not None:
            self.call_graph = nx.DiGraph()
            self.dependency_graph = nx.DiGraph()
            self.inheritance_graph = nx.DiGraph()
            self._lazy_graph_payloads = graph_payloads
            self._lazy_successors, self._lazy_predecessors = self._build_lazy_adjacency(
                self._lazy_graph_payloads
            )
        else:
            self.call_graph = call_graph or nx.DiGraph()
            self.dependency_graph = dependency_graph or nx.DiGraph()
            self.inheritance_graph = inheritance_graph or nx.DiGraph()
            self._clear_lazy_graph_state()

        self.imports_by_file = imports_by_file
        self.element_by_name = {}
        self.element_by_id = {}
        for elem in elements:
            self.element_by_id[elem.id] = elem
            self.element_by_name[elem.name] = elem

    def merge_graph_state(
        self,
        *,
        imports_by_file: dict[str, list[dict[str, Any]]],
        elements: list[CodeElement],
        graph_payloads: dict[str, dict[str, list[Any]]] | None = None,
        call_graph: nx.DiGraph[str] | None = None,
        dependency_graph: nx.DiGraph[str] | None = None,
        inheritance_graph: nx.DiGraph[str] | None = None,
    ) -> None:
        """Merge explicit graph-domain values into the current graph state."""
        self._ensure_materialized_graphs()
        if graph_payloads is not None:
            other_call_graph = self._materialize_graph_payload(graph_payloads["call"])
            other_dependency_graph = self._materialize_graph_payload(
                graph_payloads["dependency"]
            )
            other_inheritance_graph = self._materialize_graph_payload(
                graph_payloads["inheritance"]
            )
        else:
            other_call_graph = call_graph or nx.DiGraph()
            other_dependency_graph = dependency_graph or nx.DiGraph()
            other_inheritance_graph = inheritance_graph or nx.DiGraph()

        self.call_graph = nx.compose(self.call_graph, other_call_graph)
        self.dependency_graph = nx.compose(
            self.dependency_graph, other_dependency_graph
        )
        self.inheritance_graph = nx.compose(
            self.inheritance_graph, other_inheritance_graph
        )

        for elem in elements:
            if elem.id in self.element_by_id:
                continue
            self.element_by_name[elem.name] = elem
            self.element_by_id[elem.id] = elem

        self.imports_by_file.update(imports_by_file)

    def _get_caller_id_from_scope(
        self, call: dict[str, Any], file_elem: CodeElement, elements: list[CodeElement]
    ) -> str | None:
        """
        Get caller ID from call scope information (Task 4.4)
        OPTIMIZED: O(1) lookup instead of O(N) scan

        Args:
            call: Call information dictionary from CallExtractor
            file_elem: File element containing the call
            elements: List of all code elements for lookup

        Returns:
            Caller element ID if found, None otherwise
        """
        scope_id = call.get("scope_id")

        # --- ADD DEBUG LOGGING HERE ---
        # Only log if it's a method call we are interested in (reduce noise)
        if call.get("base_object") in ["self", "cls"] or (
            scope_id and "class" in scope_id
        ):
            self.logger.debug(
                f"[SCOPE] Processing call '{call.get('call_name')}' inside scope: '{scope_id}'"
            )
        # -----------------------------

        if scope_id is None:
            # Module-level call, caller is the file itself
            return file_elem.id

        # Parse scope_id format: "type::name" (e.g., "function::process_data")
        scope_parts = scope_id.split("::", 1)
        if len(scope_parts) != 2:
            self.logger.warning(f"Invalid scope_id format: {scope_id}")
            return file_elem.id

        scope_type, scope_name = scope_parts

        # OPTIMIZED: O(1) lookup using pre-computed dictionary
        key = (file_elem.file_path, scope_type, scope_name)
        caller_id = self.scope_lookup.get(key)

        if not caller_id:
            self.logger.debug(
                f"Could not find {scope_type} element '{scope_name}' in {file_elem.file_path}"
            )

        return caller_id

    def _resolve_callee_with_symbol_resolver(
        self,
        call: dict[str, Any],
        current_file_id: str,
        file_imports: list[dict[str, Any]],
        symbol_resolver: SymbolNameResolver,
        file_instance_types: dict[str, dict[str, list[str]]] | None = None,
        caller_elem: CodeElement | None = None,
    ) -> list[str]:
        """
        Resolve callee definition using SymbolResolver with instance variable type inference.

        Args:
            call: Call information dictionary from CallExtractor
            current_file_id: ID of the current file
            file_imports: List of import dictionaries for the current file
            symbol_resolver: SymbolResolver instance
            file_instance_types: Optional dict mapping instance variables to scoped potential class types

        Returns:
            List of resolved callee element IDs (supports one-to-many relationships)
        """
        call_name = call["call_name"]
        call_type = call.get("call_type", "simple")
        base_object = call.get("base_object")

        # Case 1: Simple function call: func()
        if call_type == "simple":
            resolved_id = symbol_resolver.resolve_symbol(
                call_name, current_file_id, file_imports
            )
            return [resolved_id] if resolved_id else []

        # Case 2: Module function call OR Instance method call
        if call_type == "attribute" and base_object:
            # --- FIX: Check if base_object is a local variable first! ---
            # If it is, we should treat it as an instance method call, NOT a module call.
            # This fixes verify_fix_2.py where 'service' var shadowed 'service' module.
            call_scope = call.get("scope_id") or "global"
            is_local_var = False

            if file_instance_types and (
                base_object in file_instance_types.get(call_scope, {})
                or base_object in file_instance_types.get("global", {})
                or base_object in file_instance_types.get("function::__init__", {})
            ):
                # Check current scope
                is_local_var = True

            if not is_local_var:
                # Only check module imports if it's NOT a known local variable
                for import_info in file_imports:
                    if import_info.get("module") == base_object:
                        # This is a module.function() call
                        # Resolve the full call name "module.function" using SymbolResolver
                        full_call_name = f"{base_object}.{call_name}"
                        resolved_id = symbol_resolver.resolve_symbol(
                            full_call_name, current_file_id, file_imports
                        )
                        return [resolved_id] if resolved_id else []

            # If not a module call (or it was shadowed), check if it's self/cls call
            if base_object in ["self", "cls"]:
                # Case 3: Method call: obj.method() or self.method()

                # 1. Try resolving as "ClassName.method" if we are inside a class
                if caller_elem and caller_elem.type == "function":
                    class_name = caller_elem.metadata.get("class_name")
                    if class_name:
                        full_method_name = f"{class_name}.{call_name}"
                        resolved_id = symbol_resolver.resolve_symbol(
                            full_method_name, current_file_id, file_imports
                        )
                        if resolved_id:
                            return [resolved_id]

                # 2. Fallback: Try to resolve locally (e.g. if it's actually a global function called on module alias self?)
                # or if the indexing didn't capture the class name correctly.
                resolved_id = symbol_resolver.resolve_symbol(
                    call_name, current_file_id, file_imports
                )
                return [resolved_id] if resolved_id else []
            # --- ADD DEBUG LOGGING HERE ---
            self.logger.debug(
                f"[ROUTING] Routing '{base_object}.{call_name}' to Instance Method Resolution"
            )
            self.logger.debug(
                f"          Context: File={current_file_id}, Scope={call_scope}"
            )
            # -----------------------------

            # Case 4: Instance method call
            return self._resolve_instance_method_call(
                base_object,
                call_name,
                current_file_id,
                file_imports,
                symbol_resolver,
                file_instance_types or {},
                call_scope,
            )

        # Default case: try simple resolution
        resolved_id = symbol_resolver.resolve_symbol(
            call_name, current_file_id, file_imports
        )
        return [resolved_id] if resolved_id else []

    def _resolve_instance_method_call(
        self,
        base_object: str,
        call_name: str,
        current_file_id: str,
        file_imports: list[dict[str, Any]],
        symbol_resolver: SymbolNameResolver,
        file_instance_types: dict[str, dict[str, list[str]]],
        scope_id: str,
    ) -> list[str]:
        """
        Resolve instance method calls using type inference (Phase 2.3).

        Args:
            base_object: The instance variable name (e.g., 'self.loader' from 'self.loader.load()')
            call_name: The method being called (e.g., 'load')
            current_file_id: ID of the current file
            file_imports: List of import dictionaries for the current file
            symbol_resolver: SymbolResolver instance
            file_instance_types: Dictionary mapping instance variables to potential class types

        Returns:
            List of resolved callee element IDs (one for each potential class type)
        """
        # --- DEBUG LOGGING START ---
        self.logger.debug(
            f"[RESOLVE] Attempting to resolve '{base_object}.{call_name}' from Scope: '{scope_id}'"
        )
        # --- DEBUG LOGGING END ---

        # 1. Try to find variable in the current local scope
        local_types = file_instance_types.get(scope_id, {})
        candidate_classes = local_types.get(base_object)

        # --- DEBUG LOGGING START ---
        if candidate_classes:
            self.logger.debug(
                f"    [+] Found type in LOCAL scope '{scope_id}': {candidate_classes}"
            )
        else:
            self.logger.debug(f"    [-] Not found in LOCAL scope '{scope_id}'")
        # --- DEBUG LOGGING END ---

        # 2. If not found locally, check '__init__' scope (Fix for Bug #8)
        # Instance variables (self.x) are typically defined in __init__
        if not candidate_classes:
            candidate_classes = file_instance_types.get("function::__init__", {}).get(
                base_object
            )

            if candidate_classes:
                self.logger.debug(
                    f"    [+] Found type in __init__ scope: {candidate_classes}"
                )

        # 3. If not found, check 'global' scope
        if not candidate_classes:
            candidate_classes = file_instance_types.get("global", {}).get(base_object)

            # --- DEBUG LOGGING START ---
            if candidate_classes:
                self.logger.debug(
                    f"    [+] Found type in GLOBAL scope: {candidate_classes}"
                )
            else:
                self.logger.debug("    [-] Not found in GLOBAL scope")

                # Debugging aid
                found_in_other_scopes: list[str] = []
                for s_id, vars_map in file_instance_types.items():
                    if base_object in vars_map:
                        found_in_other_scopes.append(
                            f"{s_id} -> {vars_map[base_object]}"
                        )

                if found_in_other_scopes:
                    self.logger.debug(
                        f"    [!] Variable '{base_object}' exists in other scopes: {found_in_other_scopes}"
                    )
            # --- DEBUG LOGGING END ---

        if not candidate_classes:
            return []

        resolved_ids: list[str] = []

        # For each candidate class, try to resolve the method
        for class_name in candidate_classes:
            # Step 1: Resolve the class definition first
            class_id = symbol_resolver.resolve_symbol(
                class_name, current_file_id, file_imports
            )

            if not class_id:
                self.logger.debug(
                    f"    [-] Could not resolve class definition for '{class_name}'"
                )
                continue

            # Step 2: Try to resolve the method within that class
            # Try standard "ClassName.method" resolution first
            full_method_name = f"{class_name}.{call_name}"
            method_id = symbol_resolver.resolve_symbol(
                full_method_name, current_file_id, file_imports
            )

            if method_id:
                resolved_ids.append(method_id)
                self.logger.debug(
                    f"    [+] Resolved method '{full_method_name}' to {method_id}"
                )
            else:
                # Fallback: Try resolving just the method name
                method_id = symbol_resolver.resolve_symbol(
                    call_name, current_file_id, file_imports
                )

                if method_id:
                    resolved_ids.append(method_id)
                else:
                    # Final Fallback: Look inside the class file directly
                    # If we resolved the class (class_id), look for the method INSIDE that class's file.
                    class_elem = self.element_by_id.get(class_id)
                    found_in_class_file = False

                    if class_elem:
                        # Look for function with matching name and class_name in the same file
                        for elem in self.element_by_id.values():
                            if (
                                elem.type == "function"
                                and elem.name == call_name
                                and elem.file_path == class_elem.file_path
                                and elem.metadata.get("class_name") == class_elem.name
                            ):
                                resolved_ids.append(elem.id)
                                self.logger.debug(
                                    f"    [+] Resolved method '{call_name}' in class '{class_name}' via file lookup"
                                )
                                found_in_class_file = True
                                break

                    if not found_in_class_file:
                        # Final Fallback: Link to the class itself (Recall over Precision)
                        resolved_ids.append(class_id)
                        self.logger.debug(
                            f"    [~] Could not resolve method '{call_name}' in class '{class_name}', "
                            f"linking to class {class_id} instead"
                        )

        # Deduplicate resolved IDs while preserving order
        seen: set[str] = set()
        unique_resolved_ids: list[str] = []
        for resolved_id in resolved_ids:
            if resolved_id not in seen:
                seen.add(resolved_id)
                unique_resolved_ids.append(resolved_id)

        return unique_resolved_ids
