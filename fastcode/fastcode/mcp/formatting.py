"""MCP text formatting — pure functions that turn domain dicts into MCP strings.

This is the entry_frame-local mapper for MCP.  No imports from use_flow,
meaning_core, or assembly_root internals.
"""

from __future__ import annotations

from typing import Any


def format_code_qa_response(
    answer: str, sources: list[dict[str, Any]], session_id: str
) -> str:
    parts = [answer]
    if sources:
        parts.append("\n\n---\nSources:")
        for s in sources:
            file_path = s.get("file", s.get("relative_path", ""))
            repo = s.get("repo", s.get("repository", ""))
            name = s.get("name", "")
            start = s.get("start_line", "")
            end = s.get("end_line", "")
            if (not start or not end) and s.get("lines"):
                lines = str(s.get("lines", ""))
                if "-" in lines:
                    parsed_start, parsed_end = lines.split("-", 1)
                    start = start or parsed_start
                    end = end or parsed_end
            loc = f"L{start}-L{end}" if start and end else ""
            if repo:
                parts.append(f"  - {repo}/{file_path}:{loc} ({name})")
            else:
                parts.append(f"  - {file_path}:{loc} ({name})")
    parts.append(f"[session_id: {session_id}]")
    return "\n".join(parts)


def format_session_list(sessions: list[dict[str, Any]]) -> str:
    if not sessions:
        return "No sessions found."
    lines = ["Sessions:"]
    for s in sessions:
        sid = s.get("session_id", "?")
        title = s.get("title", "Untitled")
        turns = s.get("total_turns", 0)
        mode = "multi-turn" if s.get("multi_turn", False) else "single-turn"
        lines.append(f'  - {sid}: "{title}" ({turns} turns, {mode})')
    return "\n".join(lines)


def format_session_history(session_id: str, history: list[dict[str, Any]]) -> str:
    if not history:
        return f"No history found for session '{session_id}'."
    lines = [f"Session {session_id} history:"]
    for turn in history:
        turn_num = turn.get("turn_number", "?")
        query = turn.get("query", "")
        answer = turn.get("answer", "")
        if len(answer) > 500:
            answer = answer[:500] + " …"
        lines.append(f"\n--- Turn {turn_num} ---")
        lines.append(f"Q: {query}")
        lines.append(f"A: {answer}")
    return "\n".join(lines)


def format_symbol_search_results(name: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return f"No symbols matching '{name}' found."
    lines = [f"Found {len(results)} result(s) for '{name}':"]
    for meta in results:
        elem_name = meta.get("name", "")
        etype = meta.get("type", "")
        repo = meta.get("repo_name", "")
        rel_path = meta.get("relative_path", "")
        start = meta.get("start_line", "")
        end = meta.get("end_line", "")
        sig = meta.get("signature", "")
        loc = f"L{start}-L{end}" if start and end else ""
        line = f"  - [{etype}] {elem_name}"
        if sig:
            line += f"  |  {sig}"
        line += f"\n    {repo}/{rel_path}:{loc}" if repo else f"\n    {rel_path}:{loc}"
        lines.append(line)
    return "\n".join(lines)


def format_file_summary(
    actual_path: str,
    file_meta: dict[str, Any],
    classes: list[dict[str, Any]],
    top_level_functions: list[dict[str, Any]],
    repo_name: str,
) -> str:
    parts = [
        f"File: {repo_name}/{actual_path}" if repo_name else f"File: {actual_path}"
    ]
    if file_meta:
        parts.append(f"Language: {file_meta.get('language', '?')}")
        mi = file_meta.get("metadata", {})
        parts.append(
            f"Lines: {mi.get('total_lines', '?')} (code: {mi.get('code_lines', '?')})"
        )
        num_imports = mi.get("num_imports", 0)
        if num_imports:
            parts.append(f"Imports: {num_imports}")
    if classes:
        parts.append(f"\nClasses ({len(classes)}):")
        for c in classes:
            sig = c.get("signature", c.get("name", ""))
            loc = f"L{c.get('start_line', '')}-L{c.get('end_line', '')}"
            parts.append(f"  - {sig} ({loc})")
            for m in c.get("metadata", {}).get("methods", []):
                parts.append(f"      .{m}")
    if top_level_functions:
        parts.append(f"\nFunctions ({len(top_level_functions)}):")
        for fn in top_level_functions:
            sig = fn.get("signature", fn.get("name", ""))
            loc = f"L{fn.get('start_line', '')}-L{fn.get('end_line', '')}"
            parts.append(f"  - {sig} ({loc})")
    return "\n".join(parts)


def format_call_chain(
    target_name: str,
    target_type: str,
    target_path: str,
    target_start_line: int,
    callers: list[dict[str, Any]],
    callees: list[dict[str, Any]],
) -> str:
    parts = [
        f"Call chain for '{target_name}' ({target_type})"
        f" at {target_path}:L{target_start_line}"
    ]
    if callers:
        parts.append("\n  Callers (who calls this):")
        for entry in callers:
            indent = "  " * entry.get("indent", 2)
            name = entry["name"]
            if name == "(none)":
                parts.append(f"{indent}(none)")
            else:
                loc = entry.get("loc", "")
                parts.append(f"{indent}- {name} [{loc}]")
    if callees:
        parts.append("\n  Callees (what this calls):")
        for entry in callees:
            indent = "  " * entry.get("indent", 2)
            name = entry["name"]
            if name == "(none)":
                parts.append(f"{indent}(none)")
            else:
                loc = entry.get("loc", "")
                parts.append(f"{indent}- {name} [{loc}]")
    return "\n".join(parts)


def format_explore_code_response(result: dict[str, Any]) -> str:
    groups = result.get("groups")
    if not isinstance(groups, list) or not groups:
        return "No source snippets found for this explore request."

    freshness = result.get("freshness", {})
    completeness = result.get("completeness", {})
    lines = [
        f"Explore: {result.get('query', '')}",
        (
            "Freshness: "
            f"{freshness.get('state', 'unknown')} | "
            "Completeness: "
            f"{completeness.get('state', 'unknown')} "
            f"({completeness.get('returned_snippets', 0)} shown, "
            f"{completeness.get('omitted_snippets', 0)} omitted)"
        ),
    ]
    snapshot_id = result.get("snapshot_id")
    if snapshot_id:
        lines.append(f"Snapshot: {snapshot_id}")

    for group in groups:
        repo = group.get("repo", "")
        file_path = group.get("file", "")
        location = f"{repo}/{file_path}" if repo else str(file_path)
        lines.append(f"\n## {group.get('ref_id', '?')} {location}")
        snippets = group.get("snippets", [])
        if not isinstance(snippets, list):
            continue
        for snippet in snippets:
            if not isinstance(snippet, dict):
                continue
            name = snippet.get("name", "")
            kind = snippet.get("type", "")
            ref_id = snippet.get("ref_id", "?")
            line_label = snippet.get("lines", "")
            score = snippet.get("score", "")
            lines.append(f"- {ref_id} [{kind}] {name} L{line_label} score={score}")
            signature = snippet.get("signature")
            if signature:
                lines.append(f"  {signature}")
            code = snippet.get("code")
            if code:
                language = snippet.get("language", "")
                lines.append(f"```{language}")
                lines.append(str(code))
                lines.append("```")

    next_actions = result.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.append("\nNext actions:")
        for action in next_actions[:5]:
            if not isinstance(action, dict):
                continue
            lines.append(
                f"- {action.get('tool', 'tool')}: {action.get('arguments', {})}"
            )

    return "\n".join(lines)


def format_repo_overview(
    repo_name: str,
    metadata: dict[str, Any],
    languages: dict[str, int],
) -> str:
    parts = [f"Repository: {repo_name}", ""]
    parts.append(f"Summary:\n{metadata.get('summary', 'No summary available.')}")
    if languages:
        parts.append("\nLanguages:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
            parts.append(f"  - {lang}: {count} files")
    structure = metadata.get("structure_text", "")
    if structure:
        parts.append(f"\nDirectory Structure:\n{structure}")
    return "\n".join(parts)


def format_indexed_repos(repos: list[dict[str, Any]]) -> str:
    if not repos:
        return "No indexed repositories found."
    lines = ["Indexed repositories:"]
    for repo in repos:
        name = repo.get("name", repo.get("repo_name", "?"))
        elements = repo.get("element_count", repo.get("elements", "?"))
        size = repo.get("size_mb", "?")
        lines.append(f"  - {name} ({elements} elements, {size} MB)")
    return "\n".join(lines)


def format_delete_repo_metadata(
    repo_name: str, deleted_files: list[str], freed_mb: float
) -> str:
    if not deleted_files:
        return (
            f"No metadata files found for repository '{repo_name}'. "
            "Source code was not modified."
        )
    lines = [f"Deleted metadata for repository '{repo_name}' (source code kept)."]
    lines.append(f"Freed: {freed_mb} MB")
    lines.append("Removed artifacts:")
    for fname in deleted_files:
        lines.append(f"  - {fname}")
    return "\n".join(lines)
