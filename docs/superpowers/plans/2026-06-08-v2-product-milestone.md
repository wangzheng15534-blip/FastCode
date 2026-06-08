# FastCode V2 Product Milestone TODO

**Date:** 2026-06-08
**Scope:** Product-focused V2 plan after reviewing Graphify, Code Review Graph, and CodeGraph
**Decision:** Do not replace FastCode core with any reviewed project. Borrow product workflow patterns and translate them into FastCode-owned contracts.

---

## External Review Summary

Reviewed local audit clones:

| Project | Reviewed commit | Core fit | Borrow | Do not borrow |
|---|---:|---|---|---|
| `safishamsi/graphify` | `29e57cd` | Weak fit. Plain dict extraction and NetworkX graph are looser than FastCode's frozen IR, SCIP bridge, snapshot manifests, typed stores, and projection pipeline. | Human-readable artifact bundle, browser graph/report, confidence/provenance vocabulary, simple `query/path/explain/affected` commands. | Giant orchestration files, undirected direction workarounds, broad LLM semantic extraction, mutating agent-hook installer defaults. |
| `tirth8205/code-review-graph` | `0c9a5ff` | Weak fit. SQLite node/edge hub is useful locally but stringly and less precise than FastCode's canonical IR and graph sidecars. | Review-first blast radius, context savings metadata, bounded traversal with truncation, compact first response plus next tool suggestions, read-only editor graph view. | Direct env/config/filesystem mutation from inner modules, broad `extra` JSON model, best-effort resolver as canonical truth, invasive installer. |
| `colbymchenry/codegraph` | `10defec` | Weak fit for direct import. Central TS `CodeGraph` shell mixes DB, extraction, watcher, traversal, context, and MCP in one runtime class. | Primary `explore` answer, adaptive response sizing, sibling skeletonization, stale-index banners, connect-time catch-up sync, route/handler manifests, explicit heuristic edge provenance. | Node runtime assumptions, mutable SQLite schema as core, huge MCP module, installer complexity, benchmark claims without FastCode-specific validation. |

## Core Decision

FastCode should not borrow a core graph model from these projects. Current FastCode core already has the harder contracts:

- canonical immutable IR in `fastcode/fastcode/ir/`;
- AST plus SCIP merge and semantic resolver patching in `fastcode/fastcode/app/indexing/pipeline/`;
- snapshot manifests and typed store records in `fastcode/fastcode/app/store/`;
- pure graph algorithms in `fastcode/fastcode/graph/analysis.py`;
- query/session/context orchestration through facades injected into `api/` and `mcp/`.

V2 should focus on making this core feel like a complete product: faster first answer, clearer freshness, better review workflows, deterministic graph reports, stronger agent onboarding, and measurable context savings.

## Non-Goals

- Do not replace frozen IR with plain dict nodes/edges or a mutable SQLite symbol hub.
- Do not move FastCode to TypeScript or bundle a Node runtime.
- Do not make MCP/API `entry_frame` modules parse, index, persist, or resolve symbols directly.
- Do not add package-local `ports.py` for DB, filesystem, subprocess, watchers, or network mechanisms.
- Do not make installers mutate agent config, shell files, hooks, or `.gitignore` without dry-run output and explicit consent.
- Do not use broad E2E, happy-path demos, or coverage as proof of ranking, freshness, review-risk, or parser contracts.

---

## Milestone Acceptance Criteria

- A new user can run one local command sequence to initialize, index, inspect status, open a graph/report, and connect MCP without reading architecture docs.
- The primary MCP workflow returns a grouped, line-numbered, graph-backed answer artifact with completeness, freshness, and expansion hints.
- Every indexed answer says whether it is fresh, stale, or unknown at repo/file/snapshot level.
- A diff/review workflow reports blast radius, affected flows, likely tests, high-risk symbols, confidence/provenance, and context savings.
- Browser and CLI reports expose the graph and product state without requiring direct database or artifact knowledge.
- Benchmarks measure user-visible outcomes: first-answer latency, tool calls avoided, token budget, read-backs after explore, stale-answer prevention, and review defect detection.

---

## FCIS Placement Rules For V2 Work

| Product capability | Owner placement | Notes |
|---|---|---|
| Primary explore response | `app/query` `use_flow` orchestration, pure policy in `retrieval/context`, presentation in `mcp` and `api` `entry_frame` modules | Do not put response sizing or source formatting logic into `mcp/server.py` beyond boundary mapping. |
| Adaptive context budget and skeletonization | `retrieval/context` `meaning_core` policy; possible graph features from `ir`/`graph` | Keep policy deterministic and directly testable. Query `use_flow` chooses when to apply it. |
| Freshness/watch/sync | sync decision in `app/indexing`; generic watcher/lifecycle mechanism in `infrastructure` or `runtime_support`; status exposed through `app/store` | Inner packages must not read env or mutate user config. |
| Review/blast-radius workflow | graph facts in `graph`; workflow in `app/query`; persistence in `app/store`; entry mapping in `api`/`mcp` | Review risk is not a generic graph algorithm if it uses product/business policy. |
| Route/handler manifest | extraction in `app/indexing`; pure route facts in `ir` or owner-local contracts if canonical; framework helper execution in `infrastructure/execution` | Heuristic route edges must remain tagged and auditable. |
| Graph/report export | graph analysis in `graph`; report assembly in `app/query` or `app/indexing` depending trigger; static serving in `api` | Human report is a product artifact, not canonical IR. |
| Agent installer/onboarding | CLI/main `entry_frame` and `assembly_root` | Must be explicit, reversible, and dry-run first. |
| Product benchmarks | `fastcode/tests/benchmarks` plus docs/eval scripts | Benchmarks prove budgets only; they do not prove correctness. |

---

## Phase 0: Product Baseline And Guardrails

- [ ] Add a V2 product inventory doc that maps existing commands, API routes, MCP tools, generated artifacts, and missing user workflows.
  - **Likely files:** `docs/product/`, `fastcode/fastcode/mcp/server.py`, `fastcode/fastcode/api/routes.py`, `fastcode/fastcode/main/`.
  - **Evidence:** docs-only diff plus smoke command list. No runtime proof required unless commands change.

- [ ] Add a "no invasive install by default" rule to installer/onboarding docs.
  - **Contract:** dry-run must show exactly which files/config keys would change before applying.
  - **FCIS owner:** `entry_frame`/`assembly_root`.
  - **Evidence:** CLI boundary tests for dry-run/apply/revert mapping if implementation changes.

- [ ] Define product metrics for V2: first answer latency, index freshness lag, token budget, tool-call count, explore read-back count, review-risk precision, and report-open success.
  - **FCIS owner:** test-side benchmark/eval harness, outside the production role graph unless telemetry schema is added.
  - **Evidence:** benchmark fixtures with fixed repositories and stored queries.

---

## Phase 1: Primary Explore Workflow

- [ ] Add a first-class `explore_code` MCP/API workflow that returns grouped source slices, line numbers, graph relationships, evidence refs, and expansion commands in one response.
  - **Borrowed from:** CodeGraph primary `codegraph_explore`.
  - **FastCode placement:** `use_flow` in `fastcode/fastcode/app/query/`; pure rendering policy in `fastcode/fastcode/retrieval/context/`; MCP/API as `entry_frame` modules.
  - **Must include:** `snapshot_id`, `artifact_key`, repo filter, freshness, completeness, omitted sections, and next suggested tool calls.
  - **Evidence:** direct `use_flow` tests for command/result shape; MCP/API boundary tests for parameter validation and response mapping; no LLM calls.

- [ ] Make all graph-backed MCP responses include stable expansion refs.
  - **Current base:** `get_context_bundle`, `expand_context_bundle_ref`, `directed_path`, `impact_analysis`, `find_callers`.
  - **Contract:** every compact source or graph node can be expanded by a deterministic ref.
  - **Evidence:** owner-level tests that compact output refs round-trip to expansion payloads.

- [ ] Add compact `minimal`, `standard`, and `full` detail levels to query tools.
  - **Borrowed from:** Code Review Graph compact first-pass responses.
  - **Contract:** `minimal` does not silently drop required identifiers, file paths, line ranges, freshness, or expansion refs.
  - **Evidence:** decision table over detail level x artifact kind.

---

## Phase 2: Adaptive Context Sizing

- [ ] Implement an adaptive context budget policy that sizes output to the question and graph topology instead of filling a fixed cap.
  - **Borrowed from:** CodeGraph adaptive explore sizing.
  - **FastCode placement:** deterministic policy in `fastcode/fastcode/retrieval/context/`; `use_flow` trigger in `app/query`.
  - **Contract:** named/on-path symbols remain available; off-path interchangeable siblings may be skeletonized; distinct flow steps must not be skeletonized only because budget is tight.
  - **Evidence:** decision table for named, on-path, off-path sibling, base-family, test-file, and distinct-step cases.

- [ ] Add polymorphic sibling skeletonization for repeated implementations.
  - **Contract:** files are skeletonized only when graph evidence shows many same-supertype implementations and the file is off the traced answer path, with explicit exceptions for user-named unique callables.
  - **Evidence:** owner-level unit tests for the finite policy matrix; regression fixtures for read-back-prone cases; mutation testing for gate predicates once policy stabilizes.

- [ ] Add context-saving metadata to responses.
  - **Borrowed from:** Code Review Graph context savings and CodeGraph benchmark reporting.
  - **Contract:** report estimated omitted characters/tokens, skeletonized file count, full file count, and expansion refs.
  - **Evidence:** unit tests for deterministic counters; snapshot output only after semantic assertions.

---

## Phase 3: Freshness, Watch, And Sync UX

- [ ] Add repo/file freshness status to indexed-answer boundaries.
  - **Borrowed from:** CodeGraph stale-index banners and connect-time catch-up.
  - **Current base:** `fresh` fields already exist in agent-context records; expand them into product-level status.
  - **Contract:** answers classify freshness as `fresh`, `stale`, `unknown`, or `unindexed`, with changed paths and last indexed snapshot when known.
  - **Evidence:** state-machine tests over unchanged, modified, deleted, untracked, and missing manifest states.

- [ ] Add `fastcode status` product output with repo health, last snapshot, pending changed paths, storage backend, MCP connection hints, and report links.
  - **FastCode placement:** CLI `entry_frame` calls `StoreFacade` and indexing status `use_flow`.
  - **Evidence:** CLI boundary tests for empty repo, indexed repo, stale repo, and storage unavailable.

- [ ] Add explicit `watch`/`sync` flow with debounce and shutdown semantics.
  - **Borrowed from:** Code Review Graph watch mode and CodeGraph watcher catch-up.
  - **FastCode placement:** generic watcher lifecycle under infrastructure/runtime support; semantic sync plan under `app/indexing`.
  - **Evidence:** lifecycle tests with fake clock/temp repo; integration test with file modify/delete; no direct sleeps in deterministic tests.

---

## Phase 4: Review And Blast-Radius Product

- [ ] Add a `review_diff` workflow that accepts a base/ref or changed path list and returns affected symbols, paths, flows, and likely tests.
  - **Borrowed from:** Code Review Graph review-first blast radius and CodeGraph affected flows.
  - **FastCode placement:** `app/query` `use_flow` using `graph/analysis` and snapshot manifests; entry in MCP/API/CLI.
  - **Contract:** changed-file input is resolved to snapshot symbols; traversal has max-hop and truncation metadata; test suggestions cite graph evidence.
  - **Evidence:** direct `use_flow` tests using synthetic snapshots; graph traversal edge cases; BDD scenario only for explicit review acceptance journey under `fastcode/tests/bdd/`.

- [ ] Add review risk scoring with visible confidence/provenance.
  - **Borrowed from:** Graphify confidence labels and Code Review Graph risk index.
  - **Contract:** risk score must expose inputs: changed kind, centrality/degree, boundary role, route/handler involvement, tests touched, freshness, and heuristic edge count.
  - **Evidence:** decision table for finite risk factors; mutation test for critical gate thresholds; no snapshot-only oracle.

- [ ] Add `affected_tests` suggestions with owner-local evidence.
  - **Contract:** suggestions distinguish direct graph dependency, same-package heuristic, naming heuristic, and unknown.
  - **Evidence:** unit tests for deterministic mapping plus integration fixtures for real test discovery.

---

## Phase 5: Graph Reports And Visualization

- [ ] Add a report bundle artifact: machine JSON, Markdown report, and browser graph HTML.
  - **Borrowed from:** Graphify `graph.html`, `GRAPH_REPORT.md`, `graph.json`.
  - **FastCode placement:** report assembly `use_flow`; canonical graph facts remain in `ir`/`graph`/`app/store`.
  - **Must include:** top modules, entry points, high-degree nodes, import cycles, cross-cluster links, ambiguous/heuristic edges, stale paths, and suggested questions.
  - **Evidence:** semantic assertions over generated report sections; snapshot tests only after required sections are asserted.

- [ ] Add graph report commands: `open graph`, `explain symbol`, `path`, `affected`, and `communities`.
  - **Current base:** MCP already has `directed_path`, `impact_analysis`, `leiden_clusters`, `steiner_path`, and `find_callers`.
  - **Contract:** CLI/API/MCP commands share the same `use_flow` result schema.
  - **Evidence:** schema/roundtrip tests and boundary validation tests.

- [ ] Add truncation and confidence metadata to graph analysis outputs.
  - **Borrowed from:** Code Review Graph bounded traversal metadata and Graphify confidence labels.
  - **Contract:** max-hop, max-node, omitted count, and heuristic/provenance totals are present whenever output is bounded.
  - **Evidence:** graph analysis unit tests for truncation boundaries and provenance aggregation.

---

## Phase 6: Route, Framework, And Dynamic Edge Product Layer

- [ ] Add route/handler manifests for web frameworks where FastCode can extract reliable evidence.
  - **Borrowed from:** CodeGraph route/handler query concepts.
  - **FastCode placement:** extractor/resolver `use_flow` under `app/indexing`; canonical facts only if typed and provenance-backed.
  - **Contract:** route -> handler -> implementation chains must be tagged by source: SCIP, AST, framework parser, or heuristic.
  - **Evidence:** parser fixtures for supported frameworks; negative fixtures for unsupported shapes; no broad "all frameworks" claim.

- [ ] Add explicit synthesized-edge provenance to product outputs.
  - **Borrowed from:** CodeGraph heuristic/dynamic edge provenance and Graphify confidence vocabulary.
  - **Contract:** heuristic edges are visible and never silently promoted to extracted edges.
  - **Evidence:** IR/graph boundary tests proving provenance survives serialization, storage, graph analysis, and MCP formatting.

- [ ] Add an ambiguity report for unresolved refs and low-confidence links.
  - **Contract:** unresolved calls/imports/routes are reportable artifacts, not hidden failures.
  - **Evidence:** golden dangerous fixtures for unresolved/ambiguous parser output; negative tests for false promotion.

---

## Phase 7: Agent Onboarding And Distribution

- [ ] Add `fastcode init`, `fastcode connect-agent`, `fastcode doctor`, and `fastcode uninstall-agent` UX.
  - **Borrowed from:** all three projects' agent-first install flows.
  - **Contract:** `connect-agent --dry-run` is default; apply requires explicit target and confirmation flag in non-interactive mode.
  - **Evidence:** CLI boundary tests over Codex/Cursor/Claude-style config paths using temp HOME; revert tests for generated config entries.

- [ ] Add MCP server instructions that steer agents to `explore_code` before grep/read loops when an index is fresh.
  - **Borrowed from:** CodeGraph server instructions.
  - **Contract:** instructions must also say when not to trust the index: stale, unindexed, unsupported language, or missing snapshot.
  - **Evidence:** text contract tests for required warning clauses.

- [ ] Add a short product quickstart and troubleshooting page.
  - **Contract:** no obsolete package-root paths; commands run from repository root.
  - **Evidence:** docs link check or command smoke where available.

---

## Phase 8: Benchmarks And Product Validation

- [ ] Build a product eval harness for indexed vs unindexed agent workflows.
  - **Borrowed from:** CodeGraph benchmark methodology, but revalidated on FastCode.
  - **Metrics:** cost proxy, token count, tool calls, first-answer latency, file read-backs after explore, answer citation coverage, and stale-answer prevention.
  - **Evidence:** benchmark tests with fixed repos/queries and median reporting; results are product evidence, not correctness proof.

- [ ] Add review workflow evals on known diffs.
  - **Contract:** diff risk output must identify affected symbols, at least one relevant test target when graph evidence exists, and no unsupported certainty language when evidence is heuristic.
  - **Evidence:** BDD scenario for the explicit review journey plus owner-level tests for risk and traversal decisions.

- [ ] Add release readiness gates for V2.
  - **Required gates:** architecture tests, focused owner tests for changed contracts, API/MCP boundary tests, product benchmark smoke, docs command smoke.
  - **Not accepted:** "coverage is high", "E2E covers it", or "benchmark shows it is faster" as correctness evidence.

---

## Test Policy Summary

Use this chain for each V2 implementation task:

```text
contract -> FCIS owner -> input-space shape -> failure class -> minimum evidence
```

Minimum evidence by task family:

| Task family | Contract owner | Input-space shape | Failure class | Minimum evidence |
|---|---|---|---|---|
| Adaptive skeletonization | `retrieval/context` `meaning_core` | finite semantic product | wrong branch / ranking drift | decision table plus edge rows; mutation for stable gate predicates |
| Freshness status | `app/indexing` `use_flow` plus store status | state machine | stale answer / wrong state | state x event matrix and boundary status tests |
| CLI/MCP/API onboarding | `entry_frame` / `assembly_root` | public protocol input | malformed mapping / unsafe mutation | valid + malformed boundary tests; temp HOME dry-run/apply/revert |
| Review risk | `app/query` `use_flow` and graph `meaning_core` | finite factor product plus graph traversal | wrong result / unsupported certainty | decision table, graph fixtures, mutation for critical thresholds |
| Parser/framework route manifests | parser/resolver owner | external tool/parser output | parser corruption / false promotion | golden fixtures, corrupt/unsupported fixtures, provenance assertions |
| Report generation | `use_flow` report assembly | deterministic formatted output | missing required section | semantic section assertions first, snapshots second |
| Benchmarks | benchmark harness | numeric range / run distribution | non-functional regression | explicit threshold or trend checks; not correctness proof |
| BDD review journey | `acceptance_test` only when journey is explicit business obligation | scenario | design-intent drift | BDD scenario under `fastcode/tests/bdd/`, with owner-level tests for underlying rules |

BDD files must stay in the BDD subfolder and should not be mixed with unit tests. Use scenario names for intent; do not add a `bdd` postfix just to signal the layer.

---

## Open Questions Before Implementation

- Should V2 prioritize MCP-first local agent UX, CLI-first local developer UX, or browser report UX for the first release cut?
- Which repositories become fixed product benchmarks for FastCode rather than inherited claims from external projects?
- Should route/handler manifests start with Python web frameworks only, or include TypeScript/Go/Rust once SCIP evidence is available?
- What is the acceptable default for watcher behavior: manual `sync`, opt-in background watch, or connect-time catch-up only?
- Which agent config targets are in scope for reversible installers in V2?
