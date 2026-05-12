# FastCode Architecture

This document describes the architecture currently implemented in `develop`.
It is intentionally narrower than older README or marketing language.

## Status

FastCode is in a hardened pre-release state.

The core indexing, IR merge, retrieval, semantic-upgrade, and shell boundaries
are materially stronger than the original prototype, but the stable-release bar
is not met yet. The main remaining gap is operational and dataflow hardening,
not basic pipeline existence.

The maintained implementation tracker is
[IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md).

## Design goals

FastCode is a repository-understanding system for coding agents with four core
goals:

1. Build canonical code facts once and reuse them across retrieval paths.
2. Keep the functional core test-heavy and free from shell concerns.
3. Make settings and environment access explicit at boundaries.
4. Support a package layout that can keep growing without collapsing imports.

## Package layout

The importable package is `fastcode/src/fastcode/`.

Current top-level modules:

- `api/` HTTP and web transport shells
- `graph/` graph construction primitives
- `indexing/` repository loading, extraction orchestration, projections
- `ir/` canonical frozen IR types and helpers
- `main/` composition root and CLI wiring
- `mcp/` MCP transport shell
- `query/` query understanding and answer generation orchestration
- `retrieval/` retrieval orchestration and iterative agent behavior
- `retrieval/core/` pure retrieval logic and scoring helpers
- `schemas/` boundary models and typed config
- `scip/` SCIP adapters and loaders
- `semantic/` semantic resolver registry and helper-backed upgrades
- `store/` persistence orchestration
- `store/infrastructure/` lower-level storage adapters
- `utils/` compatibility helpers and shared utilities

## Enforced layer DAG

The import graph is enforced by
`fastcode/tests/architecture/test_import_graph.py`.

Layers:

- Layer 0:
  - `schemas`
  - `ir`
  - `utils`
  - `retrieval.core`
  - `store.infrastructure`
- Layer 1:
  - `graph`
  - `indexing`
  - `query`
  - `retrieval`
  - `scip`
  - `semantic`
  - `store`
- Layer 2:
  - `api`
  - `mcp`
  - `main`

Rule: lower layers must not import upward, and cross-layer cycles are forbidden.

## Architecture contract

The repo currently enforces these rules in code, tests, and module-local
`ruff.toml` files:

1. Pydantic stops at the boundary.
   - `schemas/` owns Pydantic validation and settings parsing.
   - `ir/`, `graph/`, and `retrieval/core/` remain Pydantic-free.

2. Pure packages stay shell-free.
   - `ir/`, `graph/`, and `retrieval/` may not pull in `pydantic`,
     `sqlite3`, `subprocess`, or `urllib`.

3. Package roots stay thin.
   - `fastcode/__init__.py` and subpackage `__init__.py` are compatibility
     surfaces, not composition roots.
   - Internal packages must not rely on `from fastcode import ...` re-exports.

4. Translation stays explicit.
   - Shell packages must not mass-assign with `**model_dump()` or `**__dict__`.
   - Mapping between API payloads, runtime config, IR records, and store records
     must be visible in code.
   - Active hot paths should prefer typed records or explicit field serializers
     over generic `to_dict()`, `from_dict()`, `row_to_dict()`, or recursive
     `safe_jsonable()` cleanup.
   - Vector and graph hot paths should keep data in native-backed carriers
     (`np.ndarray`, FAISS, pgvector adapters, `IRGraphView`) until a storage,
     API, export, or compatibility boundary requires materialization.
   - Vector boundary helpers must make ownership/copy behavior explicit; a
     view-oriented path must not mutate caller-owned buffers while normalizing or
     sanitizing values.

5. Inner packages do not read environment directly.
   - `indexing`, `query`, `retrieval`, `store`, `mcp`, `schemas`, and related
     inner packages must not call `os.getenv`, `os.environ[...]`, or
     `load_dotenv()` directly.
   - Environment loading is centralized in config preparation.

## Runtime configuration flow

The current config boundary is:

1. Raw YAML and `.env` input enter through `fastcode.utils._compat`.
2. `prepare_runtime_config_mapping(...)` resolves paths and overlays env-backed
   runtime settings.
3. `fastcode.schemas.config` validates that mapping with Pydantic boundary
   models and returns frozen dataclass runtime config as `FastCodeConfig`.
4. `fastcode.main.fastcode.FastCode` owns the runtime config instance and
   exposes explicit mutation points for runtime-only overrides.
5. Legacy dict consumers still receive a compatibility dict view, but the
   canonical config is typed and frozen.

Important current rule:

- direct environment reads should happen only in config preparation code, not in
  query, indexing, store, or MCP leaf modules.

## Data model

### Canonical IR

The deepest internal truth for code structure is the IR layer under `ir/`.
Higher-level graph, retrieval, and storage representations derive from it.

The IR path is designed to support:

- repository snapshots
- file and document identity
- symbol identity and relationships
- provenance across extraction paths
- merge of AST-derived and SCIP-derived facts

### Typed boundaries

The codebase is moving away from dict-heavy shell payloads toward typed records.
Already-landed areas include:

- frozen runtime config in `schemas/config.py`
- typed snapshot and manifest records
- typed store-facing record flows in parts of persistence
- explicit `CodeElement` serializers for retrieval, graph persistence, and
  index-storage boundaries
- explicit snapshot-file serializers in `store/snapshot.py` instead of routing
  persistence through `IRSnapshot.to_dict()` / `IRSnapshot.from_dict()`
- compact snapshot symbol-index sidecars for query-time alias registration and
  single-symbol record lookup without loading the whole `IRSnapshot` on current
  snapshots

Still incomplete:

- store, query, and projection records are not fully typed end to end
- several shell and persistence paths still expose raw dict payloads
- semantic resolver patch application still copies whole IR collections, though
  it no longer uses generic dict round trips
- materialization guard coverage is narrower than the full architecture target;
  MCP graph helpers, projection transforms, snapshot persistence, and
  query-time compact symbol-index registration still need explicit guard
  treatment

## Indexing pipeline

The implemented indexing path is a layered pipeline:

1. repository load and file scan
2. structural extraction
3. optional SCIP extraction
4. canonical IR merge and validation
5. semantic upgrade by language-specific resolvers
6. graph, index, and projection persistence

Current hardened properties:

- explicit pipeline layer status, metrics, and warnings
- non-silent fallback behavior
- persisted SCIP lineage metadata
- helper-backed semantic upgrades with structural fallback
- snapshot-level regression coverage around reuse and concurrency
- explicit storage materialization for vector-store, PG retrieval, and
  unit-artifact boundaries instead of generic `CodeElement.to_dict()` expansion
- native vector staging via `np.ndarray[np.float32]` through indexing, vector
  store insertion/search, embedding cache payloads, and PostgreSQL pgvector
  binding, with literal fallback only when the Python pgvector adapter is absent
- compact IR graph persistence/loading through `IRGraphView`, with NetworkX
  reserved for compatibility/export and projection algorithms that still need it

### Incrementality

Current verdict: partially incremental, not fully incremental.

What already works:

- unchanged files can bypass repeated parse and embedding work via manifest-first
  planning in the active path
- manifest-first incremental planning now refuses reuse when prior or current
  file entries lack content fingerprints instead of trusting size/mtime fallback
- a single loader inventory can be shared across snapshot identity,
  incremental planning, and the AST extraction path during a pipeline run
- file-level `content_hash` / `blob_oid` metadata is persisted on AST-derived IR
  file units
- incremental plans can rebuild changed AST IR and merge it with the previous
  snapshot instead of rebuilding unchanged-file AST IR
- changed-unit embeddings can be reused when stable unit identity and
  `embedding_text_hash` match
- persisted vector/BM25 path shards, plus conservative legacy graph path shards,
  can reuse compatible previous snapshot artifact files for unchanged paths
  under the new artifact key
- embeddings are deduplicated and cached with model-aware keys
- helper-backed semantic resolvers can scope work to changed paths
- package/path repair-frontier logic can scope semantic refresh and SCIP reruns
  when changed API or edge surfaces are known
- local path indexing can scan the caller-provided repository in place by
  default, with explicit workspace-copy mode still available when isolation is
  required

What is still missing before stable-release claims:

- file-native artifact shard reuse as the primary execution model; persistence
  reuse exists for vector/BM25 and conservative graph shards, but not yet for all
  IR graph, relational fact, SCIP/tool, and temporary build surfaces
- local path loading is read-only and in-place by default, but explicit
  workspace-copy mode still copies whole working trees before incremental
  planning and needs content-addressed or hardlinking reuse
- truly incremental SCIP/tool-backed extraction in widened and unsupported cases
- one shared inventory/fingerprint planner across snapshot identity, incremental
  planning, SCIP scope, file-artifact reuse, and publication; loader inventory
  sharing exists for the active AST path, but not yet as a canonical planner
  object across every stage
- deterministic cache invalidation across schema, model, and tool changes
- provider batching and provider timing visibility; compatibility and
  all-cache-hit fingerprint checks no longer force provider startup merely to
  discover an unconfigured dimension

## Retrieval and query flow

The query stack is split between:

- `query/` for intent detection, rewriting, answer generation, and selection
- `retrieval/` for orchestration
- `retrieval/core/` for pure scoring, combination, graph-boundary logic, and
  context assembly

Current design:

1. query understanding extracts intent and search cues
2. retrieval combines semantic, keyword, and graph-aware evidence
3. iterative retrieval can escalate when direct retrieval is insufficient
4. answer generation consumes retrieved evidence plus provenance

Current hardened properties:

- agency-mode preserves cheap detected intent
- caller-filter handling is fixed after rerank
- semantic escalation can drive real IR-graph expansion

Known serving/materialization gap:

- public shell graph tools do not yet consistently use immutable artifact
  handles or compact graph/symbol indexes; MCP graph helpers still full-load
  snapshots or materialize NetworkX graphs per request, while main
  composition-root graph helpers use compact graph handles on the primary path

## Agent Context Integration

FastCode's shells expose repository intelligence through CLI, API, web, and MCP
entrypoints. The first agent-context v0 is now implemented: typed turn,
working-memory, evidence-ref, risk/contract, and handoff artifacts can be
compiled, persisted, fetched, expanded, and passed into later turns. The next
architecture step is the bundle layer: durable cacheable context artifacts that
agents can budget, distill, reactivate, and expand, rather than only returning
one-shot answers or raw retrieval rows.

The design follows the existing source-of-truth rule:

- authoritative repository facts remain in snapshots, IR, graph facts, vector
  artifacts, manifests, and projection artifacts
- agent-facing context is a derived artifact with explicit provenance and
  invalidation metadata
- lossy summaries are allowed only when they retain expandable source refs

FastCode should model agent context as two coupled memory surfaces, not as one
giant append-only prompt:

- historical truth:
  append-only, complete, external, and restorable
- cognitive working memory:
  small, recent, relevant, and cache-friendly

In practice that means:

- snapshots, projections, tool observations, verifier outputs, and artifact
  refs live in the historical layer
- the prompt-facing context window is a compiled working-memory view rebuilt
  from typed state after meaningful observations
- "rewrite" means replacing the compiled working-memory view, not mutating the
  historical evidence journal

The broader target context model has four record families:

- `EvidenceRef`: pointer to a code fact, projection chunk, file range, symbol,
  graph path, tool trace, or prior dialogue turn
- `ContextBundle`: ranked, token-budgeted collection of evidence refs plus
  rendered text, summaries, source refs, freshness, and expansion handles
- `DistillationRecord`: model/prompt-fingerprinted summary of one or more
  evidence refs, including covered refs and omitted refs
- `ActivationRecord`: record that an agent actually used a bundle or evidence
  item for a task, optionally with outcome signals such as edited files, tests
  run, accepted/rejected sources, or follow-up expansions

Current implementation status: `EvidenceRef`, `ToolObservation`, `RiskState`,
`AcceptanceContract`, `TurnIntent`, `TurnPlan`, `WorkingMemoryArtifact`,
`TurnJournal`, and `HandoffArtifact` exist in `retrieval/core/` with explicit
cache records and shell facades. Durable `ContextBundle`, `DistillationRecord`,
and `ActivationRecord` records are still target architecture, not implemented
release features.

This split also matches the practical lessons from long-turn agents such as
Manus:

- keep `L0` and selected `L1` sections stable for provider prompt/KV caches
- keep the execution/evidence trace append-only for replay and deterministic
  cache behavior
- keep large detail external and restorable through file/artifact/reference
  handles rather than inline prompt carry-over
- keep a rewritten tail working set near the end of the prompt so current goals,
  hypotheses, and next actions stay in recent attention

This is adjacent to, but stronger than, conversation-only dynamic context
pruning. A pruning plugin can replace old messages with summaries and protect
important tool outputs. FastCode can also use repository structure:

- compress by symbol/file/module/cluster/path/change frontier
- keep summaries expandable to source file, line range, symbol ID, snapshot, and
  artifact key
- key cached bundles by query/task fingerprint, snapshot artifact key,
  projection algorithm version, embedding fingerprint, distillation prompt
  fingerprint, and token budget
- invalidate cached context when cited evidence, projection logic, embedding
  identity, or source snapshot changes

The projection system already provides the right shape for progressive
disclosure:

- `L0`: compact orientation and task framing
- `L1`: navigation, clusters, relationships, and decision context
- `L2`: cited chunks and raw evidence anchors

Future MCP/API tools should expose context operations directly:

- build a context bundle for a task/query/snapshot
- expand an evidence ref to the next detail level
- distill or recompress stale bundle sections
- reactivate prior task bundles
- record activation feedback from agent runs

Open design questions:

- how to fingerprint tasks enough to reuse useful context without overfitting
- which bundle sections should be stable prompt-cache prefixes versus late
  dynamic context
- whether ranking should stay deterministic in v1 or learn from activation
  records
- how external memory systems should link to FastCode bundles without copying
  code facts into a separate authoritative store

### Context Package Placement

The context-bundle work should fit the existing layer DAG instead of becoming a
new composition root:

- `retrieval/core/` owns pure scoring, budget selection, evidence-ref
  normalization, bundle assembly, and text rendering helpers.
- `query/` owns orchestration: query understanding, retrieval, optional
  semantic escalation, bundle creation, answer generation, and activation
  recording.
- `store/` owns persistence of bundle, distillation, and activation records
  through typed records plus explicit serializers.
- `api/` and `mcp/` expose context operations as shell adapters only.
- `indexing/projection.py` and `indexing/projection_transform.py` continue to
  own L0/L1/L2 projection generation.

The v1 implementation should not introduce Pydantic models inside
`retrieval/core/`, should not make `store/` import API schemas, and should not
route bundle persistence through generic `dict` or recursive JSON cleanup on hot
paths.

The detailed pattern and module design for this next step lives in
[AGENT_INTEGRATION_PATTERNS.md](./AGENT_INTEGRATION_PATTERNS.md).

### Bundle Assembly Flow

A request-local bundle should be built as a deterministic pipeline:

1. Resolve snapshot and artifact handle.
2. Run retrieval and optional graph/projection expansion.
3. Convert retrieved rows, projection chunks, graph paths, and dialogue turns
   into `EvidenceRef` records.
4. Score evidence by relevance, graph support, source freshness, prior
   activation, and citation density.
5. Allocate a token budget across orientation, navigation, evidence, and raw
   snippets.
6. Reuse existing distillations when fingerprints match.
7. Render a `ContextBundle` with expandable evidence handles.
8. Persist only the derived bundle metadata and summaries; raw code facts remain
   in the snapshot/projection stores.
9. After the agent/model consumes the bundle, write an `ActivationRecord` with
   outcome signals when available.

The first implementation should keep ranking deterministic and treat activation
records as telemetry. Learning from activation history should wait until there
is enough evaluation data to avoid reinforcing accidental retrieval choices.

### Turn-Centric Toolchain Integration

The agent loop is now partially centered on typed turns instead of raw prompt
history plus ad hoc tool output.

Current code already exposes the v0 substrate:

- `retrieval/core/agent_context.py` owns typed evidence, observation, risk,
  contract, working-memory, journal, and handoff records
- `retrieval/core/context_compiler.py` renders FCX working-memory and handoff
  artifacts from typed state
- `query/handler.py` owns session-aware orchestration and persists compiled turn
  artifacts
- `retrieval/iterative.py` models multi-round confidence, tool calls, and
  keep/discard decisions while accepting compiled context from prior turns
- `retrieval/agent_tools.py` is the read-only repository boundary
- `store/cache.py` persists dialogue turns, session indexes, turn journals,
  working memory, and handoff artifacts

The remaining gap is deterministic agent control, not the absence of a turn
substrate. Verifier transitions, strict FCX parse-back, branch/ask/abstain
policy gates, rejected-hypothesis lifecycle, and cache-stability/replay proofs
still need to become first-class typed behavior while preserving:

- the append-only historical journal
- the replaceable compiled working-memory view

The target per-turn record model is:

- `TurnIntent`: user goal, clarified task, repo/snapshot scope, and current
  requested outcome
- `TurnPlan`: current hypotheses, open questions, planned tool actions, stop
  conditions, and token budget
- `ToolObservation`: normalized result of one tool call, with:
  - tool name
  - parameters
  - snapshot/artifact key
  - structured payload
  - evidence refs emitted
  - cost
  - warnings/failures
- `ObservationJournal`: append-only turn-local or session-local sequence of tool
  and verifier observations that remains replayable and restorable
- `BeliefState`: current candidate explanations or edit plans, with support and
  conflict edges
- `RiskState`: bounded uncertainty state used for action selection, with
  separate signals for evidence gaps, conflicts, freshness risk, requirement
  ambiguity, execution risk, and verifier status
- `AcceptanceContract`: current definition of done, required evidence classes,
  required verifiers, allowed writes/tools, and ask/abstain thresholds
- `RejectedHypothesisLedger`: append-only negative memory for killed
  possibilities, including why they were rejected and what would justify
  reopening them
- `WorkingSet`: what must remain in the next prompt window
- `HandoffArtifact`: compact external artifact for clean-context reset,
  delegation, resume, or rollback
- `TurnOutcome`: selected answer/edit recommendation, abstention, verifier
  result, or next-turn carry-forward state

This makes the agent loop explicit:

1. Ingest user turn and resolve snapshot/session scope.
2. Compile or refresh the current `AcceptanceContract`.
3. Build or refresh the current `BeliefState`.
4. Score a `RiskState` from explicit uncertainty signals, not only model
   wording.
5. Choose a `TurnPlan`:
   retrieve, inspect, expand graph, verify, ask, answer, branch, or reset.
6. Execute tool calls.
7. Normalize each result into `ToolObservation` records and append it to the
   `ObservationJournal`.
8. Apply promotion and rejection rules to hypotheses, facts, and protected
   constraints.
9. Recompute `RiskState`.
10. Recompile the next `WorkingSet` from typed state, not from the raw
    transcript.
11. Either commit, branch, verify further, reset from a handoff artifact, ask
    the user, or abstain.

The key point is that the working-memory rewrite happens after every meaningful
tool observation, not only at coarse session-compaction points. The journal is
append-only; only the prompt-facing working set is replaced.

### Context Compiler

The bundle renderer should behave like a context compiler for the next agent
turn.

Its inputs are:

- current `TurnIntent`
- active `BeliefState`
- recent `ToolObservation` records
- current `WorkingSet`
- reusable `ContextBundle` / `DistillationRecord` artifacts
- token budget and model constraints

Its outputs are:

- a stable prefix:
  task, constraints, accepted facts, protected evidence
- a dynamic working section:
  active hypotheses, unresolved uncertainty, fresh tool observations
- a recent journal slice:
  only the few append-only observations still relevant to the current choice
- expandable handles:
  evidence refs, graph-path refs, projection refs, prior verified summaries
- a next-action section:
  what the model is allowed to do on this turn

This is where FastCode should outperform generic compaction systems. Instead of
compressing old text ranges, it can rewrite context around:

- symbol frontiers
- changed files
- dependency neighborhoods
- verification status
- accepted versus rejected hypotheses

The compiler should therefore target the cognitive working-memory surface:

- keep the historical journal intact
- emit a small cache-friendly stable block
- emit a rewritten turn block and recent-observation slice
- keep older detail behind `L3` or artifact/file refs so it is restorable but
  not always resident in the prompt

The model-facing output of the compiler should use the compact line-oriented
DSL defined in [AGENT_CONTEXT_DSL.md](./AGENT_CONTEXT_DSL.md). That DSL is only
a prompt/rendering format. The canonical state remains typed records and
explicit serializers.

### Execution Regimes

FastCode should use one truth model with two execution regimes rather than many
informal memory modes:

- short-horizon direct mode:
  low ambiguity, low execution risk, narrow scope, and minimal need for
  branching or reset; the compiler emits `L0`, selected `L1`, the current
  contract, a small working set, and only the most relevant observations
- long-horizon managed mode:
  planning, debugging, research, multi-file edits, or repeated verification;
  the compiler emits explicit hypotheses, risk state, rejected-hypothesis
  memory, and reset/handoff support

Promotion from direct mode to managed mode should be deterministic. Typical
triggers:

- token pressure forces multiple rewrites in one task
- more than one plausible hypothesis remains active
- verifier failures or contradictory evidence appear
- write frontier expands beyond one file or one local symbol neighborhood
- user intent is inherently long-running:
  debugging, planning, research, architecture, or orchestrated editing

Demotion back to direct mode is allowed only after the contract narrows,
uncertainty drops, and the remaining action is local.

### Possibility Management

Academic uncertainty handling should map to explicit runtime objects.

Instead of one implicit chain of thought, maintain a bounded set of candidate
possibilities:

- `Hypothesis`: a candidate answer, root cause, or edit plan
- `SupportRef`: evidence refs supporting it
- `ConflictRef`: evidence refs or verifier results against it
- `ConfidenceSignals`: retrieval margin, branch agreement, verifier status,
  tool failure, missing evidence, and model confidence as a weak signal

Turn planning then becomes possibility management:

- if one hypothesis dominates and verifier signals are clean, answer or edit
- if multiple hypotheses remain plausible, branch search or ask a question
- if evidence is stale or weak, retrieve or inspect more
- if verifier results contradict the leading hypothesis, rewrite the working set
  around the surviving alternatives

This is closer to belief-state planning than to plain retrieval-plus-generation.

### Decision Control Objects

The current design needs stronger control objects than `Hypothesis` plus a
summary rewrite.

`RiskState` should be a vector, not one scalar confidence value. Suggested
fields:

- `evidence_gap`: how much required evidence is still missing
- `conflict_level`: how much strong evidence disagrees
- `freshness_risk`: whether cited support may be stale against the current
  snapshot or runtime
- `requirement_ambiguity`: whether the user intent or success condition is
  under-specified
- `execution_risk`: whether emitting an answer/edit now could cause expensive
  or hard-to-revert mistakes
- `verifier_status`: `clean`, `pending`, `mixed`, `failed`, or `blocked`
- `action_bias`: current preferred next move:
  answer, edit, retrieve, verify, ask, branch, reset, or abstain

`AcceptanceContract` should make "done" explicit instead of leaving it inside
prompt prose. Suggested fields:

- `requested_outcome`
- `required_evidence_kinds`
- `required_verifiers`
- `allowed_tools`
- `allowed_write_scope`
- `must_ask_before`
- `must_abstain_when`
- `done_condition`

`PromotionRuleSet` should be harness-owned, deterministic, and versioned. The
model may suggest a rewrite or hypothesis update, but it should not directly
promote observations into authoritative facts.

`RejectedHypothesisLedger` is the negative-memory complement to the working set.
Without it, rewritten prompts tend to reintroduce already-killed ideas. Each
ledger entry should record:

- killed hypothesis ID
- killer evidence or verifier refs
- rejection reason code
- snapshot/version basis
- reopen condition

`HandoffArtifact` is the clean-context transfer unit. It should be external,
restorable, and stable enough for caching. A handoff artifact should contain:

- normalized task intent
- current acceptance contract
- accepted facts and protected constraints
- surviving hypotheses
- rejected-hypothesis ledger slice
- unresolved questions
- allowed tools and write scope
- current code/change frontier
- recommended first action for the next agent turn

`ResetPolicy` decides when the current working memory should be rewritten in
place and when a fresh context should start from a handoff artifact. Typical
reset triggers:

- repeated non-progress across several turns
- prompt drift where the recent working set no longer reflects the contract
- too many unresolved hypotheses for the remaining budget
- verifier churn without convergence
- large write frontier that now needs a specialized clean-context worker

### Promotion And State Transitions

FastCode should treat state promotion as an explicit lattice, not as free-form
summary replacement.

Recommended transitions:

- `ToolObservation` -> cited note:
  a normalized observation with explicit refs enters the append-only journal
- cited note -> hypothesis support/conflict:
  observations and distillations update support or conflict edges on one or
  more hypotheses
- `Hypothesis(state=open)` -> `Hypothesis(state=favored)`:
  at least one fresh supporting ref exists and no stronger conflicting verifier
  result is present
- `Hypothesis(state=favored)` -> accepted fact:
  support comes from an authoritative repo fact, explicit source inspection, or
  verifier/tool result that satisfies the current acceptance contract
- accepted fact -> protected constraint:
  the fact is required for safety, correctness, or contract fulfillment and
  must remain pinned in the stable prefix or working set
- `Hypothesis(*)` -> `Hypothesis(state=rejected)`:
  a verifier fails, stronger contradictory evidence appears, the cited basis
  goes stale, or the contract rules the path invalid
- rejected hypothesis -> reopened hypothesis:
  only when new evidence, a new snapshot, or a changed contract invalidates the
  rejection basis

The model may emit rewrite candidates and action requests. The compiler owns
promotion, rejection, and reopening.

### Decision Policy

The runtime should choose actions from typed state, not from a single prose
self-assessment.

Suggested policy defaults:

- answer or edit only when the acceptance contract is satisfied and
  `RiskState` is bounded
- ask the user when requirement ambiguity is high and execution risk is not
  trivial
- branch only when multiple live hypotheses remain and the branches are
  independently testable
- reset or hand off when context drift or role isolation matters more than
  transcript continuity
- abstain when required evidence or required verifiers cannot be obtained under
  current permissions or budget

### Tool Boundary Rules

Every tool integrated into the loop should follow the same contract:

- return a structured payload, not only display text
- emit zero or more `EvidenceRef` records
- declare whether it is:
  - retrieval
  - graph expansion
  - filesystem inspection
  - verification
  - environment/runtime inspection
- include cost and freshness metadata
- be replayable or cacheable when snapshot/artifact inputs are unchanged

Tool output text for human readability can still exist, but the agent loop
should consume typed observations first and rendered text second.

### Cache And Invalidation

Bundle cache keys should include all inputs that can change meaning:

- snapshot ID and artifact key
- query or task fingerprint
- selected repositories, filters, and requested budget
- retrieval policy version
- projection algorithm version
- embedding fingerprint
- distillation model, prompt, and renderer fingerprint

Invalidation should be conservative:

- source snapshot or artifact mismatch is a hard miss
- projection or renderer mismatch rebuilds the bundle text
- embedding mismatch reruns evidence ranking that depends on vector scores
- distillation fingerprint mismatch keeps evidence refs but rewrites summaries
- changed cited files/symbols invalidate directly citing bundles
- changed dependency frontiers invalidate impact/path bundles that depend on
  affected graph neighborhoods

Summaries are never authoritative. A stale summary can be discarded and rebuilt
as long as the underlying `EvidenceRef` targets remain valid.

### Non-Goals

This track should not:

- turn FastCode into the authoritative store for general agent/user memory
- copy all code facts into an external memory graph
- mutate external agent session history as its primary integration mechanism
- replace current query/retrieval APIs before the context bundle path proves
  useful
- reuse third-party pruning implementation code without license review
- treat free-form dialogue summaries as the primary state carrier once typed
  turn artifacts exist

## Semantic upgrade flow

Semantic upgrade is a separate stage from basic parsing.

It currently supports:

- helper-backed resolvers where external language helpers exist
- graph-backed fallback resolvers for non-C-family languages
- explicit diagnostics and metrics on helper degradation
- repository-root aware helper execution instead of shell-cwd coupling

Registered resolver coverage includes:

- Python
- JavaScript / TypeScript
- Java
- Go
- Rust
- C#
- C / C++
- Zig
- Fortran
- Julia

Experimental SCIP language support is explicit and warning-bearing rather than
silently treated as production-complete.

## Storage architecture

Storage is split into orchestration and infrastructure:

- `store/` coordinates persistence behavior
- `store/infrastructure/` holds lower-level storage-facing code

Current backends in active use:

- SQLite and PostgreSQL-oriented runtime paths
- vector and cache stores
- projection and manifest persistence

Current design rule:

- infrastructure returns typed records and primitives where possible
- higher-level store code should not depend on API schemas or transport shells
- compatibility dict-return helpers may remain for callers, but active internal
  persistence and queue paths should reconstruct rows through explicit typed
  adapters

## Shells and entrypoints

The outer shells live in:

- `main/` CLI and composition root
- `api/` FastAPI and web entrypoints
- `mcp/` MCP server entrypoint

These packages may depend inward on the rest of the system. The inverse is not
allowed.

Current entrypoints from `fastcode/pyproject.toml`:

- `fastcode`
- `fastcode-api`
- `fastcode-mcp`
- `fastcode-web`

## Concurrency and state

The codebase has landed correctness hardening here, but read scalability remains
a release-risk area.

Landed:

- public `FastCode` entrypoints serialize repository load/index, snapshot
  pipeline, query/query-stream, cache load, delete, cleanup, and shutdown through
  a service-level `threading.RLock`
- `LoadedSnapshotArtifacts` handles and an artifact-key LRU cache exist in
  `IndexPipeline`
- `QueryPipeline.query_snapshot()` can use request-local retriever/graph handles
  without swapping singleton serving state
- regression tests cover mutation/query serialization and lower-level artifact
  handle isolation
- several blocking API and web paths are offloaded with `asyncio.to_thread`

Open concerns:

- public `FastCode.query()` and `FastCode.query_snapshot()` still hold the
  service lock for the whole query, so independent immutable snapshot reads are
  serialized at the composition root
- endpoint-level concurrency coverage under real ASGI scheduling
- real backend behavior under concurrent traffic

## Release blockers still open

The main remaining stable-release blockers are:

- install and packaging reproducibility from built artifacts
- workspace-root and layout stability in merged form
- true end-to-end incremental source and index caching
- broader FP/FCIS dataflow completion across store, query, and projection boundaries
- real backend and toolchain evidence across supported languages
- production deployment/auth documentation beyond the current
  trusted-local/proxy-auth API contract
- deployment and operations documentation

See [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md) for the maintained
release-gap list and acceptance criteria.

## Source of truth

Use this order of trust:

1. architecture tests and module-local lint rules
2. current code under `fastcode/src/fastcode/`
3. [IMPLEMENTATION_TODOS.md](./IMPLEMENTATION_TODOS.md)
4. this document
5. historical README or marketing claims

If this document and the tests disagree, the tests win.
