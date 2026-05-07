# FastCode Agent Integration Patterns

This document turns the current agent-context research into an integration
design that fits FastCode's existing package boundaries.

It is intentionally narrower than a generic "multi-agent framework" proposal.
The goal is not to build an agent platform from scratch. The goal is to make
FastCode a reliable context substrate for agents that need repository truth,
budgeted working memory, and replayable evidence.

## Scope

This design covers:

- execution patterns for research-style and coding-style agents
- how those patterns should use FastCode `L0` / `L1` / `L2` projections
- module placement inside the current layer DAG
- typed record boundaries for working memory, handoff, and activation
- why FastCode should support multiple execution modes instead of one default

Checkpoint and transactional state design is tracked separately in
[AGENT_CHECKPOINT_STATE_DESIGN.md](./AGENT_CHECKPOINT_STATE_DESIGN.md).

This design does not cover:

- training or fine-tuning models
- generic chat orchestration outside repository-aware tasks
- replacing snapshots, projections, manifests, or IR as source of truth

## Sources

This design is based on a mix of external references and local implementation
inspection:

- Anthropic, "How we built our multi-agent research system" (Jun 13, 2025):
  `https://www.anthropic.com/engineering/built-multi-agent-research-system`
- Google Research, "Towards a science of scaling agent systems: When and why
  agent systems work" (Jan 28, 2026):
  `https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/`
- Google Research, "Chain of Agents: Large language models collaborating on
  long-context tasks" (Jan 23, 2025):
  `https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/`
- Google Research, "Multi-Agent Design: Optimizing Agents with Better Prompts
  and Topologies" (ICLR 2026):
  `https://research.google/pubs/multi-agent-design-optimizing-agents-with-better-prompts-and-topologies/`
- AutoGen paper:
  `https://arxiv.org/abs/2308.08155`
- CAMEL paper:
  `https://arxiv.org/abs/2303.17760`
- MetaGPT paper:
  `https://arxiv.org/abs/2308.00352`
- More Agents Is All You Need:
  `https://arxiv.org/abs/2402.05120`
- MiroFlow code inspection in local clone under `/tmp/MiroFlow`, especially:
  `src/core/orchestrator.py`, `src/utils/summary_utils.py`,
  `src/tool/mcp_servers/searching_mcp_server.py`,
  `config/agent_prompts/main_agent_prompt_gaia.py`
- Local FastCode design notes:
  [ARCHITECTURE.md](/home/jacob/develop/FastCode/ARCHITECTURE.md),
  [AGENT_CONTEXT_DSL.md](/home/jacob/develop/FastCode/AGENT_CONTEXT_DSL.md),
  [2026-03-30-hybrid-memory-code-graph-design.md](/home/jacob/.codex/memories/2026-03-30-hybrid-memory-code-graph-design.md),
  [2026-03-30-transform-layer-projection-design.md](/home/jacob/.codex/memories/2026-03-30-transform-layer-projection-design.md),
  [2026-03-30-three-layer-json-schema-draft.md](/home/jacob/.codex/memories/2026-03-30-three-layer-json-schema-draft.md)

## Grounded Research Takeaways

### 1. Multi-agent is not a universal default

The strongest recent result is negative, not positive: multi-agent systems help
only when the task shape fits the coordination pattern.

Google's 2026 scaling study evaluated single-agent, independent, centralized,
decentralized, and hybrid systems across multiple benchmarks. Their result is
directly relevant to FastCode:

- multi-agent helps substantially on parallelizable tasks
- multi-agent degrades sequential planning tasks
- independent parallel agents amplify error badly
- centralized orchestrator-worker systems contain error better

For FastCode this means:

- do not default to "more agents"
- do not default to peer-to-peer agent debate
- use centralized delegation only when subtasks are actually separable

### 2. Research workflows benefit from context separation

Anthropic's Research system is useful evidence for breadth-first information
gathering. Their lead agent plans, spawns subagents, and receives condensed
findings back. The key claim is not "parallelism is always good." The key claim
is that separate context windows let workers explore branches and compress
results for the lead agent.

For FastCode this supports:

- delegated exploration for broad research tasks
- separate context windows for branch work
- summary-return handoff instead of raw transcript replay

It does not support:

- using many agents for tightly coupled coding loops
- making multi-agent the default for repository edits

### 3. Long-context tasks can be decomposed without shared full history

Chain-of-Agents is relevant because it frames multi-agent work as distributed
processing over long context, not as role-play. The core idea is chunk-local
processing plus staged aggregation.

For FastCode this supports:

- projection- and evidence-sliced branch work
- `L2` / `L3` expansion by partition
- handoff artifacts that summarize covered evidence instead of replaying all
  intermediate prompt text

### 4. Simpler topologies and strong prompts matter

Google's MASS paper argues that better systems often emerge from simpler design
spaces and that prompts matter heavily. AutoGen, CAMEL, and MetaGPT are useful
as examples of agent frameworks, but they should not be treated as proof that
role proliferation itself is the right design.

For FastCode this means:

- keep the execution topology small
- prefer explicit task modes over many agent persona types
- prefer typed handoff and evidence contracts over free-form agent chat

### 5. Search-agent engineering works when the lead context stays narrow

MiroFlow's local code shows a practical orchestrator-worker implementation:

- main agent is narrow and reasoning-heavy
- worker gets search, scrape, read, code, and audio tools
- hint generation and answer extraction are separate optional passes
- search includes temporal tools such as Wikipedia revision lookup and Wayback

That is useful for FastCode's research mode, but it is still mostly a
retrieval-and-summary architecture. It is not a complete long-horizon memory
system by itself.

## FastCode Design Position

FastCode should support three execution patterns and choose among them based on
task shape. The system should not collapse them into one "agent mode."

The important distinction is not model price by itself. It is authority,
statefulness, and transaction rights:

- `orchestrator/planner`:
  SOTA reasoning model at high-value control points. Owns global task state,
  acceptance contract, budget, merge decisions, user-facing synthesis, and
  fallback judgment.
- `stateful specialist`:
  good model plus specialized framework recipe. Used for researcher/debugger
  style uncertain exploration. Can receive follow-up messages on the same
  investigation thread, but cannot mutate global state directly.
- `one-shot worker`:
  cheap or medium model for a clearly specified bounded task. Treated like a
  tool call: one input contract, one typed output artifact, then closed.

This gives FastCode a protocol model instead of a vague "agent teammate" model.
Stateful specialists are not peers of the orchestrator. They are continuation
capable investigation threads with strict schemas, leases, and verifiers.

## Pattern 1: Subagents As Tools

### Shape

One lead agent owns the task and can call specialized worker agents as tools.

```text
lead agent -> worker tool call -> summarized result -> lead agent
```

### Role Authority

MiroFlow names the delegated role a worker, but for FastCode the important
distinction is contract, statefulness, and authority.

Use `one-shot worker` for the generic tool-like primitive: a clean-context agent
called by the orchestrator with a bounded input and a typed return artifact.

Use `stateful specialist` for the continuation-capable middle tier. Researcher
and debugger both belong here at the macro level because both explore
non-deterministic uncertainty. Their details must still be recipe-specific in
code.

Recommended vocabulary:

- `orchestrator/planner`:
  owns task state, acceptance contract, pattern selection, budget, and final
  synthesis
- `stateful specialist`:
  continuation-capable investigation thread for uncertain research or debugging
- `researcher recipe`:
  source discovery, citation quality, temporal scope, competing claims, and
  suggested expansions
- `debugger recipe`:
  reproduction state, suspected causes, tested hypotheses, verifier results,
  and patch risk
- `one-shot worker`:
  bounded tool-like execution for trivial or clearly specified subtasks
- `builder recipe`:
  implementation worker for a clearly specified edit plan
- `verifier recipe`:
  checker for tests, regressions, citations, consistency, or schema compliance

So the best research architecture is not "orchestrator plus generic worker."
It is:

```text
orchestrator -> stateful_researcher_thread(query, scope, evidence_policy)
orchestrator -> follow_up(thread_id, refined_question)
stateful_researcher_thread -> evidence delta + handoff artifact
```

For coding tasks, the analogous shape is different:

```text
orchestrator -> one_shot_builder(clear edit contract) -> patch artifact
orchestrator -> one_shot_verifier(acceptance contract) -> verification artifact
```

The orchestrator should keep the durable task state. Specialists and workers
should be replaceable executors with strict input and output records.
Specialists can be resumed by thread ID. One-shot workers cannot.

### Best Fit

- breadth-first research
- multi-source evidence gathering
- temporal or archival lookup
- multimodal or tool-heavy exploration
- independent subquestions with low cross-dependency

### Why

This matches the strongest evidence from Anthropic's research system and
Google's centralized architecture results.

### Risks

- worker summaries can omit critical evidence
- weak decomposition causes duplicated search
- too many workers increase cost and coordination overhead
- coding tasks usually do not decompose cleanly enough to justify this mode

### FastCode Contract

Workers must return typed artifacts, not just prose. Researcher workers need
the strictest evidence contract:

- `claim_set`
- `evidence_refs`
- `omitted_refs`
- `uncertainty_flags`
- `open_questions`
- `raw_log_refs`
- `cost`
- `freshness`

Researcher artifacts should also include:

- `candidate_answers`
- `conflict_edges`
- `source_quality`
- `temporal_scope`
- `suggested_expansions`

Stateful specialists must also obey a lease:

- `max_turns`
- `max_cost`
- `must_return_delta`
- `must_cite_evidence`
- `no_write_rights`
- `close_if_no_new_evidence`

### FastCode Usage Rule

Use this pattern only when subproblems are independently explorable. For
repository tasks, typical examples are:

- "find every implementation path for feature flag X"
- "collect all modules that touch auth token refresh"
- "verify historical behavior across snapshots"

Do not use this as the default edit loop.

## Pattern 2: Main Agent Fork And Summary Back

### Shape

The main agent forks a branch execution with a clean context, explores within
that branch, then returns a distilled handoff artifact.

```text
main state -> fork -> branch exploration -> distilled handoff -> main state
```

### Best Fit

- long-turn research
- debugger-style work
- planner or architect tasks
- ambiguous problem framing
- multi-hypothesis exploration
- tasks that would pollute one shared prompt window

### Why

This pattern matches the two-surface memory model already described in
[ARCHITECTURE.md](/home/jacob/develop/FastCode/ARCHITECTURE.md) and
[AGENT_CONTEXT_DSL.md](/home/jacob/develop/FastCode/AGENT_CONTEXT_DSL.md):

- historical truth stays append-only, complete, external, and restorable
- working memory stays small, rewritten, and cache-friendly

It is also compatible with Chain-of-Agents style partitioned processing, but it
does not require many concurrent workers.

### Risks

- branch summaries can distort what happened
- rollback is impossible unless branch state is externalized
- indiscriminate rewriting can discard useful low-level evidence
- this adds overhead for small tasks

### FastCode Contract

Each fork must emit a `HandoffArtifact` with:

- `fork_id`
- `parent_turn_id`
- `task_fingerprint`
- `snapshot_ref`
- `covered_evidence_refs`
- `distillation_refs`
- `claim_set`
- `risk_state`
- `rejected_hypotheses`
- `next_actions`
- `reopen_conditions`
- `raw_journal_ref`

### FastCode Usage Rule

This should be the default for long-horizon researcher, debugger, and planner
tasks.

For debugging, prefer `fork` or `inline` when the task is tightly sequential.
Use `delegate` only when the debug search can be split into independent
hypotheses or environments.

## Pattern 3: One Agent Plan, Clear Context, Execute

### Shape

A single agent builds a plan, compiles a bounded working context, and executes
without delegation.

```text
plan -> compile context -> execute -> verify
```

### Best Fit

- atomic coding tasks
- narrow edits
- straightforward explanation requests
- single-path retrieval
- low-ambiguity repository questions

### Why

Google's scaling study shows that multi-agent coordination hurts sequential
tasks. Most coding tasks are sequential, verification-heavy, and coupled to one
shared state. They are usually a poor fit for heavy delegation.

### Risks

- context grows too much over long runs
- early wrong assumptions can persist
- one agent may miss alternative branches if the task is under-specified

### FastCode Usage Rule

This is the default for the normal coding loop. Rewrite only the compiled
working-memory view, not the historical journal.

## Pattern Selection Policy

FastCode should choose pattern by task properties, not by user-visible "agent
branding."

## Task Properties

The selector should score:

- `decomposability`
- `parallel_branch_value`
- `sequential_dependency`
- `tool_count`
- `repository_scope_size`
- `freshness_risk`
- `verification_cost`
- `expected_turn_count`
- `need_for_historical_replay`

## Default Decision Table

- `inline`:
  low decomposability, high sequential dependency, low expected turn count
- `delegate`:
  high decomposability, medium or high branch value, bounded subtask contracts
- `fork`:
  medium decomposability, high ambiguity, high expected turn count, high prompt
  pollution risk

## Repository-Specific Guidance

- code editing:
  default to `inline`
- repo reconnaissance:
  prefer `delegate`
- architectural research:
  prefer `fork`
- historical diff or provenance tracing:
  use `delegate` or `fork`, depending on branch coupling

## Protocols

FastCode should model two communication protocols. They share record types, but
they do not share lifecycle semantics.

### Stateful Specialist Protocol

Used for researcher/debugger-style uncertain exploration.

```text
START_THREAD -> OBSERVE | EXPAND | CHALLENGE | REFINE -> HANDOFF -> CLOSE
```

Rules:

- the orchestrator may send follow-up messages to the same thread
- every follow-up must reference a `thread_id` and expected delta schema
- the specialist returns deltas, not whole-task rewrites
- the specialist cannot mutate global state directly
- thread continuation is bounded by lease fields
- code verifies every returned evidence ref and artifact ref

Minimal records:

- `SpecialistThreadStart`
- `SpecialistFollowup`
- `SpecialistDelta`
- `SpecialistHandoff`
- `SpecialistLease`

### One-Shot Worker Protocol

Used for cheap bounded tasks.

```text
WORK_ORDER -> ARTIFACT | NEEDS_CLARIFICATION | FAILED
```

Rules:

- no follow-up is assumed
- worker receives a clear goal and narrow context pack
- worker returns one typed artifact or a typed failure
- worker output must pass schema validation before the orchestrator sees it
- worker cannot ask broad strategic questions

Minimal records:

- `WorkOrder`
- `WorkerArtifact`
- `WorkerFailure`
- `ClarificationRequest`

### Shared Agent Records

The shared protocol vocabulary should be explicit:

- `AgentProfile`:
  role, model tier, tool allowlist, statefulness, max turns, max cost, write
  rights, supported recipes
- `ContextPack`:
  `L0` refs, `L1` refs, `L2` refs, raw expansion refs, token budget, freshness
- `AcceptanceContract`:
  required evidence classes, required verifiers, allowed writes/tools,
  abstain/ask thresholds
- `Artifact`:
  claims, evidence refs, uncertainty, raw log refs, patch refs, verifier refs
- `TransactionProposal`:
  intent, target files, patch refs, evidence refs, expected effects, rollback
  refs
- `ValidationResult`:
  schema status, evidence status, verifier status, risk status, violations
- `CommitDecision`:
  `commit`, `reject`, `continue`, `ask_user`, or `fallback_to_orchestrator`

### Recipe Split

`researcher` and `debugger` share the `StatefulSpecialistProtocol`, but they
should not share one schema.

Research recipe fields:

- `source_plan`
- `candidate_claims`
- `citation_quality`
- `temporal_scope`
- `conflicting_sources`
- `suggested_expansions`

Debug recipe fields:

- `repro_state`
- `suspected_causes`
- `tested_hypotheses`
- `instrumentation_refs`
- `verifier_results`
- `patch_risk`

Builder and verifier recipes should be one-shot by default unless the external
orchestrator explicitly upgrades them to a stateful specialist thread.

## Memory Model

FastCode should keep the memory split already established in
[ARCHITECTURE.md](/home/jacob/develop/FastCode/ARCHITECTURE.md):

- historical truth:
  append-only, complete, external, restorable
- working memory:
  small, recent, relevant, cache-friendly

The operational consequence is important:

- tool results, verifier outputs, branch journals, and raw evidence belong to
  the historical layer
- `L0`, selected `L1`, current hypotheses, acceptance contract, and next
  actions belong to the working-memory layer
- "rewrite" means recompiling working memory from typed records, not mutating
  the history log

## Projection Use By Pattern

FastCode's projection system already has the right levels:

- `L0`: task orientation and compact repo frame
- `L1`: navigation, clusters, relationships, decision context
- `L2`: cited evidence anchors
- `L3`: raw expansion targets and deep reads

Pattern use should differ:

- `inline`:
  inject stable `L0`, selective `L1`, then expand to `L2` only on demand
- `delegate`:
  give workers narrow `L0` plus branch-specific `L1`; require `L2` refs back
- `fork`:
  give branch a compact parent handoff plus scoped `L0` / `L1`; store branch
  journal externally and return a distilled handoff

## Module Placement

This work should not create a new top-level package. It should fit the existing
layer DAG.

FastCode also should not become the full agent runtime. It is a repository
intelligence and context module. The external orchestrator project owns model
routing, agent lifecycle, retries, thread scheduling, and transaction flow.
FastCode owns repository-grounded context and evidence contracts.

## `retrieval/core/`

Owns pure logic and pure records:

- `agent_context_records.py`
  - `EvidenceRef`
  - `ContextBundle`
  - `DistillationRecord`
  - `ActivationRecord`
  - `TurnIntent`
  - `TurnPlan`
  - `ToolObservation`
  - `ObservationJournalRef`
  - `BeliefState`
  - `RiskState`
  - `AcceptanceContract`
  - `RejectedHypothesis`
  - `WorkingSet`
  - `HandoffArtifact`
  - `AgentProfile`
  - `SpecialistLease`
  - `WorkOrder`
  - `WorkerArtifact`
  - `TransactionProposal`
  - `ValidationResult`
  - `CommitDecision`
- `agent_pattern_selection.py`
  - task-shape scoring
  - pattern choice logic
- `agent_bundle_compiler.py`
  - evidence scoring
  - token budget allocation
  - stable prefix selection
  - `L0` / `L1` / `L2` assembly
- `agent_distillation.py`
  - reuse and fingerprint rules
  - distillation eligibility
- `agent_handoff.py`
  - fork handoff assembly
  - worker return normalization
- `agent_protocol.py`
  - stateful specialist protocol validation
  - one-shot worker protocol validation
- `agent_transaction.py`
  - deterministic transaction gates
  - proposal validation
  - commit/reject decision helpers

No Pydantic, no shell IO, no environment reads.

## `query/`

Owns orchestration:

- call pattern selector
- trigger retrieval or expansion
- decide `inline` vs `delegate` vs `fork`
- compile prompt-facing working memory
- record activation after completion
- route handoff artifacts into resume flows

Suggested files:

- `query/agent_runtime.py`
- `query/agent_modes.py`
- `query/activation.py`

FastCode's `query/` package should expose module-level operations for an
external orchestrator, not own the full orchestration runtime:

- `build_context_pack()`
- `expand_evidence_ref()`
- `compile_working_memory()`
- `validate_handoff_refs()`
- `record_activation()`

## `retrieval/`

Owns imperative retrieval-and-tool orchestration:

- bind worker calls to existing retrieval tools
- support iterative expansion and verification
- keep imperative tool loop outside `retrieval/core/`

Suggested files:

- `retrieval/agent_delegate.py`
- `retrieval/agent_fork.py`

## `indexing/`

Remains the owner of projection generation:

- `indexing/projection.py`
- `indexing/projection_transform.py`

The new integration work should consume projections, not redefine them.

## `store/`

Owns typed persistence and explicit serializers for:

- `ContextBundleRecord`
- `DistillationRecord`
- `ActivationRecord`
- `HandoffArtifactRecord`
- `ObservationJournalRecord`

Suggested files:

- `store/context.py`
- `store/context_records.py`

Raw code facts remain in snapshots, projection artifacts, manifests, and IR
stores. Context records persist derived state only.

## `api/` and `mcp/`

Shell adapters only.

Suggested operations:

- build context bundle
- expand evidence ref
- create fork handoff
- resume from handoff
- record activation outcome
- validate worker artifact
- validate transaction proposal

These shells must not become the composition root for context logic.

## External Orchestrator Boundary

The separate orchestration project owns:

- agent thread lifecycle
- model selection and routing
- retries and backoff
- human clarification routing
- transaction scheduling
- cross-project memory and task state
- monorepo-level workflow composition

FastCode owns:

- repository snapshots and projection refs
- context pack compilation
- evidence expansion
- artifact and evidence validation
- activation recording
- deterministic gates over repository-grounded operations

If these projects later consolidate into a monorepo, keep the package boundary
real:

```text
fastcode/      repository intelligence and context substrate
scatter/       orchestration runtime and agent protocols
shared/        stable schemas only when duplication becomes harmful
```

Do not move agent lifecycle into FastCode just because the repositories become
physically colocated.

## Stable Interfaces

Regardless of execution pattern, every agent-facing result should normalize to
the same shape:

- `claim_set`
- `evidence_refs`
- `uncertainty_flags`
- `rejected_hypotheses`
- `open_questions`
- `next_actions`
- `raw_log_refs`

This is what lets FastCode switch execution patterns without changing the rest
of the runtime.

## Pattern-Specific Contracts

### Delegate Contract

Inputs:

- task text
- snapshot/artifact handle
- scoped `L0`
- scoped `L1`
- acceptance contract
- token budget

Outputs:

- worker result artifact
- evidence refs
- covered refs
- omitted refs
- uncertainty flags

The delegate contract is one-shot unless it is explicitly promoted to
`StatefulSpecialistProtocol`.

### Fork Contract

Inputs:

- parent handoff artifact
- branch goal
- snapshot/artifact handle
- scoped projection refs
- budget

Outputs:

- branch handoff artifact
- raw journal ref
- reopen conditions

The fork contract is the preferred shape for long ambiguous investigation when
the branch needs a clean context but should still preserve raw history outside
the prompt.

### Inline Contract

Inputs:

- task text
- snapshot/artifact handle
- compiled working set
- verifier plan

Outputs:

- final answer or edit recommendation
- activation record

The inline contract remains the default for normal coding tasks and tightly
coupled debug execution.

## Deterministic Transaction Layer

The orchestrator should reason over exceptions, not routine mechanics.

Move repeatable control operations into deterministic FastCode logic:

- pattern selection pre-checks
- context pack assembly
- schema validation
- evidence ref existence checks
- stale snapshot detection
- worker budget and timeout enforcement
- patch proposal validation
- rollback and snapshot ref validation
- verifier/test execution hooks
- artifact persistence
- merge/reject gates when rules are clear

Keep fallback judgment with the orchestrator:

- ambiguous task interpretation
- conflicting evidence judgment
- deciding whether missing evidence is acceptable
- choosing between valid competing plans
- asking the user for scope clarification
- overriding a deterministic reject with explicit rationale

This replaces ad hoc orchestrator actions with code over time. The target is
not to remove the orchestrator. The target is to shrink the manual control
surface to cases where reasoning is actually needed.

## Cache And Fingerprinting

Bundle and distillation reuse should be keyed by explicit fingerprints, not raw
prompt text.

Minimum fingerprint fields:

- task fingerprint
- normalized task mode
- snapshot ID
- artifact key
- projection algorithm version
- embedding fingerprint
- distillation prompt fingerprint
- token budget

This is necessary so that:

- `L0` and stable `L1` can become cache-friendly prefixes
- invalidation is deterministic when snapshots or projection logic change
- delegation and fork handoffs remain replayable

## Why Not One Universal Multi-Agent Runtime

The main engineering reason is that task shapes differ too much.

The research reason is stronger:

- Google's 2026 evaluation says architecture must align to decomposability and
  tool burden
- Anthropic's evidence supports multi-agent breadth-first research, not general
  coding
- MiroFlow shows practical value in narrow orchestrator + search worker splits
  but does not justify applying that pattern everywhere

FastCode should therefore expose multiple modes over one context substrate,
instead of hard-coding one fashionable agent topology.

## Rollout Plan

### Phase 1

Land typed records and bundle compilation for `inline` mode only:

- `EvidenceRef`
- `ContextBundle`
- `ActivationRecord`
- `AgentProfile`
- `WorkOrder`
- deterministic `L0` / `L1` / `L2` compiler

### Phase 2

Add one-shot worker mode with strict return contracts:

- scoped worker inputs
- worker result normalization
- activation telemetry
- worker artifact validation

### Phase 3

Add stateful specialist and fork mode:

- `HandoffArtifact`
- branch journal persistence
- resume and rollback hooks
- specialist leases and follow-up validation

### Phase 4

Add deterministic transaction gates:

- transaction proposal validation
- evidence and verifier gates
- commit/reject helpers

### Phase 5

Use activation history as evaluation telemetry only. Do not let it change
ranking by default until there is enough data.

## Bottom Line

FastCode should not become "a multi-agent framework."

It should become a repository-aware context system with:

- one authoritative historical layer
- one compiled working-memory layer
- three execution patterns selected by task shape
- one-shot workers for bounded subtasks
- stateful specialists for follow-up investigation threads
- typed handoff and evidence contracts
- deterministic transaction gates for repeatable operations
- stable module placement inside the current architecture

That is a simpler and more defensible design than forcing every task through
subagents, or treating prompt rewrite as the only answer to long-turn work.
