# FastCode Agent Context DSL

This document defines a model-facing DSL for compact agent context. It is not
the authoritative storage format. The source of truth remains typed FastCode
records, snapshots, projections, manifests, and stores.

The DSL exists for one purpose: render enough structured context into the model
window with fewer tokens and less ambiguity than JSON or free prose.

FCX is the prompt-facing working-memory surface, not the historical truth
surface. Historical truth remains in typed FastCode records, append-only
observation journals, snapshots, projections, and external artifacts that can
be restored by reference.

## Design Goals

- Keep `L0` and `L1` stable enough for prompt caching.
- Let agents request `L2` evidence and raw/deep context by reference.
- Preserve citations and expansion handles through every rewrite.
- Represent turn state, hypotheses, tool observations, and verifier state.
- Represent uncertainty policy, acceptance contract, and rejected hypotheses.
- Be cheap for the model to read and cheap for FastCode to parse.
- Avoid repeated JSON key names, braces, quotes, and nested payloads in prompts.
- Support a two-surface model:
  append-only historical journal plus replaceable compiled working memory.

## Non-Goals

- Do not use this as a database format.
- Do not expose raw unrestricted tool output as DSL.
- Do not let the model overwrite authoritative context state directly.
- Do not put large source chunks inline when a stable reference can be used.

## Name

The working name is `FCX`, for FastCode Context.

Version marker:

```text
@fcx v=1
```

## Core Shape

The DSL is a line-oriented record stream.

Each line has:

```text
TAG id? key=value key=value | free text
```

Rules:

- `TAG` is one to three uppercase letters.
- `id` is optional and has no `key=`.
- key/value fields are space-separated.
- quoted values use JSON string escaping only when needed.
- unquoted values must not contain spaces.
- `|` starts a free-text tail.
- unknown tags are ignored by old parsers unless the header requires them.
- record order is meaningful inside a section but references are by ID.

Minimal grammar:

```text
stream      := header line*
header      := "@fcx" field*
line        := tag id? field* tail?
tag         := /[A-Z][A-Z0-9]{0,2}/ | "L0" | "L1" | "L2" | "L3" | "ERR" | "END"
id          := /[a-z][a-z0-9_:-]*/
field       := key "=" value
key         := /[a-z][a-z0-9_-]*/
value       := bare | quoted | list
bare        := /[^ \t\n|]+/
quoted      := JSON string
list        := bare ("," bare)*
tail        := "|" text
```

Example:

```text
@fcx v=1 sid=sess42 turn=7 snap=snp9 art=ak3 budget=1800
L0 p=proj_a tok=112 | Query serving uses snapshot artifacts plus retriever state.
L1 p=proj_a nav=e12,e14,e31 xref=g4,g9 | Important areas: query pipeline, cache store, snapshot loader.
I i1 kind=debug out=patch q="why snapshot query reloads artifacts"
C out=patch need=repo_evidence,pytest allow=search_codebase,read_file,pytest done=verified_patch fail=ambiguous_scope
H h1 p=.72 s=e12,e31 c=e08 | artifact handle cache missing
H h2 p=.18 s=e14 c=- | stale singleton state
R eg=1 cf=1 fr=0 ra=0 xr=2 vs=pending act=verify | Patch should wait for verifier result.
N act=expand,verify,abstain expand=e31 verify=t_pytest ask=-
```

## Two-Surface Semantics

FastCode should keep two different things distinct:

- historical truth:
  append-only, complete, external, and restorable
- cognitive working memory:
  small, recent, relevant, and cache-friendly

In FCX terms:

- `O`, `V`, `A`, and `ERR` belong to the append-only journal surface
- `L0`, `L1`, `L2`, `I`, `P`, `N`, `H`, `Q`, `F`, `C`, `R`, `W`, `D`, and
  `HO` belong to the compiled working-memory surface
- `X` belongs to the append-only rejected-hypothesis ledger surface, though a
  small recent slice may also be injected into working memory
- `E`, `G`, and `L3` bridge the two by pointing from working memory back to
  authoritative evidence

The compiler may include a small recent slice of journal records in the prompt.
It should not inline the full journal when references are enough.

## Record Tags

### Header

`@fcx` declares stream metadata.

Fields:

- `v`: DSL version.
- `sid`: session ID.
- `turn`: turn number.
- `repo`: repository name when single-repo.
- `snap`: snapshot ID.
- `art`: artifact key.
- `budget`: target token budget for this rendered context.
- `fp`: context compiler fingerprint.
- `mode`: `stable`, `turn`, `tool`, `journal`, or `debug`.

Example:

```text
@fcx v=1 sid=s7 turn=3 repo=FastCode snap=snp_abc art=ak_7 budget=2200 fp=ctx_v1 mode=turn
```

### Projection Records

`L0` is the compact orientation layer.

Fields:

- `p`: projection ID.
- `tok`: estimated token count.
- `refs`: optional comma-separated evidence refs.

Example:

```text
L0 p=proj_f3 tok=91 refs=e1,e2 | FastCode query flow is retrieval, optional semantic escalation, answer generation, and session persistence.
```

`L1` is navigation and relationship context.

Fields:

- `p`: projection ID.
- `nav`: recommended expansion refs.
- `xref`: graph or relation refs.
- `tok`: estimated token count.

Example:

```text
L1 p=proj_f3 nav=e12,e18,e22 xref=g3,g8 tok=240 | Query state crosses handler, hybrid retriever, iterative agent, cache store.
```

`L2` is cited evidence summary. `L2` records may be inline only when compact.
Large chunks should be referenced through `E`.

Example:

```text
L2 e12 path=fastcode/src/fastcode/query/handler.py lines=237-420 | query() retrieves, escalates, generates answer, then saves dialogue turn.
```

`L3` is not injected by default. It names raw/deep expansion targets that tools
can fetch.

Example:

```text
L3 e12 op=read_source max=220 lines=237-420
```

## Turn Records

`I` records the turn intent.

Fields:

- `kind`: `debug`, `design`, `explain`, `edit`, `review`, `research`.
- `out`: desired outcome, such as `answer`, `patch`, `plan`, `tests`.
- `q`: normalized query. Quote when it contains spaces.
- `scope`: optional scope ID.

Example:

```text
I i1 kind=design out=spec q="agent context rewrite DSL" scope=s_fastcode
```

`P` records the current turn plan.

Fields:

- `step`: current step number.
- `act`: intended action.
- `why`: short reason code.
- `stop`: stop condition.
- `budget`: remaining budget.

Example:

```text
P p1 step=2 act=expand(e12) why=missing_boundary stop=verified budget=900
```

`N` records allowed next actions.

Fields:

- `act`: allowed high-level actions.
- `expand`: refs that may be expanded.
- `tool`: tools that may be called.
- `verify`: verifier IDs.
- `ask`: `-` or a question ID.

Example:

```text
N act=expand,verify,answer,abstain expand=e12,e18 tool=search_symbol,read_file verify=t_pyright,t_pytest ask=-
```

## Evidence Records

`E` is an evidence reference. It points to authoritative FastCode data.

Fields:

- `kind`: `file`, `symbol`, `range`, `path`, `cluster`, `turn`, `tool`,
  `projection`, `verifier`.
- `repo`: repository name.
- `snap`: snapshot ID.
- `path`: repository-relative path.
- `sym`: symbol ID.
- `lines`: source range.
- `p`: projection ID.
- `chunk`: projection chunk ID.
- `fresh`: `ok`, `stale`, `unknown`.
- `tok`: estimated expansion token count.

Example:

```text
E e12 kind=range repo=FastCode snap=snp_abc path=fastcode/src/fastcode/query/handler.py lines=237-420 fresh=ok tok=620
```

`G` is a graph evidence reference.

Fields:

- `kind`: `call`, `dependency`, `inheritance`, `reference`, `containment`,
  `impact`, `steiner`.
- `from`: source node.
- `to`: target node when applicable.
- `hops`: hop count.
- `refs`: supporting evidence refs.

Example:

```text
G g3 kind=impact from=QueryPipeline.query hops=2 refs=e12,e22,e31
```

## Belief Records

`H` is a hypothesis. It must have support or an explicit reason why support is
missing.

Fields:

- `p`: probability-like confidence in `[0,1]`. This is a decision signal, not a
  calibrated probability.
- `s`: supporting refs.
- `c`: conflicting refs, or `-`.
- `state`: `open`, `favored`, `rejected`, `verified`, `blocked`.

Example:

```text
H h1 p=.68 state=favored s=e12,g3 c=e08 | context rewrite should live after tool observation normalization
```

`Q` is an unresolved question.

Fields:

- `for`: hypothesis or plan ID.
- `need`: `evidence`, `decision`, `user`, `tool`, or `verify`.

Example:

```text
Q q1 for=h1 need=verify | Need test proving source refs survive summary rewrite.
```

`F` is an accepted fact.

Fields:

- `refs`: supporting refs.
- `scope`: `turn`, `session`, `repo`, or `snapshot`.

Example:

```text
F f1 scope=session refs=e12,e18 | Query handler is the outer session-aware orchestrator.
```

`X` records a rejected hypothesis ledger entry.

Fields:

- `from`: rejected hypothesis ID.
- `by`: evidence or verifier refs that killed it.
- `why`: rejection reason code.
- `reopen`: condition code or `-`.

Example:

```text
X x1 from=h2 by=v1,e91 why=type_error reopen=new_snapshot
```

## Decision Control Records

`C` records the current acceptance contract.

Fields:

- `out`: requested outcome.
- `need`: required evidence/verifier classes.
- `allow`: allowed tools or write scopes.
- `done`: done-condition code.
- `fail`: ask/abstain trigger.

Example:

```text
C out=answer need=repo_evidence,citation allow=search_codebase,expand_context_ref done=cited_answer fail=missing_primary_support
```

`R` records the current risk state.

Fields:

- `eg`: evidence gap in `[0,3]`.
- `cf`: conflict level in `[0,3]`.
- `fr`: freshness risk in `[0,3]`.
- `ra`: requirement ambiguity in `[0,3]`.
- `xr`: execution risk in `[0,3]`.
- `vs`: `clean`, `pending`, `mixed`, `failed`, or `blocked`.
- `act`: preferred next action.

Example:

```text
R eg=2 cf=1 fr=0 ra=2 xr=2 vs=pending act=verify | Missing verification and intent ambiguity still block commit.
```

## Tool Observation Records

`O` records a normalized tool observation.

`O`, `V`, and `ERR` are journal records. Once written to typed state they are
append-only; later compilers may omit them from the prompt, but should not
rewrite them in place.

Fields:

- `tool`: tool name.
- `ok`: `1` or `0`.
- `refs`: emitted evidence refs.
- `cost`: token or operation cost estimate.
- `fresh`: `ok`, `stale`, `unknown`.
- `warn`: warning code or `-`.

Example:

```text
O o1 tool=search_codebase ok=1 refs=e41,e42 cost=36 fresh=ok warn=-
```

`V` records verifier output.

Fields:

- `tool`: verifier name.
- `ok`: `1`, `0`, or `?`.
- `refs`: evidence refs.
- `kills`: rejected hypotheses.
- `supports`: supported hypotheses.

Example:

```text
V v1 tool=pyright ok=0 refs=e91 kills=h2 supports=- | type error in observation adapter
```

## Working Set Records

`W` declares what must be preserved in the next prompt.

`W`, `D`, `H`, `Q`, `F`, `C`, `R`, `P`, `N`, `HO`, and the
`L0`/`L1`/`L2` projection lines are compiled working-memory records. They may
be regenerated on every turn from the same typed inputs.

Fields:

- `keep`: refs or record IDs.
- `drop`: refs or record IDs.
- `protect`: refs or record IDs that cannot be summarized away.
- `reason`: short reason code.

Example:

```text
W keep=h1,e12,g3 drop=h2 protect=f1,e12 reason=active_design
```

`D` records a distillation.

Fields:

- `src`: covered refs.
- `omit`: omitted refs or `-`.
- `fp`: model/prompt fingerprint.
- `tok`: token count.
- `expand`: refs that restore detail.

Example:

```text
D d1 src=e12,e18 omit=- fp=dist_v1 tok=72 expand=e12,e18 | Query/session state is orchestrated in handler; iteration state is prompt-heavy today.
```

`A` records activation feedback.

Fields:

- `used`: refs consumed by the agent.
- `good`: useful refs.
- `bad`: misleading refs.
- `out`: `answer`, `patch`, `ask`, `abstain`, `fail`.

Example:

```text
A a1 used=e12,e18,d1 good=e12,d1 bad=e08 out=patch
```

`HO` records a handoff artifact reference for reset, delegation, or resume.

Fields:

- `art`: handoff artifact key.
- `for`: `reset`, `delegate`, `resume`, or `rollback`.
- `keep`: refs or record IDs carried forward.
- `drop`: refs or record IDs intentionally excluded, or `-`.

Example:

```text
HO art=hf12 for=reset keep=f9,h1,q1 drop=o17,o18 | Clean-context restart around verified cache-key bug only.
```

## Control Records

`ERR` reports malformed or unusable context.

Example:

```text
ERR code=stale_ref ref=e12 action=expand_required
```

`END` closes the stream and may include a checksum over canonical typed input
records.

Example:

```text
END refs=18 tok=1740 sum=sha1:7bd83c
```

## Model-Writable Subset

Most FCX records are compiler-owned. The model may propose only request or
rewrite records. FastCode validates those records and then updates typed state
itself.

`REQ` asks for an allowed action.

Fields:

- `act`: `expand`, `search`, `verify`, `ask`, `branch`, `reset`, `answer`,
  `abstain`.
- `ref`: target evidence/hypothesis/question ref, or `-`.
- `depth`: `L2`, `L3`, `raw`, or `-`.
- `why`: short reason code.
- `max`: max expansion token budget.

Example:

```text
REQ r1 act=expand ref=e12 depth=L3 why=need_impl max=700
```

`RWP` is a rewrite proposal. It never directly replaces canonical facts.

Fields:

- `target`: `working`, `hypothesis`, `fact`, `distill`, or `plan`.
- `src`: source refs covered by the proposal.
- `replace`: target record ID, or `-`.
- `conf`: model confidence as a weak signal.

Example:

```text
RWP rw1 target=hypothesis src=e12,g3 replace=h1 conf=.62 | Artifact handle caching, not answer generation, is the likely bottleneck.
```

`SEL` selects among existing possibilities.

Fields:

- `pick`: selected hypothesis or plan.
- `drop`: rejected IDs, or `-`.
- `need`: remaining need, or `-`.

Example:

```text
SEL pick=h1 drop=h2 need=verify
```

Compiler rules:

- reject model-emitted `F`, `E`, `G`, `O`, `V`, `A`, `D`, `C`, `R`, `X`, or
  `HO` as authoritative updates
- accept `REQ` only when it matches the latest `N` allowed-action record
- accept `RWP` only as a candidate summary or hypothesis update
- require cited `src` refs for every `RWP`
- turn `SEL` into state only after verifier and policy checks pass

## Model-Facing Sections

A normal agent prompt should use three FCX blocks:

```text
<fcx:stable>
@fcx v=1 mode=stable ...
L0 ...
L1 ...
F ...
C ...
</fcx:stable>

<fcx:turn>
@fcx v=1 mode=turn ...
I ...
R ...
P ...
H ...
Q ...
W ...
N ...
</fcx:turn>

<fcx:obs>
@fcx v=1 mode=tool ...
O ...
V ...
END ...
</fcx:obs>
```

The `stable` block should change rarely and should contain `L0`, selected `L1`,
accepted facts, the acceptance contract, and protected constraints. The `turn`
block changes each model turn. It should carry the current risk state,
hypotheses, unresolved questions, and the current working-set directive. The
`obs` block changes after tool execution and is usually small. It is only a
recent relevant slice of the append-only journal, not the full journal.

When snapshot, task, and protected constraints are unchanged, the `stable`
block should remain byte-stable when possible to maximize prompt/KV cache hits.

## Token Strategy

The DSL saves tokens by:

- replacing repeated JSON keys with short tags
- replacing nested structures with refs
- using comma-separated ID lists
- making stable prefixes cache-friendly
- pushing raw source behind `L3` expansion tools
- using small controlled vocabularies for states and actions

Do not over-optimize with unreadable character soup. The model must reliably
understand the context. A short line protocol is a better tradeoff than dense
single-character encodings.

## JSON Comparison

Minified JSON is still useful for APIs and tool payloads. FCX is for prompt
rendering.

JSON weaknesses in the model window:

- repeated keys consume tokens in every object
- nested punctuation adds visual and token noise
- object order is not naturally used as a prompt-reading path
- raw payloads tempt the system to inline too much evidence
- model-generated JSON often fails because one quote or comma breaks parsing

FCX weaknesses:

- it needs a custom parser
- it is less self-describing than JSON
- schema evolution must be explicit through versioned tags
- compact fields can become cryptic if the vocabulary grows without discipline

Decision rule:

- use typed dataclasses internally
- use JSON at API, persistence, and external tool boundaries when appropriate
- use FCX only for model-facing context and model-emitted requests
- benchmark token count and task accuracy before making FCX the default

## Expansion Policy

`L0` and selected `L1` records may be injected in the stable block. `L2` records
may be injected only when they are compact and directly relevant. `L3` is a
request target, not a default prompt section.

Expansion flow:

1. Compiler renders `L0`, `L1`, active hypotheses, and allowed `N` actions.
2. Model emits `REQ act=expand ref=e12 depth=L3`.
3. FastCode checks that `e12` is fresh and expansion is allowed.
4. Explore tool fetches source/projection/graph evidence.
5. Tool result becomes `O` plus new `E`, `G`, `L2`, or `L3` records.
6. Compiler rewrites `H`, `Q`, `W`, and `N` for the next turn.

This keeps the model from dragging raw source into every prompt while still
making deep context reachable.

## Parsing Rules

The parser should be strict by default:

- reject duplicate IDs in one stream
- reject references to missing required IDs
- reject `H` records without support unless `state=open` and a `Q` exists
- reject `X` records without killer refs
- reject `D` records without expansion refs
- reject `HO` records without an artifact key and target mode
- reject stale `E` refs unless the stream includes `ERR` or `N expand=...`
- reject missing `END` in persisted DSL snapshots

The model-facing renderer may be permissive for display, but persisted
turn-journal reconstruction should use the strict parser.

## FastCode Mapping

Typed core records compile to FCX as follows:

- `ProjectionBuildResult.l0` -> `L0`
- `ProjectionBuildResult.l1` -> `L1`
- projection chunks -> `L2` and `E`
- source ranges and symbols -> `E`
- graph tool results -> `G`
- `TurnIntent` -> `I`
- `TurnPlan` -> `P` and `N`
- `ToolObservation` -> `O`
- verifier observations -> `V`
- `BeliefState` -> `H`, `Q`, `F`, and `X`
- `AcceptanceContract` -> `C`
- `RiskState` -> `R`
- `WorkingSet` -> `W`
- `DistillationRecord` -> `D`
- `ActivationRecord` -> `A`
- `HandoffArtifact` -> `HO`

## Example: Journal To Working Memory

Raw observations stay in the journal:

```text
<fcx:journal>
@fcx v=1 sid=s88 turn=9 repo=FastCode snap=snp_b art=ak_9 budget=500 fp=ctx_v1 mode=journal
O o41 tool=grep ok=1 refs=e41 cost=12 fresh=ok warn=- | cache.py reuses projection_key across snapshot changes
O o42 tool=pytest ok=0 refs=e42 cost=84 fresh=ok warn=- | test_cache_invalidation fails when snapshot_id changes
V v9 tool=pytest ok=0 refs=e42 kills=h2 supports=h1 | failing test supports cache-key bug
END refs=3 tok=126 sum=sha1:example_journal
</fcx:journal>
```

Compiled working memory is then rewritten from those records:

```text
<fcx:turn>
@fcx v=1 sid=s88 turn=9 repo=FastCode snap=snp_b art=ak_9 budget=700 fp=ctx_v1 mode=turn
D d12 src=o41,o42,v9 omit=- fp=dist_v1 tok=27 expand=e41,e42 | Cache invalidation fails when snapshot_id changes but projection_key is reused.
C out=patch need=repo_evidence,pytest allow=read_file,pytest done=verified_patch fail=missing_repro
F f9 scope=turn refs=d12,e42 | projection_key currently omits snapshot_id.
H h1 p=.82 state=favored s=f9,v9 c=- | Fix cache key composition before further retriever tuning.
R eg=0 cf=0 fr=0 ra=0 xr=2 vs=failed act=edit | Root cause is clear but verifier still blocks completion.
W keep=f9,h1,d12 drop=- protect=f9,e42 reason=active_debug
END refs=4 tok=168 sum=sha1:example_turn
</fcx:turn>
```

The journal remains restorable. The working-memory view is what gets rewritten.

## Example: Design Turn

```text
<fcx:stable>
@fcx v=1 sid=s77 turn=4 repo=FastCode snap=snp_9 art=ak_4 budget=2600 fp=ctx_v1 mode=stable
L0 p=proj_q tok=84 refs=e1,e2 | FastCode agent context should use projections as orientation and tools for deep evidence.
L1 p=proj_q nav=e12,e18,g3 xref=g3,g7 tok=211 | Main surfaces: query handler, iterative agent, agent tools, cache store, projection transformer.
C out=spec need=repo_evidence,citations allow=build_context_bundle,expand_context_ref done=doc_ready fail=missing_primary_support
F f1 scope=session refs=e12 | Query handler owns session-aware orchestration.
F f2 scope=session refs=e18 | Iterative agent currently stores confidence/tool history as transient dict state.
</fcx:stable>

<fcx:turn>
@fcx v=1 sid=s77 turn=4 repo=FastCode snap=snp_9 art=ak_4 budget=900 fp=ctx_v1 mode=turn
I i1 kind=design out=spec q="detailed DSL for context rewrite"
H h1 p=.74 state=favored s=f1,f2 c=- | DSL should render typed turn state, not replace storage.
H h2 p=.21 state=open s=e31 c=- | JSON-minified format may be enough for tools but costly for model prompt.
Q q1 for=h2 need=evidence | Need token comparison after parser exists.
R eg=1 cf=0 fr=0 ra=0 xr=0 vs=clean act=answer | Enough support exists to write the initial spec.
P p1 step=1 act=write_spec why=design_requested stop=doc_ready budget=900
W keep=h1,f1,f2 drop=- protect=f1 reason=core_design
N act=expand,verify,answer,abstain expand=e12,e18,e31 tool=build_context_bundle,expand_context_ref verify=parser_tests ask=-
END refs=8 tok=846 sum=sha1:example
</fcx:turn>
```

## Example: Reset Handoff

When the current working set is drifting or role isolation matters, the
compiler should prefer a clean-context reset over carrying the whole tail
forward.

```text
<fcx:turn>
@fcx v=1 sid=s91 turn=15 repo=FastCode snap=snp_k art=ak_15 budget=850 fp=ctx_v1 mode=turn
I i1 kind=debug out=patch q="fix flaky cache invalidation path"
C out=patch need=repo_evidence,pytest allow=read_file,pytest done=verified_patch fail=ambiguous_root_cause
H h1 p=.41 state=open s=e12,e18 c=v7 | cache key bug remains plausible
H h2 p=.39 state=open s=e22,e24 c=- | stale artifact restore path also plausible
X x1 from=h3 by=v5 why=bad_assumption reopen=new_snapshot
R eg=2 cf=2 fr=1 ra=0 xr=3 vs=mixed act=reset | Competing root causes and verifier churn justify clean-context restart.
HO art=hf31 for=reset keep=h1,h2,x1 drop=o51,o52,o53 | Restart with surviving hypotheses and killed-path memory only.
N act=expand,verify,reset,abstain expand=e12,e18,e22,e24 tool=read_file,pytest verify=t_pytest ask=-
END refs=8 tok=211 sum=sha1:example_reset
</fcx:turn>
```

## Implementation Order

1. Define typed records first.
2. Implement strict serializer and parser in a pure inner package.
3. Add golden DSL fixtures with token counts.
4. Add a renderer that can emit `stable`, `turn`, and `obs` blocks.
5. Add MCP/API tools only after parser and renderer tests exist.
6. Compare FCX against pretty JSON, minified JSON, and prose on the same
   context fixture.
