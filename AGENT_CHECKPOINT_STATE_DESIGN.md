# Agent Checkpoint and State Management Design

This document defines a practical checkpoint and state-management design for
agent-driven project work.

The core position is simple:

- rollback files
- version project state
- compensate events
- isolate processes
- record network side effects

Do not claim that arbitrary commands are rollbackable. They are not. The system
should make rollback capabilities explicit by layer.

## Scope

This design covers:

- workspace file checkpoints
- command and event journaling
- non-rollbackable network and process effects
- project-level task state
- daemon event storage
- optional host/container snapshot adapters

This design does not cover:

- full VM snapshot orchestration
- distributed transaction guarantees across external APIs
- replacing Git, Dolt, PostgreSQL, SQLite, Btrfs, ZFS, or Kubernetes

## Sources

The design is based on these primitives:

- Git reflog, stash, worktree, reset, and refs:
  `https://git-scm.com/docs/git-reflog`,
  `https://git-scm.com/docs/git-stash`,
  `https://git-scm.com/docs/git-worktree`,
  `https://git-scm.com/docs/git-reset`
- Docker commit, storage drivers, volumes, and checkpoint/restore:
  `https://docs.docker.com/reference/cli/docker/container/commit/`,
  `https://docs.docker.com/engine/storage/drivers/`,
  `https://docs.docker.com/engine/storage/volumes/`,
  `https://docs.docker.com/reference/cli/docker/checkpoint/`
- Btrfs snapshots and copy-on-write subvolumes:
  `https://btrfs.readthedocs.io/en/stable/dev/dev-btrfs-design.html`
- ZFS snapshots:
  `https://openzfs.github.io/openzfs-docs/man/master/8/zfs-snapshot.8.html`
- Dolt version-control SQL procedures:
  `https://docs.dolthub.com/sql-reference/version-control`
- PostgreSQL `LISTEN` / `NOTIFY` and logical decoding:
  `https://www.postgresql.org/docs/current/sql-listen.html`,
  `https://www.postgresql.org/docs/current/logicaldecoding-explanation.html`
- SQLite WAL:
  `https://www.sqlite.org/wal.html`
- K3s lightweight Kubernetes:
  `https://docs.k3s.io/`

## Capability Tiers

### Tier 1: Workspace Files

Files are the most rollbackable part of an agent run.

Default:

- Git checkpoint refs for normal repository work
- explicit untracked-file capture policy
- explicit ignored-file policy
- restore plan before rollback

Optional stronger adapters:

- Btrfs snapshot
- ZFS snapshot
- archive copy fallback for non-Git/non-snapshot paths

Rule:

```text
file rollback is allowed only inside an approved workspace scope
```

### Tier 2: Project Task State

Project/task/domain state should be versioned, not hidden in prompt history.

Use Dolt for state that benefits from branches, commits, diffs, and merges:

- task plans
- acceptance contracts
- agent handoffs
- project metadata
- decision records
- task-level state transitions

Rule:

```text
Dolt stores versioned project state, not high-volume daemon events
```

### Tier 3: Daemon Events

Daemon events are append-only history.

Use PostgreSQL for multi-process/server mode and SQLite WAL for local mode.

Store:

- command intents
- command results
- tool calls
- privilege decisions
- process metadata
- network side-effect records
- verifier results
- checkpoint lifecycle events
- compensation events

Rule:

```text
events are never rolled back; later events compensate or supersede earlier ones
```

### Tier 4: Processes

Temporary processes are generally not worth rollback.

Use isolation and cleanup:

- run operations in containers when practical
- use K3s/Kubernetes for stronger service/process isolation when needed
- kill task-owned processes on abort
- record process tree, exit code, logs, and resource usage

Rule:

```text
processes are isolated and killed, not rolled back
```

### Tier 5: Network Calls

Network calls cannot be rolled back in the general case.

Use privilege management and evidence:

- network allowlist by recipe
- proxy/logging where practical
- rate and cost budgets
- manual-remediation flag for irreversible effects
- explicit side-effect class on every operation

Rule:

```text
network effects are controlled, recorded, and manually resolved when needed
```

## Architecture

```text
CheckpointManager
  GitWorkspaceSnapshotter
  ArchiveSnapshotter
  FsSnapshotAdapter
    BtrfsSnapshotter
    ZfsSnapshotter
  CommandJournal
  ResourceJournal
  DoltStateManager
  DaemonEventStore
    PostgresEventStore
    SqliteEventStore
  RestorePlanner
  TransactionCoordinator
  PrivilegeManager
  ProcessIsolator
```

## Component Responsibilities

### CheckpointManager

Coordinates checkpoint lifecycle:

- begin checkpoint
- list checkpoint state
- commit checkpoint
- abort checkpoint
- build restore plan
- execute restore plan

It delegates actual storage behavior to lower-level adapters.

### GitWorkspaceSnapshotter

Portable default for repository files.

Responsibilities:

- snapshot current `HEAD`
- snapshot tracked modifications
- capture selected untracked files
- record ignored/excluded policy
- write hidden checkpoint refs
- restore workspace files from checkpoint tree

Recommended ref namespace:

```text
refs/fastcode/checkpoints/<checkpoint_id>
```

The implementation should avoid disturbing the user's current index when
possible. A temporary index or alternate worktree is preferable to staging user
files directly.

### ArchiveSnapshotter

Fallback when Git is unavailable or insufficient.

Responsibilities:

- copy selected workspace paths to content-addressed archive storage
- record file mode, size, mtime, hash, and relative path
- restore only approved paths

This is slower than Git but portable.

### FsSnapshotAdapter

Optional host-level snapshot interface.

Adapters:

- `BtrfsSnapshotter`
- `ZfsSnapshotter`

Responsibilities:

- detect host support
- verify workspace is on a snapshot-capable volume
- create snapshot before operation
- rollback snapshot when approved

These adapters are accelerators, not the baseline. They often do not work
inside ordinary containers because the container does not own the host
subvolume or dataset.

### CommandJournal

Records command history, not rollback truth.

Fields:

- `command_id`
- `transaction_id`
- `cwd`
- `argv`
- `env_allowlist`
- `started_at`
- `finished_at`
- `exit_code`
- `stdout_ref`
- `stderr_ref`
- `pre_file_delta_ref`
- `post_file_delta_ref`
- `side_effect_class`

### ResourceJournal

Records effects outside simple workspace files.

Fields:

- `resource_event_id`
- `transaction_id`
- `resource_kind`
- `resource_id`
- `operation`
- `effect_class`
- `rollback_adapter`
- `compensation_status`
- `manual_resolution_required`

Resource kinds:

- `network`
- `process`
- `database`
- `cache`
- `container`
- `volume`
- `host`

Effect classes:

- `file_only`
- `workspace_plus_cache`
- `external_resource`
- `network_irreversible`
- `unknown`

### DoltStateManager

Owns versioned project state.

Responsibilities:

- create task branch
- commit accepted state transitions
- diff state between task branches
- merge task branch into accepted project state
- reset or preserve abandoned branches by policy

Suggested branch namespace:

```text
task/<task_id>
agent/<agent_id>/<task_id>
checkpoint/<checkpoint_id>
```

Dolt is not the daemon event store. Use it for state that benefits from
branching and review.

### DaemonEventStore

Append-only event storage.

PostgreSQL mode:

- multi-process daemon
- server deployment
- concurrent workers
- richer query and retention policies

SQLite WAL mode:

- local single-user mode
- development mode
- portable embedded deployment

Events should be immutable. Corrections are new events.

### RestorePlanner

Builds an explicit restore plan before any rollback.

Inputs:

- checkpoint snapshot
- current workspace state
- command journal
- resource journal
- user/agent policy

Outputs:

- files to restore
- files to delete
- files to preserve
- unresolved external effects
- required manual actions
- rollback risk summary

### TransactionCoordinator

Owns lifecycle:

```text
BEGIN_TASK_TX
RUN_OP
DECIDE
COMMIT_TX
ABORT_TX
```

It does not run arbitrary model logic. It executes deterministic state
transitions and records decisions.

### PrivilegeManager

Applies recipe-level permissions:

- network allowlist
- filesystem write scope
- command allowlist or risk class
- secret/env exposure policy
- max cost
- max runtime

### ProcessIsolator

Runs operations in isolated execution contexts where practical:

- local subprocess for low-risk commands
- container per task or operation for medium risk
- K3s/Kubernetes job or pod for stronger service isolation

Process isolation is not rollback. It reduces blast radius and makes cleanup
more predictable.

## Transaction Flow

### Begin

```text
BEGIN_TASK_TX
  create workspace checkpoint
  create or select Dolt task branch
  create daemon event stream id
  apply privilege profile
  record acceptance contract
```

### Run Operation

```text
RUN_OP
  record command intent
  run in selected isolation mode
  append command event
  capture file delta
  capture process/network metadata
  update Dolt task branch only through coordinator
```

### Commit

```text
COMMIT_TX
  preserve workspace changes
  commit or merge Dolt task state
  append accepted event
  keep checkpoint refs for audit or retention window
```

### Abort

```text
ABORT_TX
  build restore plan
  restore approved workspace files
  delete generated files only if policy allows
  kill task-owned processes
  append compensation event
  keep, reset, or archive Dolt task branch by policy
  mark network effects as external/manual
```

## Rollback Policy

Do not expose a single `rollback()` button internally. Expose a restore plan.

A restore plan should classify every change:

- `restore_file`
- `delete_generated_file`
- `preserve_user_file`
- `manual_external_resolution`
- `adapter_compensation`
- `cannot_rollback`

This prevents accidental deletion of user changes made after the checkpoint.

## Container And Filesystem Notes

### Git Baseline

Git is the default because it works in containers, CI, local machines, and
remote workspaces.

Limitations:

- does not capture ignored files unless policy includes them
- does not capture files outside repo scope
- does not capture process, network, database, or package-manager state

### Btrfs/ZFS Adapter

Btrfs/ZFS snapshots are better when available because they can cover workspace
state below Git's awareness.

Limitations:

- require host filesystem support
- require appropriate privileges
- ordinary containers usually cannot manage host snapshots
- bind mounts and volumes must be checked carefully

### Docker/Container Snapshot

Docker container snapshots are not a baseline checkpoint mechanism.

Reasons:

- workspace is often a bind mount or volume, not inside the container writable
  layer
- `docker commit` does not include mounted volumes
- CRIU checkpoint/restore is environment-dependent and not a portable default

Use containers for process isolation, not as the primary rollback mechanism.

### K3s/Kubernetes

Use K3s/Kubernetes when the environment needs service isolation:

- per-task pod/job
- network policy
- resource limits
- service namespace separation
- cleanup by namespace/job label

This improves containment but still does not rollback external side effects.

## Data Model Sketch

### CheckpointRecord

```text
checkpoint_id
task_id
workspace_id
backend
created_at
base_ref
snapshot_ref
dirty_manifest_ref
untracked_policy
ignored_policy
status
```

### CommandEvent

```text
command_id
transaction_id
checkpoint_id
cwd
argv_ref
env_ref
isolation_mode
started_at
finished_at
exit_code
stdout_ref
stderr_ref
side_effect_class
```

### ResourceEvent

```text
resource_event_id
transaction_id
resource_kind
operation
resource_ref
effect_class
rollback_adapter
compensation_status
manual_resolution_required
```

### RestorePlan

```text
restore_plan_id
checkpoint_id
created_at
file_actions_ref
resource_actions_ref
risk_summary
requires_user_approval
status
```

### ProjectStateCommit

```text
state_commit_id
task_id
dolt_branch
dolt_commit
parent_commit
transition_kind
accepted_by
created_at
```

## Module Boundary Guidance

If implemented inside a larger agent monorepo, keep these packages separate:

```text
checkpoint/
  workspace snapshots, restore planner, transaction coordinator

state/
  Dolt-backed project/task state

events/
  PostgreSQL/SQLite append-only daemon events

isolation/
  subprocess, container, k3s/kubernetes execution wrappers

privileges/
  command, network, filesystem, secret policies

fastcode/
  repository intelligence and context substrate
```

FastCode should integrate with this system by references:

- checkpoint ID
- transaction ID
- workspace snapshot ref
- context bundle ref
- evidence ref
- activation record

FastCode should not own the full checkpoint daemon unless the projects are
later consolidated and the package boundary remains explicit.

## Minimum Viable Implementation

V0 should implement:

- Git workspace checkpoint refs
- selected untracked-file capture
- append-only SQLite WAL event store
- command journal
- restore planner
- manual external-effect reporting

V1 should add:

- PostgreSQL daemon event store
- Dolt project-state branches
- privilege profiles
- container execution wrapper

V2 should add:

- Btrfs/ZFS host adapters
- K3s/Kubernetes isolation mode
- resource compensation adapters for known services
- hash-chained event audit records

## Non-Negotiable Invariants

- Never claim command rollback.
- Never mutate historical event rows.
- Never rollback files without a restore plan.
- Never delete untracked files without policy approval.
- Never treat network calls as reversible.
- Never let model output directly execute rollback.
- Always preserve enough evidence to explain what was changed, what was
  restored, and what remains unresolved.

## Bottom Line

The checkpoint system should be honest about physics:

```text
Git covers most file mistakes.
Btrfs/ZFS can strengthen workspace rollback when the host supports it.
Containers and K3s isolate process blast radius.
Dolt versions project state.
PostgreSQL/SQLite records daemon history.
Network effects are controlled and audited, not rolled back.
```

That model is portable enough for containers, strong enough for local power-user
hosts, and explicit enough for long-running agent work.
