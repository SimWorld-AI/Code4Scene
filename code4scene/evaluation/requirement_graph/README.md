# RequirementGraph engine

The Stage 0–3 engine behind the `semantic_requirements` verifier. It compiles a
task prompt (or a repair case's input and ground-truth scenes) into a frozen
requirement graph, then checks a candidate scene against that graph using the
scene inventory, deterministic geometry and targeted renders judged by a VLM.

This package is shared evaluation machinery, not a verifier registry. The
verifier entry point is `code4scene/evaluation/verifiers/semantic_requirements.py`;
The benchmark harness owns episode lifetime, scene evidence, renderer access, verifier
reports and publication, and this package plugs into them through
`evidence_adapter.py`, `frame_provider.py` and `vlm_client.py`.

```text
offline: requirements-draft -> (optional review) -> requirements-freeze
runtime: frozen bundle -> scoped inventory -> Stage 1
        -> unresolved Actor identity -> Stage 2 + Top-K identity judge
        -> confirmed Actor bindings -> deterministic count/spatial geometry
        -> unresolved attribute/material/global visual leaves -> Stage 3
        -> semantic_requirements report
```

## Authoring a bundle (Stage 0)

### Prompt-to-scene tasks

Every public text-to-scene task ships its frozen bundle; the requirements are
fixed before any candidate is evaluated. The authoring commands of the
benchmark harness (draft, optional review, freeze) are not part of this
release. Stage 0 drafts with the LLM configured through
`vlm_client.tool_client_from_env` and refuses a draft with blocking
diagnostics; without the LLM the deterministic rule compiler blocks every
clause it cannot fully parse.

A task names its frozen bundle under the verifier:

```yaml
verifiers:
- name: semantic_requirements
  verification_bundle: requirements/<case>.bundle.json
```

Prompt authoring uses `stage0_v2.py`, a semantic compiler that preserves every
explicit prompt constraint. Its structured output is `entities` plus typed
`requirements`. The vocabulary covers presence, quantity (exact, approximate,
bounds, ranges, qualitative amounts), sets and lists, spatial relations,
distribution, composition, style bundles, environment, logic, boundary,
attributes, materials and scene identity. Scope objects represent
each/every/some/most/per-group language. A logic requirement owns its operands,
so `A or B` receives one `any_of` scoring budget.

A deterministic normalizer handles source-span alignment, local-ID rewriting,
reference resolution, alias de-duplication, argument roles and ordinals,
coordinated-list expansion and equivalent disjunction forms. It rejects missing
semantic terms, missing scope, ungrounded spans and multi-sentence collapse.

`capabilities.py` is the versioned verifier capability matrix
(`CAPABILITY_MATRIX_VERSION`). Each requirement is bound `supported` when a
verifier route exists for its semantic type and `unplanned` otherwise;
unplanned requirements stay in the frozen graph and in the score denominator.

### Image-to-scene repair tasks

GT-backed repair uses a deterministic Stage 0 (`repair_target_authoring.py`)
with exactly two inputs, the frozen input scene graph and the canonical GT
scene graph:

```text
frozen input scene graph + canonical GT scene graph
      -> Actor-level input-to-GT diff
      -> repair target set
      -> target-scoped RequirementGraph controller
```

- GT-only Actors are add targets, input-only Actors are remove targets, and
  transform or property changes are repair targets. Only targets enter the
  semantic graph; `source_preservation` covers every non-target Actor.
- Repeated GT additions with the same visible asset identity become one
  collection/count leaf; single additions or removals become existence leaves;
  retained transform or property changes become attribute/material leaves.
- Stable IDs and asset paths are retrieval metadata and are not scored
  directly. Prompt-only relations such as `around`, `row` or `on_top_of` are
  not inferred.
- Reference images are inputs to the scene-building agent only; Stage 0 does
  not read them. Exact reconstruction of position, rotation, geometry, layout
  and appearance is scored by `scene_diff`.

## Evaluation (Stages 1–3)

`pipeline.py` is the runtime controller.

**Bindings.** The runtime first creates explicit Actor bindings. Exact
GUID, asset or name matches need no RGB. An unresolved or substituted asset
goes through lazy Top-K identity grounding (`identity_grounding.py`): one
close view, one alternate view only after `UNKNOWN`, and an early stop on
`MATCH`. The identity judge sees one designated Actor and decides only its
visible object category. `bundle.py` records ownership and population scope;
`evidence_adapter.py` resolves `candidate_all` and `additions` (additions need
a validated before/after diff or trusted operation provenance).

**Deterministic checks.** Count and spatial requirements are measured by
`deterministic.py`, `structured_deterministic.py` and the rules in `rules/`,
using only confirmed Actor bindings. They publish a continuous `check.score`
in `[0, 1]`.

**Stage 2 capture.** Stage 2 localizes candidate Actors and captures
actor-focus, collection-wide, joint-relation and global views according to
each requirement's evidence shape. Capture is claim-routed and finite:

- Only active `scene_identity` or `atmosphere` claims use whole-scene overview
  views. Attribute and material claims use actor-focus views.
- Count and spatial-relation requirements may use locator candidates to aim
  views; candidates remain hypotheses until identity grounding confirms them.
- Every localizable entity group gets a targeted view, every set or relation
  gets its shape-specific views, and an unlocalized requirement gets a grid
  fallback.
- Facets that resolve to the same Actor set share one focus program, and all
  independent Stage 2 poses run as one single-camera sweep.

**Camera placement.** Before a targeted render, `camera_visibility.py`
generates scale-aware context, close and orbital candidates from the target's
live bounding box. The engine rejects poses inside geometry or outside the
target's navigable compartment and traces visibility to the centre, faces and
corners of each requested Actor. If no pose keeps the targets in frame with a
clear line of sight, the observation reports `camera_pose_unresolved`. Close
views are raised above the target centreline. Selection uses no task id, case
name, object class, asset path or hand-authored coordinate. Actorless overview
and grid views use the planner's scene-level pose and pass through the
pixel-only health gate (`RgbFrameHealthError`), which rejects black, blank,
badly exposed or low-detail images.

**Stage 3 judging.** Unresolved attribute, material and global visual leaves go
to the Stage 3 judge (`stage3_judge.py`). Evidence arrives focus-first: the
dominant centred instance in the first frame is the subject, and later frames
establish its attributes or surroundings. The first request for a target
contains only frames routed to that Actor. Stage 3 reuses up to ten Stage 2
frames and captures only missing focus or coverage poses. The verifier's
`stage3_batch_policy` setting splits each claim's evidence into bounded
stateless VLM requests.

**Statuses.** `UNKNOWN` is internal: identity may schedule one alternate view,
and an eligible Stage 3 facet may schedule one reframe. Exhausted ambiguous
observations publish `NOT_EVALUATED`; transport, parse, contract and engine
failures publish `ERROR`.

## Published artifacts

| Artifact | Contents |
| --- | --- |
| `stage2_capture_audit.json`, `stage3_capture_audit.json` | Requirement and task IDs, target Actor IDs, bounds and capture mode per frame |
| `stage2_judge_visible_frames.json`, `stage3_judge_visible_frames.json` | Only opaque frame ID, channel and neutral view type, as the judge sees them |
| `stage3_batch_results.json` | Each request's frame joins and the claim-type-aware aggregate decision |
| `camera_visibility` (report field) | Requested and resolved pose, candidate count, visible Actor count, minimum clear-ray fraction, navigation decision and reason per targeted acquisition |

Judge requests carry only opaque frame IDs and RGB arrays.

## Scoring

Case scores follow `SEMANTIC_SCORE_POLICY` in
`code4scene/evaluation/semantic_scoring.py`
(`case-family-clause-macro-missing-zero.v4`):

- `MATCH` and `MISMATCH` contribute their measured score. `ERROR`,
  `NOT_EVALUATED` and `unplanned` requirements stay in the denominator with a
  score of 0; `known_coverage` reports the weighted share with a known score.
- Nested `scene_identity`, `set` and `logic` composites are not counted again
  when scored descendants represent them. A conjunctive parent scores no higher
  than its lowest direct child.
- Requirements are grouped into prompt clauses (sentence or semicolon). Every
  atomic predicate type has weight 1 within one of four semantic families:
  identity/environment, content/quantity, attributes/materials, and spatial
  composition/relations. Each family macro-averages its clause scores, and the
  case score macro-averages the families with `SEMANTIC_FAMILY_WEIGHTS` (all
  1). Family scores, weights and coverage are published under
  `semantic_subscores`.

Suite scores (`weighted-macro-mean-of-case-scores`) average case scores, never
requirement rows, so every case has one vote regardless of how many
requirements it parsed into. Explicit positive case weights may be supplied as
policy. A missing case leaves the suite score unresolved and is reported in
`complete_case_weight_coverage`. Compare models on the same frozen task and
requirement graph.

Physics and GT repair claims are recorded under their own verifiers and are not
part of the semantic denominator; a task declares those verifiers separately.

## Modules

| Area | Files |
| --- | --- |
| Contracts and bundle | `contracts.py`, `bundle.py`, `capabilities.py`, `inputs.py` |
| Stage 0 authoring | `stage0.py`, `stage0_v2.py`, `stage0_contracts.py`, `stage0_artifacts.py`, `authoring.py`, `rule_compiler.py`, `repair_target_authoring.py`, `legacy_migration.py` |
| Stage 1 and deterministic rules | `stage1.py`, `actor_inventory.py`, `deterministic.py`, `deterministic_alignment.py`, `structured_deterministic.py`, `legacy_atomic.py`, `rules/` |
| Identity and retrieval | `identity_grounding.py`, `llm_identity_locator.py`, `semantic_retrieval.py`, `asset_candidates.py` |
| Stage 2 capture | `stage2.py`, `stage2_capture.py`, `stage2_routing.py`, `stage2_tasks.py`, `stage2_frames.py`, `stage2_contracts.py`, `stage2_artifacts.py`, `overview_planning.py`, `camera_visibility.py`, `actions.py` |
| Stage 3 judging | `stage3.py`, `stage3_explorer.py`, `stage3_judge.py`, `stage3_contracts.py`, `stage3_artifacts.py`, `visual_claims.py` |
| Adapters | `pipeline.py`, `evidence_adapter.py`, `frame_provider.py`, `runtime.py`, `vlm_client.py`, `existing_llm.py` |
