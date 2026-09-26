# Scoring

This document specifies how Code4Scene scores a submission, exactly as in the
paper *Code4Scene: Benchmarking Coding Agents for Constructing and Editing 3D
Scenes* (Sections 2.3-2.4, Appendix C), and how to run the scorer offline.
The implementation is the `code4scene` package in this repository:
`code4scene/evaluation` holds the verifiers and `code4scene/protocol` holds
the scoring protocol (every weight, tolerance, rounding rule and policy ID is
in `code4scene/protocol/constants.py`).

Code4Scene scores the **scene the agent saves**, not its code, trajectory or
screenshots. Each case belongs to one of three settings:

| Setting | Task | Case score |
|---|---|---|
| `text-to-scene` (T2S) | build a scene from a prompt | Detailed Alignment, Overview Alignment, Physical Safety |
| `image-to-scene/indoor` | repair a corrupted indoor scene from reference images | Repair F1, Physical Safety |
| `image-to-scene/outdoor` | repair a corrupted outdoor scene from reference images | Repair F1, Physical Safety |

Image-to-scene is abbreviated I2S below. The public case lists are
`benchmark/public-{t2s,indoor,outdoor}-cases.txt`; the scorer reads the
schedule from those files and never hard-codes case IDs.

## Verifier outputs

Every verifier returns a report with a status:

* `measured`: a completed measurement with a finite score in [0, 1] (not a
  pass/fail threshold);
* `not_applicable`, `not_evaluated`, `error`: no numeric measurement;
* `valid` / `invalid`: submission integrity (Candidate Integrity only).

A withheld status is never an alias for a zero. How an unavailable component
enters a case score is decided by the protocol below.

## Candidate Integrity

The evaluator independently exports the saved scene and applies three checks:

1. **Scene snapshot**: the exported scene graph matches the schema and holds
   at least the task's minimum actor count (default one; an empty level would
   otherwise measure perfectly on every rate).
2. **Content and dependencies**: every package the saved scene references
   resolves under the frozen dependency policy (assets created during a run
   but not saved with the scene fail here).
3. **Asset-library manifest**: the submission, the scorer and the frozen
   content snapshot agree.

All three must pass. An invalid candidate receives a case score of zero, and
so do all of its subscores (including Repair Precision, Recall and F1).
Offline scoring re-checks the snapshot schema and minimum actor count from
the bundled snapshot and uses the recorded verdicts of checks 2 and 3, which
need the editor and the content packs.

## Physical Safety

```
S_physics = 0.5 * (1 - r_unsup) + 0.5 * S_pen
```

* `r_unsup` (leaf `floating`, lower is better) is the fraction of evaluated
  actors for which no support is found.
* `S_pen` (leaf `solid_penetration`) is the collision-free fraction of
  eligible actors.
* Both leaves keep their fixed 0.5 weight. A required leaf without a numeric
  measurement (not evaluated, error, missing, or explicitly not applicable)
  contributes **zero** (policy `physics-all-leaves-zero.v2`). A check that was
  evaluated but had no eligible actor scores the neutral value 1.
* The case value is rounded to four decimals.

**Support.** Both settings use a 5 cm support-gap tolerance.
T2S evaluates the generated actors (infrastructure and degenerate helpers
excluded) with world-space axis-aligned bounding boxes: an actor is supported
when its bottom is at most 5 cm above the world ground plane, or when another
actor lies beneath it with a vertical gap in [0, 5] cm (flush contact
included) and a footprint that contains the actor's pivot within a small
margin (metric `t2s-aabb-floating-contact-5cm.v2`). This rule is recomputed
offline from the candidate snapshot (`code4scene.protocol.physics`).
I2S evaluates the edited actors with targeted in-engine support checks (5 cm
ground gap and, when applicable, 5 cm lateral support); offline scoring uses
the recorded measurement. These are static checks, not dynamic stability.

**Solid penetration.** Actors get deterministic scene-relative physics roles;
non-solid actors, support surfaces and oversized environment proxies are not
scored targets (a trusted collision with an environment proxy still counts
as a containment failure). AABB overlap is only a broad phase: a violation
requires confirmed collision-body overlap and a native minimum-translation
depth above the actor's tolerance, which is 5 % of the shortest world-AABB
span clamped to [5, 50] cm for T2S and a fixed 5 cm for I2S. Incomplete
evidence can prove a violation but cannot establish a clean actor. These are
engine measurements; offline scoring uses the recorded values.

## Semantic Verifier (text-to-scene)

### Detailed Alignment

Each prompt is parsed once, before evaluation, into a frozen graph of atomic
requirements in four families: identity and environment, content and
quantity, attributes and materials, and spatial composition. Requirements
are resolved by an evidence ladder: Stage 1 uses the generated-scene
inventory and deterministic geometry for claims that are structurally
decidable; Stage 2 localizes the actors needed by the remaining claims and
captures targeted RGB views; Stage 3 sends only the unresolved visual claims
to the VLM judge, which answers MATCH, MISMATCH or UNKNOWN from the rendered
evidence (with a bounded alternate view and a best-evidence binary
arbitration when images remain). Missing targets or images give
NOT_EVALUATED; transport, parsing, contract or capture failures give ERROR.

Each accepted decision has a score `q_j` in [0, 1] (MATCH 1, MISMATCH 0,
deterministic checks may be fractional); an unresolved applicable requirement
scores 0 and keeps its weight. A conjunction is capped by its nested
requirements, and duplicate summaries already represented by scored
descendants carry no weight. Within family `f`, with `C_f` its non-empty
prompt clauses and `R_fc` the active predicates of clause `c`:

```
S_f        = (1/|C_f|) * sum_{c in C_f} (1/|R_fc|) * sum_{j in R_fc} q_j
S_detailed = sum_{f applicable} w_f * S_f / sum_{f applicable} w_f
```

with `w = (identity/environment 0.25, content/quantity 0.40,
attributes/materials 0.15, spatial composition 0.20)`. A family the prompt
does not ask for is removed and the remaining weights are renormalized; an
unresolved requirement inside an applicable family is not removed. Family
scores enter the weighted mean at their published four-decimal precision and
`S_detailed` is rounded to four decimals.

### Overview Alignment

A scene-graph and clearance-aware camera plan produces exactly four RGB
overview views of the generated scene. A prompt-aware VLM call scores four
dimensions with anchors 0 / 0.25 / 0.5 / 0.75 / 1:

```
S_prompt = 0.40 * prompt_alignment + 0.25 * layout
         + 0.20 * style_atmosphere + 0.15 * completeness_polish
```

An independent prompt-free call scores intrinsic structural integrity
`S_struct` (stretching, shearing, tearing, collapse, incoherent
fragmentation; anchors 1 / 0.75 / 0.5 / 0.25 / 0). Then

```
S~overview = S_prompt * (0.75 + 0.25 * S_struct)
S_overview = min(S~overview, 0.40)   if a severe defect is identified with
                                     confidence >= 0.8 and evidence from
                                     at least two views
           = S~overview              otherwise
```

rounded to four decimals.

## Ground-Truth-Based Verifier: Actor Repair F1 (image-to-scene)

Repair F1 compares three independently exported snapshots: the input scene,
the withheld ground-truth scene and the candidate. Repair targets are fixed
from input vs. ground truth before the candidate is inspected; each affected
actor is one target. `n+` targets must be present after repair (additions and
restorations), `n-` are required removals. Camera and scene-capture actors
are excluded, and changes to runtime identifiers, labels or role tags alone
are not edits.

A present-target is recovered only by a candidate actor with the correct
asset and class that passes every applicable test, with a one-to-one
matching that maximizes the number of recovered targets:

| Quantity | Test |
|---|---|
| asset and class | exact normalized structural descriptor (class, mesh or Blueprint, component-asset signature) |
| world position | `||p_c - p_g|| <= 5 cm` |
| world rotation | quaternion angle `<= 5 deg` |
| signed scale | `max_j |s_cj / s_gj - 1| <= 0.05` |
| bounds centre / extent, if recorded | `<= 5 cm` / relative `<= 0.05` |
| attributes, if recorded | exact property dictionary, material set and slot agreement |

Removal requires semantic absence of the original actor, not a rename.

Input/candidate correspondence never uses names, labels or GUIDs. It first
maximizes unchanged one-to-one correspondences (editor round-trip tolerances
0.1 cm per axis, 0.01 deg, 1e-4 scale), preserving the multiplicity of
unaffected background actors before target actors; then assigns remaining
actors with the same structural descriptor at minimum cost
`d/(1+d) + 0.2 * phi/180 + 0.05 * k` (position distance `d` in cm, rotation
distance `phi` in degrees, `k` changed fields); then lets an unchanged pose
anchor an asset replacement of a same-class actor. Correspondence only
identifies an edit; success still requires the tests above. Comparisons use
absolute world coordinates without fitting any global offset or symmetry.

With `m` matched present-targets, `d` completed removals, `C` the candidate
actors that are newly added, semantically changed, or surviving
present-targets, and `o` the unintended background losses (including
background actors repurposed to satisfy a target):

```
TP = m + d
FP = |C| - m + o
FN = (n+ - m) + (n- - d)
P  = TP / (TP + FP)        (0 when TP + FP = 0)
R  = TP / (TP + FN)
F1 = 2 TP / (2 TP + FP + FN)
```

Every target contributes exactly one TP or FN. Precision, Recall and F1 are
averaged over cases with equal case weight (macro F1 is not the harmonic mean
of macro precision and recall). Policy: `unified-actor-repair-success-f1.v2`.
Implementation: `code4scene/protocol/actor_f1.py`.

The image-to-scene paired visual comparisons and caption similarity are
reported but **not scored**.

## One overall number

A case result (`result.json`) has exactly one overall number: `overall_score`,
equal to `primary_score.score`, which is the paper case score defined below.

* **I2S.** The `gt_repair` verifier measures Actor Repair F1 from the three
  snapshots it exports and publishes it as its report `score`
  (`metrics.actor_repair_f1` holds the counts). The local/global weighted
  composite that verifier also computes is kept only as
  `metrics.diagnostics.legacy_gt_repair_composite` (and mirrored in
  `primary_score.diagnostics`); it never enters a score. The primary score is
  `round(0.8 * F1 + 0.2 * S_physics, 6)` with source
  `i2s-actor-f1-0.8-physics-0.2-case-macro.v1`. A result recorded before the
  F1 measurement existed is `unresolved` until it is rescored from its
  snapshots (`code4scene rescore`); it is never scored with the composite.
* **T2S.** The `floating` leaf measures the 5 cm AABB rule on the candidate
  snapshot the scorer exported (falling back to the in-editor measurement
  record only when no snapshot is available), so the recorded leaf equals an
  offline recomputation. The primary score is the packaged
  `text-to-scene-human-aligned` policy: `0.20 Detailed + 0.60 Overview + 0.20
  Physics`, with Physics taken leaf-wise under `physics-all-leaves-zero.v2`.

The formulas live once, in `code4scene.protocol`; the verifier layer's result
assembly (`code4scene.evaluation.primary_score`, `score_policy`) calls them.
`code4scene rescore --result-out` rebuilds a saved result through the
verifier layer and reports whether its `overall_score` equals the protocol
case score (`verifier_layer.consistent`).

## Case scores and the model score

```
S_T2S(case) = 0.20 * S_detailed + 0.60 * S_overview + 0.20 * S_physics   (4 decimals)
S_I2S(case) = 0.80 * RepairF1   + 0.20 * S_physics                       (6 decimals)
```

An unavailable required component keeps its weight and contributes zero.
Every scheduled case stays in its setting's denominator: an invalid
candidate, a missing submission and a missing result all contribute zero.
For each setting, the mean is reported only when every scheduled case is
resolved; it is never computed over fewer cases.

```
S_T2S   = mean of the T2S case scores
S_I2S   = mean of ALL I2S case scores = (N_in * S_in + N_out * S_out) / (N_in + N_out)
S_model = 0.5 * S_T2S + 0.5 * S_I2S
```

Indoor and outdoor cases are weighted equally per case (with 25 indoor and
50 outdoor public cases, the domains weigh 1/3 and 2/3). Case scores keep
full precision when averaged. Policy IDs recorded with every score:
`text-to-scene-human-aligned`, `i2s-actor-f1-0.8-physics-0.2-case-macro.v1`,
`t2s-0.5-i2s-0.5-pooled-case-macro.v2`.

## The VLM judge

Visual requirements (Stage 3 of Detailed Alignment) and Overview Alignment
need a multimodal judge. The paper used a self-hosted **Qwen3.8-27B** served
through an OpenAI-compatible chat-completions endpoint with structured tool
output, temperature 0, seed 0, thinking disabled, a 16,000-token output
budget and a 600 s timeout. The model and decoding parameters are part of
the protocol; the serving location is not. No endpoint ships with this
package: supply it with a flag or the environment.

| Variable | Meaning |
|---|---|
| `CODE4SCENE_VLM_BASE_URL` | OpenAI-compatible base URL, e.g. `http://localhost:8000/v1` |
| `CODE4SCENE_VLM_MODEL` | served model name (default `Qwen/Qwen3.8-27B`) |
| `CODE4SCENE_VLM_API_KEY` | optional bearer token (falls back to `OPENAI_API_KEY`) |
| `CODE4SCENE_EMBED_BASE_URL` | embedding endpoint for the report-only caption diagnostic |

Everything else, including Repair F1, Physical Safety, Candidate Integrity
and the aggregation, is deterministic and needs no model.

## Scoring offline

### Evidence bundles

`code4scene score` reads an evidence bundle (`code4scene.bundle.v1`): a
directory with a `bundle.json` manifest and the files it references by
relative path, each listed with its SHA-256. A bundle holds the candidate
(and, for I2S, input and ground-truth) scene snapshots, the recorded
Physical Safety report, the recorded integrity verdicts, and for T2S the
frozen Detailed Alignment decisions, the four overview renders and,
optionally, the recorded judge outputs and the Stage 3 request schedule. The
manifest is documented field by field in [EVIDENCE_BUNDLE.md](EVIDENCE_BUNDLE.md),
together with how to produce each field from your own run. Bundles built from real
runs contain scene content of third-party packs and are not distributed with
this repository.

### Commands

```bash
pip install -e .            # Python >= 3.10

# Score one bundle. VLM leaves are re-judged by the configured endpoint.
code4scene score path/to/bundle --task benchmark/public/<setting>/<case>/task.yaml \
    --vlm-base-url http://localhost:8000/v1

# Structured leaves only (no model); T2S results are flagged "partial".
code4scene score path/to/bundle --task .../task.yaml --no-vlm

# Re-aggregate the judge outputs recorded in the bundle (no model).
code4scene score path/to/bundle --judge recorded

# Paper score from a saved verifier result.json (+ its scene_evidence/);
# --result-out also writes the result rebuilt through the verifier layer.
code4scene rescore path/to/result.json --model "<label>" -o case.json \
    --result-out rebuilt.result.json

# Case scores -> setting means -> model score, over the public schedule.
code4scene aggregate scores/ --schedule benchmark --format csv -o model-scores.csv
```

`aggregate` accepts case-score JSON files (from `score`/`rescore`), JSONL,
or CSV rows with `model,setting,case_id,score` and, for I2S, optional
`repair_f1,physics,precision,recall` columns. A scheduled case without a row
counts as a missing result (zero); pass `--no-missing-as-zero` to leave the
aggregate unreported instead.

Judge modes of `score`: `live` (default) re-judges VLM-dependent leaves;
`recorded` re-aggregates recorded judge outputs; `none` (`--no-vlm`) marks
VLM-dependent leaves as not evaluated. An I2S case score never depends on the
judge, so it is complete in every mode. A T2S score produced without the
judge is reported with `status: partial` and `comparable_to_paper: false`.

### Rounding

The scorer reproduces the published values bit for bit, which fixes a few
implementation details: the support rate is rounded to four decimals before
entering `S_physics`; family scores are rounded to four decimals before the
family-weighted Detailed Alignment mean; the T2S case score is rounded to
four decimals and the I2S case score to six; aggregates are not rounded.
