# Evidence bundles (`code4scene.bundle.v1`)

An evidence bundle holds everything needed to score one case offline:
`code4scene score <bundle_dir>` reads nothing else. This page describes the
format field by field and how to produce each field from your own run.
The reference implementation is `code4scene/bundle.py`; the scoring rules are
in [SCORING.md](SCORING.md).

Bundles built from real runs contain actor lists of third-party content packs.
Treat them as evaluation artifacts: keep them next to your runs, do not commit
them to this repository.

## Layout

```
<bundle_dir>/
  bundle.json                         manifest (below)
  scenes/candidate.scene.json         the saved scene, as exported by the scorer
  scenes/input.scene.json             image-to-scene only: the corrupted input
  scenes/ground_truth.scene.json      image-to-scene only: the withheld target
  physics/physical_safety.json        the recorded Physical Safety report
  semantic/decisions.json             text-to-scene: Detailed Alignment decisions
  semantic/stage3_plan.json           text-to-scene, optional: Stage 3 request schedule
  overview/judgement.json             text-to-scene, optional: recorded overview judge output
  renders/overview/view_{1..4}.png    text-to-scene: the four overview RGB views
```

Only `bundle.json` has a fixed name; every other path is whatever the manifest
says. Any JSON file may be stored gzip-compressed with a `.json.gz` suffix.

Rules enforced by `code4scene validate-bundle` and on every load:

* all paths are relative to the bundle directory, use `/`, and may not be
  absolute or contain `..`;
* every referenced file appears in `files` with its SHA-256, and the hash is
  checked when the bundle is loaded;
* unknown top-level keys are rejected.

## `bundle.json`, field by field

```json
{
  "schema_version": "code4scene.bundle.v1",
  "task": {"id": "<case id>", "setting": "image-to-scene/outdoor", "sha256": "<optional>"},
  "run": {"model": "<label>", "submission": "submitted", "candidate_sha256": "<optional>",
          "batch_status": "complete", "authoritative": true},
  "candidate_integrity": {"status": "valid",
                          "leaves": {"candidate_snapshot_integrity": "valid",
                                     "content_and_dependency_parity": "valid",
                                     "asset_library_manifest_parity": "valid"},
                          "failure_reason": null},
  "scenes": {"candidate": "scenes/candidate.scene.json",
             "input": "scenes/input.scene.json",
             "ground_truth": "scenes/ground_truth.scene.json"},
  "physics": {"report": "physics/physical_safety.json"},
  "semantic": {"decisions": "semantic/decisions.json", "stage3_plan": "semantic/stage3_plan.json"},
  "overview": {"judgement": "overview/judgement.json",
               "views": ["renders/overview/view_1.png", "...", "renders/overview/view_4.png"]},
  "renders": [{"role": "overview", "view": "view_1", "channel": "rgb",
               "path": "renders/overview/view_1.png"}],
  "files": {"scenes/candidate.scene.json": "<sha256>", "...": "..."},
  "notes": "free text, optional"
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | exactly `code4scene.bundle.v1` |
| `task.id` | yes | the case ID (`benchmark/public-*-cases.txt`) |
| `task.setting` | yes | `text-to-scene`, `image-to-scene/indoor` or `image-to-scene/outdoor` |
| `task.sha256` | no | SHA-256 of the `task.yaml` the run used; reported next to the loaded task's hash by `score --task` |
| `run.model` | no | label that `aggregate` groups by |
| `run.submission` | no | `submitted` (default) or `missing`; a missing submission scores zero |
| `run.candidate_sha256` | no | hash of the saved `.umap`, for provenance |
| `run.batch_status`, `run.authoritative` | no | copied from the verifier result, for provenance |
| `candidate_integrity.status` | if submitted | `valid` or `invalid`; an invalid candidate scores zero |
| `candidate_integrity.leaves` | no | verdict of each of the three integrity checks |
| `candidate_integrity.failure_reason` | no | recorded reason for an invalid verdict |
| `scenes.candidate` | valid I2S; T2S recommended | candidate scene snapshot |
| `scenes.input`, `scenes.ground_truth` | valid I2S | input and ground-truth snapshots; Repair F1 needs all three |
| `physics.report` | valid I2S | recorded `physical_safety` verifier report |
| `semantic.decisions` | T2S | `{"report_status": "<status of the Semantic report>", "decisions": [...]}` |
| `semantic.stage3_plan` | no | recorded Stage 3 schedule, used only by `score --judge live` to re-judge visual requirements |
| `overview.judgement` | no | recorded prompt-aware dimension scores, structural score and severe-defect eligibility |
| `overview.views` | T2S live | the four overview RGB views, in order |
| `renders` | no | index of every image in the bundle (role, view, channel) |
| `files` | yes | SHA-256 of every file referenced above |

### What each field is used for

| Score component | Bundle fields | Recomputed offline? |
|---|---|---|
| Candidate Integrity | `candidate_integrity`, `scenes.candidate` | snapshot schema and minimum actor count are re-checked; dependency and asset-library parity use the recorded verdicts |
| Repair F1 (I2S) | `scenes.input`, `scenes.ground_truth`, `scenes.candidate` | yes, fully |
| Support / floating, T2S | `scenes.candidate`, else `physics.report` | yes, the 5 cm AABB rule on the snapshot |
| Support / floating, I2S | `physics.report` | no: an in-engine support trace |
| Solid penetration | `physics.report` | no: in-engine collision-body overlap and minimum-translation depth |
| Detailed Alignment | `semantic.decisions` (+ `semantic.stage3_plan`, images for `--judge live`) | structural decisions yes; visual decisions need the VLM or the recorded verdicts |
| Overview Alignment | `overview.views` for `--judge live`, `overview.judgement` for `--judge recorded` | needs the VLM, or the recorded judge outputs |

## Producing a bundle from your own run

### Recommended: score with the verifier layer, then pack

1. Save the agent's scene (`.umap`) and open it in a separate **scoring**
   editor (Unreal Engine 5.8, with the task's content packs mounted as
   described in [BUILD_DATASET.md](BUILD_DATASET.md)). The agent's editor
   must not be used for scoring.
2. Run the canonical verifiers against that editor with
   `code4scene.evaluation.verifiers.run(task, record, ids, bridge=..., out_dir=...)`.
   `bridge` is a `code4scene.core.bridge.Bridge` connected to a Python bridge
   inside the editor (TCP or Unix socket, one JSON request per connection;
   see the module docstring). The verifiers export the snapshots and run the
   in-editor probes themselves, writing them under `out_dir` (see the table
   below). Assemble the result from the returned reports with
   `score_policy.apply_to_result(result, score_policy.load(config_file(
   "score-policies", "text-to-scene-human-aligned.yaml")))` for text-to-scene
   or `primary_score.apply_to_result(result)` for image-to-scene (modules in
   `code4scene.evaluation`, `config_file` in `code4scene.resources`), and save
   it as `result.json`. Its `overall_score` is the paper case score.

   What the call needs, beyond the task:

   * **An editor-side bridge.** This repository ships the client only; a
     reference server is `tools/c4s_editor_bridge.py` (start the scoring
     editor with `-ExecutePythonScript=tools/c4s_editor_bridge.py` and
     `C4S_BRIDGE_SOCK=<socket>`). `UnrealEditor-Cmd` closes the editor when
     that script returns, and the editor's embedded Python does not run
     background threads while the editor is idle, so the server answers
     requests from the startup script itself. The bridge runs any Python it
     is sent: put the socket in a directory only you can open (for example
     `$(mktemp -d)/c4s.sock`), and start the scoring editor after the agent
     has exited.
   * **The candidate level open in that editor**, copied into the scoring
     project under the `/Game/...` path it was saved at (the package name is
     stored in the `.umap`), together with any assets the agent saved for it
     (its own materials, textures or blueprints) at their `/Game/...` paths:
     a package the level references that the scoring project lacks fails
     Candidate Integrity.
   * **`record`**: at least `scene_map` (the candidate's `/Game/...` package)
     and `scene_dependencies` (the JSON written by
     `ue_scripts/export_scene_dependencies.py` for that level). Without the
     manifest Candidate Integrity reports `error` and every score is withheld.
   * **`scoring=`**: an object with `bridge` (the Bridge) and `measure_path`
     (a file path under your output directory). It marks the editor as the
     independent scoring editor; without it `gt_repair` and
     `physical_safety` refuse to capture ("runtime repair scene capture
     requires the independent scoring editor"). Pass `bridge=None`.

   `tools/score_saved_level.py` does all of this for an image-to-scene case,
   and `tools/score_saved_build.py` for a text-to-scene case, whose pictures
   need a scoring editor that renders (`tools/c4s_render_bridge.py`).
3. Pack it:

   ```bash
   code4scene make-bundle path/to/result.json path/to/bundle \
       --scenes-dir path/to/scene_evidence/<run>/ --model "<label>"
   code4scene validate-bundle path/to/bundle
   ```

   `--scenes-dir` holds `{input,ground_truth,candidate}.scene.json[.gz]`;
   when omitted, the unique `scene_evidence/*/` directory next to the result
   is used. The four overview frames are copied when the result's overview
   report references them on disk. `--compress` stores the snapshots gzipped;
   `--link` symlinks instead of copying.

`code4scene rescore result.json` performs steps 3 and scoring in one go and,
with `--result-out`, writes the result rebuilt through the verifier layer.

### Manual: run the editor scripts yourself

The scripts in `code4scene/ue_scripts/` run inside the Unreal editor's Python
interpreter (enable the *Python Editor Script Plugin*). They are data files of
the package, never imported by it. Each script reads its inputs from
module-level globals and writes one JSON file. Any mechanism that executes
Python in the editor works: the Output Log's Python console, a startup script
passed with `-ExecutePythonScript`, UE's Python remote execution, or the
bridge above. The simplest wrapper is `runpy`:

```python
# Run inside the Unreal editor's Python.
import runpy
SCRIPTS = r"<site-packages>/code4scene/ue_scripts"   # python -c "import code4scene, os; print(os.path.dirname(code4scene.__file__))"

runpy.run_path(SCRIPTS + "/export_scene_snapshot.py", init_globals={
    "SCENE_DISTANCE_MAP": "",                           # "" = the level currently open
    "SCENE_DISTANCE_OUTPUT": r"D:/runs/case/candidate.scene.json",
})
```

| Script | Globals it reads | Writes | Bundle field |
|---|---|---|---|
| `export_scene_snapshot.py` | `SCENE_DISTANCE_MAP` (map package to load; empty = current level), `SCENE_DISTANCE_OUTPUT` (output path) | scene snapshot, `schema_version` `0.3.0` | `scenes.candidate`; run it on the task's `init_map` for `scenes.input` and on the ground-truth map for `scenes.ground_truth` |
| `measure_scene_physics.py` | `SCENE_PHYSICS_OUTPUT`, `SCENE_PHYSICS_TARGETS_JSON` (JSON list of `{actor_path \| stable_actor_id \| label}`), `SCENE_PHYSICS_OPTIONS_JSON` (support model and tolerances) | per-actor support and collision measurements, `schema_version` `0.5.0` | input to the `physical_safety` verifier, whose report is `physics.report` |
| `export_scene_dependencies.py` | `SCENE_DEPENDENCIES_OUTPUT`, `SCENE_DEPENDENCIES_MAP` (empty = current level) | every package the level references and whether it resolves | input to Candidate Integrity (content and dependency parity) |
| `measure_reachability.py` | `SCENE_REACHABILITY_OUTPUT`, `SCENE_REACHABILITY_OPTIONS_JSON` | navmesh reachability per actor | not scored (the Embodied Utility Score is reported separately) |

Scene snapshot contents: `actors` (one entry per level actor), `actor_count`,
`map_path`, `units: "cm"`, a material-parameter catalog, and
`export_metadata.status` (`success` or `error` with the error text; a failed
export must not be scored). Each actor carries `label`, `name`, `actor_path`,
`class`, `asset_path`, `component_asset_paths`, `material_paths`,
`component_material_slots`, `actor_tags`, `stable_actor_id`, `collision`,
`transform` (`location_cm`, `rotation_deg`, `scale`), `bounds` (`origin_cm`,
`extent_cm`) and `properties`.

The physics probe needs the measurement targets and tolerance options that
the verifier derives from the task (the scene-relative T2S penetration
tolerance, the fixed 5 cm I2S tolerance, the support model); they are built by
`code4scene.evaluation.ue_evidence`. Run the probe through the verifier layer
rather than by hand, so the `physical_safety` report uses exactly those
settings. `ue_evidence` can also read pre-exported files instead of the
editor (verifier spec keys `candidate_scene`, `candidate_measurements`,
`input_scene`, `input_measurements`); that route is not exercised by the
release tests. Dependency parity and the asset-library manifest check always
need an editor with the content mounted.

The text-to-scene overview views and the Stage 2/3 frames are captured by the
`overview_prompt_alignment` and `semantic_requirements` verifiers through the
editor (`code4scene/evaluation/render.py`); they are not standalone scripts.

## Scoring a bundle

```bash
code4scene score path/to/bundle --task benchmark/public/<setting>/<case>/task.yaml \
    --vlm-base-url http://localhost:8000/v1     # VLM leaves re-judged
code4scene score path/to/bundle --no-vlm        # structured leaves only
code4scene score path/to/bundle --judge recorded
```

Image-to-scene bundles never need the judge. See [SCORING.md](SCORING.md) for
the judge modes and the model-level aggregation.
