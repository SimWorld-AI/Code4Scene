# Building the Code4Scene public dataset

The public set is 225 cases: 150 text-to-scene construction, 25 indoor
image-to-scene editing and 50 outdoor image-to-scene editing. Its levels are built from
third-party Unreal Engine content packs, which this repository does not
redistribute. You download the packs from Fab yourself, then run the dataset
builder, which rebuilds on a stock Unreal Engine 5.8 editor:

* the ground-truth (GT) level of every image-to-scene source scene,
* the corrupted Input level of every image-to-scene case,
* the empty start level of the text-to-scene cases,
* the reference views shown to the agent (optional, needs a GPU),

and then checks each level against a content fingerprint of the level the
benchmark was scored on.

## What the repository ships

Under `benchmark/public/`:

| Path | Contents |
|---|---|
| `text-to-scene/<case>/task.yaml` | prompt, packs, budget, verifiers |
| `text-to-scene/<case>/requirements/<case>.bundle.json` | frozen requirement graph used by the semantic verifier |
| `image-to-scene/{indoor,outdoor}/<case>/task.yaml` | prompt, packs, budget, verifiers, expected fingerprint roots |
| `.../<case>/task.label.json` | GT map path and case id for the scorer |
| `.../<case>/recipe.json` | the ordered edit operations that turn the GT level into the Input level |
| `.../<case>/cameras.json` | reference-view poses, lighting, capture and output settings |
| `.../<case>/input.fingerprint.json` | expected Input fingerprint, stored as a delta over the GT fingerprint |
| `scenes/<scene>/scene.json` | how to build the GT level from the pack's demo map |
| `scenes/<scene>/gt.fingerprint.json` | expected GT fingerprint (per-actor hashes only) |

Nothing here contains a level, mesh, texture, rendered image, or a
transform-level description of a seller's demo map. Recipes name `/Game/...`
asset paths and give absolute transforms only for the few actors a case edits.

## Requirements

* Unreal Engine **5.8** (the benchmark levels were exported from 5.8.0; later
  5.8.x releases are expected to work, and the verification step tells you if
  they do not). Windows or Linux. macOS is untested.
* Python 3.9 or newer on the host. Optional: `PyYAML` (nicer pack hints).
  The `render` step needs `Pillow` or `ffmpeg` to resize and encode the
  reference views.
* Disk: about 125 GB for the installed packs of the whole public set (about
  80 GB for the image-to-scene packs alone, one pack is 34 GB), plus about
  5 GB for the built levels, about 0.5 GB for the exported snapshots, and the
  editor's derived-data cache (typically 20 to 60 GB).
* A GPU only for the `render` step. All other steps run with `-NullRHI`.

## 1. Create an empty project

Create a new UE 5.8 project from the **Blank** template, Blueprint, no starter
content, for example `Code4SceneData`. Close the editor again: the builder
starts its own editor process and the project must not be open elsewhere.

On a headless Linux server (no editor GUI), create the same project by hand:
copy `<UE>/Templates/TP_BlankBP/Config` into a new directory, copy
`<UE>/Templates/TP_BlankBP/TP_BlankBP.uproject` next to it as
`Code4SceneData.uproject`, drop `Config/TemplateDefs.ini`, and create an empty
`Content/` directory. A minimal `.uproject` with only `"FileVersion": 3` and an
empty `"Plugins"` list also works; step 3 adds the plugins the builder needs.

The indoor scene `cyberpunk-toilet` and the outdoor scene `old-building` also
use Epic's **Starter Content** (`Content/StarterContent`). Add it with *Add >
Add Feature or Content Pack > Content > Starter Content*. For `old-building` the
`gt` and `inputs` steps copy its `T_MacroVariation` texture to
`/Game/Cabin_Pack/Master_Mat/T_MacroVariation`, which the pack's materials
reference but the pack does not ship (the scene's `supplements` in
`scene.json`).

## 2. Install the packs

`benchmark/packs.yaml` (and `docs/PACKS.md`) lists every pack with its Fab
listing, the `Content/<folder>` it must occupy, and the cases that need it.
For each pack:

1. Open the Fab listing, add it to your library, and use *Add to Project* in
   the Epic Games Launcher (or Fab in the editor) with your project as target.
   If the launcher hides the project because the pack does not list 5.8, tick
   *Show all projects*.
   Where a pack's `notes` in `packs.yaml` name the build the benchmark was
   made from (for example the UE 4.19 build of `DetectiveOffice`), install
   that build rather than the newest one: some packs were reworked in later
   builds, and with a different build `verify` can report a mismatch (see
   [PACKS.md](PACKS.md), "Build differences").
2. Check that the pack landed in exactly `Content/<folder>` as listed. Do not
   rename, move, re-save or "fix up redirectors" in pack folders: the
   benchmark's identities are derived from the demo maps' actor names and a
   re-saved demo map may no longer match.

You can install only the packs of the cases you want and pass `--cases` or
`--settings` (below).

## Where to run the builder

Run every `python -m dataset_builder.build` command below from the root of this
repository: `pip install -e .` installs only the `code4scene` package, so
`dataset_builder` is importable from the repository root only. Relative
`--dataset` paths are resolved against the current directory. The editor
reads its job script from the repository, so the repository path must not
contain `.py` anywhere before `dataset_builder/ue/c4s_job.py` (for example a
directory named `my.pyprojects`), and for `render` no comma or apostrophe;
the builder refuses such a path.

## 3. Enable the editor scripting plugins

The builder needs *Python Editor Script Plugin* and *Editor Scripting
Utilities*. Either enable them in *Edit > Plugins*, or let the builder add them
to the `.uproject` (a backup is written):

```bash
python -m dataset_builder.build --project /path/Code4SceneData/Code4SceneData.uproject --steps init-project
```

The same step writes the package redirects the packs need and the renderer
settings the benchmark renders with (`benchmark/renderer_settings.txt`) into
`Config/DefaultEngine.ini`, keeping a backup of the original file.

## 4. Check the packs (no editor needed)

```bash
python -m dataset_builder.build --project /path/Code4SceneData/Code4SceneData.uproject \
    --dataset ./code4scene-dataset --steps check
```

This looks for each required `Content/<folder>`, each source demo map and a
few key assets per case (for text-to-scene, every asset in the prompt's
palette). `code4scene-dataset/reports/packs.json` lists what is missing, with
the Fab listing for every missing folder.

## 5. Build the levels and verify them

```bash
export UE_EDITOR=/path/to/UE_5.8/Engine/Binaries/Linux/UnrealEditor-Cmd    # Win64: UnrealEditor-Cmd.exe
python -m dataset_builder.build --project /path/Code4SceneData/Code4SceneData.uproject \
    --dataset ./code4scene-dataset --steps check blank gt inputs verify package --allow-missing-packs
```

If any pack of the selected cases is missing, the builder stops before the
`gt`, `inputs` and `render` steps unless `--allow-missing-packs` is given;
with it, the cases whose packs are installed are built and the others are
skipped. The `TrainStation` pack has no Fab listing yet, so its 5 outdoor
cases are skipped in every build. The builder also stops before starting the
editor if the `.uproject` does not enable the plugins of step 3.

What each step does:

* `blank` creates the empty start level `/Game/SceneBench/BlankStage` used by
  the text-to-scene tasks.
* `gt` loads each scene's demo map (`scenes/<scene>/scene.json`), writes the
  benchmark identity tags, applies the scene's patches, saves the result as
  `/Game/Code4SceneGT/<scene>/GT`, and exports a snapshot to
  `snapshots/gt/<scene>.scene.json`.
* `inputs` loads the local GT, applies the case's `recipe.json`, saves
  `/Game/Code4SceneInputs/<case>/Input`, and exports
  `snapshots/input/<case>.scene.json`.
* `verify` compares the snapshots with the shipped fingerprints and writes
  `reports/verify.txt` and `reports/verify.json`.
* `package` writes the dataset as two trees: `agent/` with what an agent may
  see (the task prompt, the case facts and, after `render`, the reference
  views) and `scorer/` with everything else (see section 8).

Existing levels are kept; pass `--force` to rebuild them. `--dry-run` writes
the job files under `code4scene-dataset/jobs/` and prints the editor commands
without running them (to run one by hand, set `C4S_JOB` to its
`jobs/<step>.job.json` and `C4S_UE_DIR` to `dataset_builder/ue`, as printed);
it does not change the `.uproject` and writes no levels, reports or packaged
files. Each editor job logs to `jobs/<step>.log` and records a
per-task result in `jobs/<step>.result.json`. The builder exits with a
non-zero status if an editor job produced no result, a task failed or a
reference view is missing, and lists those problems at the end. If the editor
cannot run a level-editing script under `-ExecutePythonScript` on your
platform, try `--mode commandlet` (uses `-run=pythonscript`).

## 6. Render the reference views (GPU)

```bash
python -m dataset_builder.build --project ... --dataset ./code4scene-dataset --steps render --allow-missing-packs
```

For every image-to-scene case the GT level is opened, a temporary
SceneCapture2D is placed at each view in `cameras.json` (with the recorded
field of view, fill and top lights and exposure bias), and the capture is
written to `work/references_raw/<setting>/<case>/`. The images are then resized
to the published size and written into the agent tree,
`agent/<setting>/<case>/references/` (`view-01.png`, `view-02.png` for indoor
cases, 1280x720 or 1920x1080 as recorded; `reference.jpg` at 1600x900 for
outdoor cases, encoded with `ffmpeg -q:v 3` when ffmpeg is available). The
level is never saved during rendering.

Each GT level is rendered in its own editor process, and only after the editor
has drawn `--render-settle-ticks` frames (default 400, and at least
`--render-settle-seconds`, default 20) so that shaders compile and textures
stream in. A capture taken before the editor has ticked is black. A nearly
black frame is reported as a `WARNING`; render again with a larger value. If
one level's editor fails, the other levels are still rendered and published.

Rendered pixels will not be bit-identical to the images the benchmark agents
saw (GPU, driver and texture streaming differ). The pose, lens, lighting and
resolution are the same.

## 7. Reading the verification report

`reports/verify.txt` lists every GT level and Input level as `MATCH` or
`MISMATCH`, and for Inputs whether the level is consistent with its recipe.

| Result | Meaning | What to do |
|---|---|---|
| GT `MATCH` | the local GT level has exactly the benchmark's actors, assets, materials, labels, tags, properties and placements | nothing |
| GT `MISMATCH`, actors **missing** | the stable IDs are derived from the demo map's actor object names; a different pack version (or an edited/re-saved demo map) renames or drops actors | install the pack version listed in `packs.yaml`; do not edit pack maps |
| GT `MISMATCH`, **extra** actors | the installed pack adds actors, or a project plugin spawns actors on load | same as above; disable project plugins that add actors |
| **identity** facet | class, label, mesh assets or tags differ | usually a pack update that swapped meshes |
| **materials** facet | a material assignment differs | a pack update, or a material dependency that failed to load |
| **transform** facet | a placement moved by more than float noise | a pack update |
| Input `MISMATCH` but recipe consistency `match` | the Input was built correctly from a GT that itself differs | fix the GT first |
| Input `MISMATCH` and recipe consistency `mismatch` | the editor did not apply the recipe as specified | see `jobs/inputs.result.json`; please report it |

Scores obtained on a mismatching level are not comparable with the published
results. The report lists at most 50 differing actors per level, by stable ID
and label (taken from your own level).

Some pack demo maps reference content that is not part of the pack
(`known_unresolved_dependencies` in `scene.json`, for example a texture from
another pack). They were unresolved in the benchmark's scoring environment as
well, so they do not affect comparability.

## 8. Using the built dataset

The dataset directory is split so that an agent under test can be given what
it needs without the answers:

| Path | Who may see it | Contents |
|---|---|---|
| `agent/<setting>/<case>/prompt.txt` | the agent | the task prompt from `task.yaml` (a harness adds its own instructions around it) |
| `agent/<setting>/<case>/case.json` | the agent | case id, setting, start level, budget, packs, reference-view list (and the plate size for text-to-scene) |
| `agent/<setting>/<case>/references/` | the agent | the rendered reference views (image-to-scene) |
| `scorer/<setting>/<case>/` | scorer only | `task.yaml`, `task.label.json`, `recipe.json`, `cameras.json`, `input.fingerprint.json`, and the requirement bundle for text-to-scene |
| `snapshots/`, `reports/`, `jobs/`, `work/` | scorer only | builder outputs: exported GT and Input snapshots, verification reports, editor job files, raw renders |

The recipe, the fingerprints, the cameras and the GT snapshots are enough to
reconstruct the ground truth. **Never expose `scorer/`, `snapshots/`, `jobs/`,
`work/`, this repository's `benchmark/` directory or the `Code4SceneGT` content
to an agent under test**, including through a shell, a file tool or a shared
volume. The paths inside `task.yaml` refer to the levels you built
(`/Game/Code4SceneInputs/...` for the agent's start level,
`/Game/Code4SceneGT/...` for the scorer's ground truth).

`Code4SceneGT` is scorer-only content. When you run agents, give the agent an
editor instance whose project contains the packs and `Code4SceneInputs` but
not `Code4SceneGT` (`case.json` lists exactly the packs to mount; the GT root
is mounted only for scoring).

## How the levels are identified

The scorer matches the saved scene against GT actor by actor through a
*stable ID*. For tagged scenes the builder writes it into the actor's tags
(`simcodearena.stable_actor_id=<id>`, plus `simcodearena.actor_origin=source`)
before saving the GT copy. Three identity schemes occur, all fully
determined by `scene.json` and the installed demo map:

* `uuid5-source-v1` (most scenes): a UUID5 of the scene's `gt_id`, the source
  map and the actor's object path and class in the source map.
* `sha256-source-v1` (`dungeon-hall`): a SHA-256 of the source map, the actor's
  object name and class.
* `untagged` (`laboratory-loft`, `victorian-dining`): no tags; the snapshot
  exporter derives `sca_generated_` IDs from object name, class and component
  assets. Child actors and actors in streamed sublevels use this fallback in
  every scene, as in the benchmark.

Actors a recipe spawns keep the object name recorded in the recipe, so their
fallback IDs match; if the engine refuses the rename, the builder writes the
expected ID as an explicit tag instead (the fingerprint does not depend on
which of the two carries the ID).

## Content fingerprints

A fingerprint summarises one saved level without describing its arrangement.
It is computed on the host from the exported snapshot
(`dataset_builder/fingerprint.py`, algorithm `c4s-fp-1`):

* One **leaf** per actor, keyed by stable ID. The leaf is a SHA-256 over four
  facet digests:
  * *identity*: class path, actor label, component mesh assets, actor tags
    (except the stable-ID tag), benchmark origin/role/logical-object tags,
    and the attachment parent (as its stable ID);
  * *materials*: every (component, slot, material) triple; dynamic material
    instances created at load time are reduced to their class, because their
    object paths change with the map name and between sessions;
  * *transform*: location quantised to 0.5 cm, the rotation as its 3x3
    rotation matrix quantised to 5e-4 (about 0.03 degrees; the matrix makes
    equivalent rotators such as yaw 180 and -180 agree), and scale quantised
    to 5e-4;
  * *properties*: the light, fog and post-process properties the scorer
    records, floats rounded to 6 significant digits.
* The **root** is a SHA-256 over the sorted `"<stable_id> <leaf>"` lines, so it
  does not depend on actor order.
* Shipped expectations contain the root, the actor count and, per stable ID,
  a 16-hex-digit leaf and 16-hex-digit facet digests. Input expectations store
  only the actors that differ from the GT. No transform, label or asset path is
  shipped; a mismatch report shows labels from your own level.
* **Float tolerance.** A value within float noise of a rounding boundary would
  hash differently on two machines, so the comparison re-hashes an actor with
  the neighbouring bucket for each quantised coordinate near a boundary. As a
  result, differences below 0.1 cm, about 0.006 degrees and 1e-4 in scale
  always match, and differences above one quantum never do. The scorer's own
  tolerances (5 cm, 5 degrees, 5 %) are far coarser.

Every saved copy of the benchmark's levels that exists from the scoring runs
was fingerprinted when these files were produced. All copies of each GT level
agree, and all copies of each Input level agree except for the middle-east
Inputs, which were changed once during the benchmark's preparation; the
shipped expectation is the later version.

## Scene-specific notes

* `middle-east-river`: the GT is the pack's `L_MiddleEasternTown` with the
  streamed sublevel `L_CityBuildings` replaced by a small new sublevel
  (`buildings_gt`, 13 placed level instances; the placements are part of
  `scene.json`) and one actor label changed. Each Input streams a copy of that
  sublevel (`/Game/Code4SceneInputs/_shared/middle-east-river/buildings_base`)
  instead, so that no Input references the GT root. Two actors of that
  sublevel get scope-dependent fallback IDs, so they appear under different
  IDs in GT and Input; this reproduces the benchmark exactly.
* `laboratory-loft`: the GT differs from the stock demo map in one actor label
  and one light property on two lamps; `scene.json` applies both.
* `high-school-classroom`: the GT differs from the stock demo map in two places,
  both applied by `scene.json`: the clock Blueprint, which references a curve
  asset the pack does not ship, is replaced by the clock mesh as a static actor
  at the same transform, and two rear chairs are set to a mirror-symmetric pose.
* Some indoor Input levels contain an extra actor whose label begins with
  `SB_ATOMIC20_EXTRA__` or `SB_ATOMIC80_EXTRA__`; that is the label the
  benchmark's Input levels used and it is reproduced as is.

## Time estimates

Rough figures for a workstation with a fast SSD; the first run of each pack is
dominated by asset loading and derived-data (DDC) generation:

| Step | Estimate |
|---|---|
| `check`, `verify`, `package` | seconds to a few minutes |
| `blank` | about 1 minute (editor start-up) |
| `gt` (all scenes) | 1 to 3 hours on a cold DDC; most scenes take minutes, `middle-east-river` (about 6,700 actors) takes longest |
| `inputs` (all image-to-scene cases) | 1 to 3 hours; each case loads its GT once |
| `render` (all image-to-scene cases, GPU) | 1 to 4 hours, dominated by shader compilation on first use of each pack |

For reference, a CPU-only Linux server (`-NullRHI`, cold local DDC) built the
GT levels of three small indoor scenes in about 2 minutes including editor
start-up, and their 11 Input levels in under a minute.

## Troubleshooting

* *The editor hangs at start-up with 0 % CPU*: on some Linux hosts the
  editor's platform-SDK probe (a child `Build.sh ... -Mode=ValidatePlatforms`)
  never returns. Stop that child process (its PID is a child of the PID in
  `jobs/<step>.pid`) and the editor continues. Extra editor flags can be
  passed with `--editor-arg`, for example `--editor-arg=-notrace
  --editor-arg=-NoUba --editor-arg=-AssetRegistry.DisableDirectoryWatcher=1`,
  or `--editor-arg=-LocalDataCachePath=<dir>` to keep the DDC out of your
  home directory.
* *The editor produced no result file*: open `jobs/<step>.log`. The usual
  causes are a missing plugin (step 3), the project being open in another
  editor, or a wrong `UE_EDITOR` path.
* *target ... is not in the level (pack version mismatch?)* in
  `jobs/inputs.result.json`: the local GT does not contain an actor the recipe
  edits; the GT verification will show why.
* *materials differ after mesh swap* notes: the pack's mesh has different
  default materials than the benchmark's copy; verification will report a
  materials mismatch for that actor.
