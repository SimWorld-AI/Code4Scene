# Code4Scene

**Benchmarking coding agents for constructing and editing 3D scenes.**

Code4Scene evaluates coding agents that operate Unreal Engine: they write and run code, inspect the result and revise the scene. It does not score the code or a rendered image. It scores the engine-native scene (`.umap`) the agent saves, on task fulfilment, artifact integrity and static physical validity. Edits are also compared against withheld ground truth.

It has two settings:

- **Text-to-Scene Construction.** Starting from an empty level, the agent builds a scene from an open-ended language description and the pack's asset catalog. Many realizations are valid.
- **Image-to-Scene Editing.** The agent gets a corrupted copy of a human-assembled scene plus reference views of the original. It must restore the intended state (within 5 cm · 5° · 5%) and leave everything else untouched.

This repository accompanies the paper. It contains:

| | |
|---|---|
| `code4scene/` | The verifiers and the paper's scoring protocol, as a Python package with a `code4scene` CLI. |
| `benchmark/public/` | Task definitions for the **public set**: prompts, requirement bundles, edit recipes, reference-view cameras and content fingerprints. |
| `benchmark/packs.yaml`, `docs/PACKS.md` | Where to get each Unreal Engine content pack the public set uses (Fab links). |
| `dataset_builder/`, `docs/BUILD_DATASET.md` | A builder that recreates the public dataset from those packs on a stock Unreal Engine 5.8 editor, then verifies it against the fingerprints. |

The private set is not included.

## No third-party content is distributed

The scenes are built from third-party content packs sold or given away on [Fab](https://www.fab.com). This repository ships none of them. It contains no levels, meshes, textures or rendered images, and no transform-level description of any seller's scene. You obtain each pack from its original listing under its own license, then run the builder. The builder recreates:

- the ground-truth levels from the packs' own demo maps;
- the corrupted input levels, from small edit recipes;
- the reference views.

A content fingerprint confirms that your build matches the levels the benchmark was scored on, and tells you which actors differ if it does not.

## Install

```bash
pip install -e .            # Python ≥ 3.10; add [dev] for the test suite
code4scene --help
```

## Build the public dataset

1. Create an empty UE 5.8 project.
2. Install the packs listed in [docs/PACKS.md](docs/PACKS.md).
3. Run the builder:

```bash
python -m dataset_builder.build --project /path/Code4SceneData/Code4SceneData.uproject \
    --dataset ./code4scene-dataset --steps check blank gt inputs verify package
python -m dataset_builder.build --project ... --dataset ./code4scene-dataset --steps render   # reference views, needs a GPU
```

[docs/BUILD_DATASET.md](docs/BUILD_DATASET.md) covers disk and time estimates, the verification report, and scene-specific notes.

## Score a scene

The verifiers read an **evidence bundle**: scene snapshots, physics measurements and renders, exported from the saved level by the editor scripts in `code4scene/ue_scripts/`. See [docs/EVIDENCE_BUNDLE.md](docs/EVIDENCE_BUNDLE.md).

```bash
code4scene score path/to/bundle --task benchmark/public/<setting>/<case>/task.yaml   # full scoring (needs the VLM judge)
code4scene score path/to/bundle --task .../task.yaml --no-vlm                      # structured leaves only
code4scene aggregate scores/ --schedule benchmark --format csv -o model-scores.csv  # case scores → model score
```

Text-to-Scene semantic and overview judgments use a VLM judge. The paper used **Qwen3.8-27B**; point the CLI at your own OpenAI-compatible deployment with `CODE4SCENE_VLM_BASE_URL` and `CODE4SCENE_VLM_MODEL`.

[docs/SCORING.md](docs/SCORING.md) gives every verifier, formula and zero rule. In short:

- **Image-to-Scene case:** `0.8 · Repair F1 + 0.2 · Physical Safety`.
- **Model score:** `0.5 · S_T2S + 0.5 · S_I2S`.

Scores are comparable to the paper's only when agents run under the same interface, asset catalog and budget.

## Tests

```bash
pip install -e .[dev] && pytest
```

The suite runs fully offline.

## License

The code is released under the MIT License ([LICENSE](LICENSE)). The content packs are not part of this repository and remain under their own Fab licenses.

## Citation

```bibtex
@article{code4scene2026,
  title   = {Code4Scene: Benchmarking Coding Agents for Constructing and Editing 3D Scenes},
  author  = {Ye, Xiaokang and Mantri, Siddhant Hitesh and Chen, Zimeng and Zhang, Edward and Zheng, Zhaoxu and Li, Yuanheng and Chen, Yizhao and Huang, Tianyang and Qin, Lianhui},
  year    = {2026}
}
```
