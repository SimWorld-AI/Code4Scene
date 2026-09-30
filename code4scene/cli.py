"""``code4scene`` command line.

Subcommands
-----------
``score <bundle_dir> --task <task.yaml>``
    Score one evidence bundle offline under the paper protocol. VLM-dependent
    leaves are re-judged by the configured judge (``--vlm-base-url``);
    ``--no-vlm`` scores only structured leaves and marks the rest as not
    evaluated; ``--judge recorded`` re-aggregates the judge outputs stored in
    the bundle without calling a model.
``rescore <result.json>``
    Re-derive the paper case score from a saved verifier result (and, for
    image-to-scene, its scene snapshots). No model is called.
``aggregate <results...>``
    Case scores -> setting means -> model score (0.5 * T2S + 0.5 * I2S).
``make-bundle <result.json> <out_dir>``
    Pack a saved verifier result and its snapshots as a ``code4scene.bundle.v1``.
``validate-bundle <bundle_dir>``
    Check a bundle's manifest, relative paths and file hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEDULE_FILES = {
    "text-to-scene": "public-t2s-cases.txt",
    "image-to-scene/indoor": "public-indoor-cases.txt",
    "image-to-scene/outdoor": "public-outdoor-cases.txt",
}
SETTINGS_CHOICES = tuple(SCHEDULE_FILES)
AGGREGATE_FIELDS = [
    "model_rank", "i2s_rank", "model", "model_score", "model_score_100", "t2s_score",
    "i2s_score", "i2s_score_100", "indoor_i2s_score", "outdoor_i2s_score",
    "indoor_f1_case_macro", "outdoor_f1_case_macro", "i2s_f1_case_macro",
    "i2s_precision_case_macro", "i2s_recall_case_macro", "i2s_physics_case_macro",
    "indoor_physics_case_macro", "outdoor_physics_case_macro", "t2s_cases", "indoor_cases",
    "outdoor_cases", "t2s_zero_cases", "indoor_zero_cases", "outdoor_zero_cases",
    "unresolved_cases", "i2s_policy", "model_policy",
]


def _configure_judge(args: argparse.Namespace) -> None:
    """CLI flags win over the environment; set them before evaluation imports."""

    if getattr(args, "vlm_base_url", None):
        os.environ["CODE4SCENE_VLM_BASE_URL"] = args.vlm_base_url
    if getattr(args, "vlm_model", None):
        os.environ["CODE4SCENE_VLM_MODEL"] = args.vlm_model
    if getattr(args, "embed_base_url", None):
        os.environ["CODE4SCENE_EMBED_BASE_URL"] = args.embed_base_url


def _write(value: Any, out: str | None) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    if out:
        Path(out).write_text(text)
    else:
        sys.stdout.write(text)


def _load_task(path: str | None):
    if not path:
        return None
    from .tasks import task as task_mod

    return task_mod.load(path)


# ---------------------------------------------------------------------------
# score / rescore / bundles
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    _configure_judge(args)
    from . import bundle as bundle_mod
    from . import scoring

    judge = "none" if args.no_vlm else args.judge
    loaded = bundle_mod.load(args.bundle_dir, verify=not args.no_verify)
    # Image-to-scene cases never use the judge, so only warn for text-to-scene.
    if judge == "live" and loaded.is_t2s:
        from .evaluation import vlm_model_config

        if not vlm_model_config.base_url():
            print("note: no judge endpoint configured; VLM-dependent leaves will fail. "
                  "Pass --vlm-base-url, set CODE4SCENE_VLM_BASE_URL, or use --no-vlm.",
                  file=sys.stderr)
    task = _load_task(args.task)
    record = scoring.score_bundle(loaded, task, judge=judge, include_audit=args.audit)
    _write(record, args.out)
    return 0


def _scene_paths(directory: Path) -> dict[str, Path]:
    scenes = {}
    for role in ("input", "ground_truth", "candidate"):
        for name in (f"{role}.scene.json", f"{role}.scene.json.gz"):
            if (directory / name).is_file():
                scenes[role] = directory / name
                break
    return scenes


def cmd_rescore(args: argparse.Namespace) -> int:
    from . import bundle as bundle_mod
    from . import scoring

    semantic = None
    if args.semantic_report:
        source = bundle_mod.read_json(Path(args.semantic_report))
        if isinstance(source, dict) and "reports" in source:
            source = {r["report_id"]: r for r in source["reports"]}["semantic_requirements"]
        semantic = source
    scenes = _scene_paths(Path(args.scenes_dir)) if args.scenes_dir else None
    with tempfile.TemporaryDirectory(prefix="code4scene-rescore-") as tmp:
        loaded = bundle_mod.from_result(args.result, Path(tmp) / "bundle", scenes=scenes,
                                        semantic_report=semantic, model=args.model,
                                        setting=args.setting, link=True)
        record = scoring.score_bundle(loaded, _load_task(args.task), judge="recorded",
                                      include_audit=args.audit)
        rebuilt = scoring.rebuild_result(bundle_mod.read_json(Path(args.result)), loaded,
                                         semantic_report=semantic)
    record["source"] = {"result": Path(args.result).name}
    record["verifier_layer"] = scoring.verifier_layer_summary(rebuilt, record)
    if args.result_out:
        _write(rebuilt, args.result_out)
    _write(record, args.out)
    return 0


def cmd_make_bundle(args: argparse.Namespace) -> int:
    from . import bundle as bundle_mod

    semantic = None
    if args.semantic_report:
        source = bundle_mod.read_json(Path(args.semantic_report))
        if isinstance(source, dict) and "reports" in source:
            source = {r["report_id"]: r for r in source["reports"]}["semantic_requirements"]
        semantic = source
    scenes = _scene_paths(Path(args.scenes_dir)) if args.scenes_dir else None
    made = bundle_mod.from_result(args.result, args.out_dir, scenes=scenes,
                                  semantic_report=semantic, model=args.model,
                                  setting=args.setting, link=args.link,
                                  compress=args.compress)
    print(json.dumps({"bundle": str(made.root), "task_id": made.task_id,
                      "setting": made.setting, "files": len(made.manifest["files"])}))
    return 0


def cmd_validate_bundle(args: argparse.Namespace) -> int:
    from . import bundle as bundle_mod

    loaded = bundle_mod.load(args.bundle_dir)
    print(json.dumps({"valid": True, "schema_version": bundle_mod.SCHEMA_VERSION,
                      "task_id": loaded.task_id, "setting": loaded.setting,
                      "files": len(loaded.manifest["files"])}))
    return 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


#: What a case row needs besides a model label (which --model may supply).
ROW_FIELDS = ("setting", "case_id", "score")
RESULT_SUFFIXES = (".json", ".jsonl", ".csv")


def _rows_from_file(path: Path, *, in_directory: bool = False) -> list[dict[str, Any]]:
    """Case rows from one file; a file in a results folder that holds none is skipped."""

    from . import scoring

    text = path.read_text()
    if path.suffix == ".csv":
        reader = csv.DictReader(text.splitlines())
        missing = [f for f in ROW_FIELDS if f not in (reader.fieldnames or ())]
        if missing and in_directory:
            print(f"note: skipped {path}: not a case score table", file=sys.stderr)
            return []
        if missing:
            raise ValueError(f"{path}: CSV lacks the column(s) {', '.join(missing)}")
        return [dict(r) for r in reader]
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value if isinstance(value, list) else [value]
    rows = []
    for value in values:
        if isinstance(value, dict) and value.get("schema_version") == scoring.SCHEMA_VERSION:
            rows.append(scoring.case_row(value))
        elif isinstance(value, dict) and all(f in value for f in ROW_FIELDS):
            rows.append(value)
        elif in_directory:
            print(f"note: skipped {path}: not a case score", file=sys.stderr)
            return []
        else:
            raise ValueError(f"{path}: not a case score (a code4scene score record or a row "
                             f"with {', '.join(ROW_FIELDS)})")
    return rows


def _schedule(args: argparse.Namespace) -> dict[str, list[str]] | None:
    schedule: dict[str, list[str]] = {}
    if args.schedule:
        root = Path(args.schedule)
        missing = [name for name in SCHEDULE_FILES.values() if not (root / name).is_file()]
        if missing:
            raise ValueError(f"--schedule {root} lacks {', '.join(missing)}")
        for setting, name in SCHEDULE_FILES.items():
            schedule[setting] = (root / name).read_text().split()
    for setting, value in (("text-to-scene", args.t2s_cases),
                           ("image-to-scene/indoor", args.indoor_cases),
                           ("image-to-scene/outdoor", args.outdoor_cases)):
        if value:
            schedule[setting] = Path(value).read_text().split()
    return schedule or None


def cmd_aggregate(args: argparse.Namespace) -> int:
    from .protocol import aggregate as agg

    rows: list[dict[str, Any]] = []
    for item in args.results:
        path = Path(item)
        if path.is_dir():
            for file in sorted(p for p in path.rglob("*") if p.suffix in RESULT_SUFFIXES):
                rows.extend(_rows_from_file(file, in_directory=True))
        else:
            rows.extend(_rows_from_file(path))
    if not rows:
        raise ValueError(f"no case scores found in {', '.join(args.results)}")
    if args.model:
        rows = [dict(r, model=args.model) for r in rows]
    if any(not r.get("model") for r in rows):
        raise ValueError("a case row has no model label; pass --model")
    schedule = _schedule(args)
    if schedule is None:
        print("note: no case lists given; each setting is averaged over the cases present, "
              "which is not the paper's schedule", file=sys.stderr)
    else:
        listed = {agg.normalize_setting(s) for s in schedule}
        unlisted = sorted({agg.normalize_setting(r["setting"]) for r in rows} - listed)
        if unlisted:
            raise ValueError(f"rows for {', '.join(unlisted)} but no case list for them; "
                             "pass --schedule or the matching --*-cases file")
    models = agg.aggregate(rows, schedule=schedule,
                           missing_as_zero=not args.no_missing_as_zero)
    if args.format == "csv":
        stream = open(args.out, "w", newline="") if args.out else sys.stdout
        writer = csv.DictWriter(stream, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for r in models:
            writer.writerow({k: (";".join(v) if isinstance(v, list) else
                                 "" if v is None else v) for k, v in r.items()})
        if args.out:
            stream.close()
    else:
        _write(models, args.out)
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code4scene", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def judge_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vlm-base-url", help="OpenAI-compatible judge endpoint "
                       "(default: $CODE4SCENE_VLM_BASE_URL)")
        p.add_argument("--vlm-model", help="served judge model name "
                       "(default: $CODE4SCENE_VLM_MODEL or Qwen/Qwen3.8-27B)")
        p.add_argument("--embed-base-url", help="embedding endpoint for report-only "
                       "caption similarity (default: $CODE4SCENE_EMBED_BASE_URL)")

    p = sub.add_parser("score", help="score one evidence bundle offline")
    p.add_argument("bundle_dir")
    p.add_argument("--task", help="the case's task.yaml (checked against the bundle)")
    p.add_argument("--judge", choices=("live", "recorded", "none"), default="live")
    p.add_argument("--no-vlm", action="store_true",
                   help="score structured leaves only; VLM leaves are not evaluated")
    p.add_argument("--no-verify", action="store_true", help="skip bundle hash checks")
    p.add_argument("--audit", action="store_true", help="include the Actor F1 audit")
    p.add_argument("--out", "-o")
    judge_flags(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("rescore", help="re-derive the paper score from a saved result.json")
    p.add_argument("result")
    p.add_argument("--scenes-dir", help="directory with {input,ground_truth,candidate}"
                   ".scene.json (default: the result's scene_evidence directory)")
    p.add_argument("--semantic-report", help="result.json or report whose Semantic "
                   "report replaces the one in the result")
    p.add_argument("--task")
    p.add_argument("--model", help="model label to record")
    p.add_argument("--setting", choices=SETTINGS_CHOICES,
                   help="override the setting when the result does not record it")
    p.add_argument("--result-out", help="also write the result.json rebuilt through the "
                   "verifier layer (its overall_score is the paper case score)")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--out", "-o")
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("aggregate", help="case scores -> model score")
    p.add_argument("results", nargs="+", help="case-score JSON/JSONL files, directories "
                   "of them, or CSV files with model,setting,case_id,score columns")
    p.add_argument("--schedule", help="directory with public-{t2s,indoor,outdoor}-cases.txt")
    p.add_argument("--t2s-cases")
    p.add_argument("--indoor-cases")
    p.add_argument("--outdoor-cases")
    p.add_argument("--model", help="override the model label of every row")
    p.add_argument("--no-missing-as-zero", action="store_true",
                   help="leave a scheduled case without a row unresolved instead of zero")
    p.add_argument("--format", choices=("json", "csv"), default="json")
    p.add_argument("--out", "-o")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("make-bundle", help="pack a saved result as an evidence bundle")
    p.add_argument("result")
    p.add_argument("out_dir")
    p.add_argument("--scenes-dir")
    p.add_argument("--semantic-report")
    p.add_argument("--model")
    p.add_argument("--setting", choices=SETTINGS_CHOICES)
    p.add_argument("--link", action="store_true", help="symlink evidence instead of copying")
    p.add_argument("--compress", action="store_true", help="store scenes as .json.gz")
    p.set_defaults(func=cmd_make_bundle)

    p = sub.add_parser("validate-bundle", help="check a bundle")
    p.add_argument("bundle_dir")
    p.set_defaults(func=cmd_validate_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, OSError) as exc:
        print(f"code4scene {args.command}: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - the command line reports; it does not dump a traceback
        print(f"code4scene {args.command}: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
