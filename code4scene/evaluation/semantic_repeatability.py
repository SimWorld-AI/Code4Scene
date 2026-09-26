"""Conditional judge repeatability on frozen T2S Stage 3 request schedules.

Earlier deterministic decisions, evidence acquisition and historical request
routing are held fixed. Every saved logical batch (including saved binary
arbitration) is issued in every repetition. This measures conditional judge
variation, not end-to-end adaptive verifier variation. Source results are never
modified. Preparation has no network client; running requires an explicit command.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from . import vlm_model_config as config
from .vlm_concurrency import runtime_config
from .requirement_graph.contracts import RequirementGraph
from .requirement_graph.stage2_contracts import Stage2Task, Stage2TaskArgument
from .requirement_graph.stage2_routing import visual_claim_payload
from .requirement_graph.stage3 import (
    _aggregate_stage3_batches,
    finalize_stage3_binary_decision,
    finalize_stage3_unknown_decision,
)
from .requirement_graph.stage3_judge import (
    LLMStage3UnknownJudge,
    Stage3JudgeFrame,
    Stage3UnknownDecision,
)
from .requirement_graph.vlm_client import tool_client_from_env
from .semantic_scoring import attach_semantic_families, score_semantic_case

PROTOCOL = "t2s-frozen-stage3-request-schedule.v2"
TEMPLATE_FIELDS = (
    "system_text", "payload_text", "schema", "call_kind", "tool_name",
    "maximum_evidence_frames", "enforce_claim_grounding",
)


def read(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): file_hash(p)
            for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}


def judge_config():
    # Credentials are deliberately never included.
    values = {k: getattr(config, k) for k in (
        "BACKEND", "BASE_URL", "MODEL", "MAX_TOKENS", "TIMEOUT_S",
        "TEMPERATURE", "SEED", "ENABLE_THINKING",
    )}
    values["runtime"] = runtime_config().to_dict()
    return values


def checked_json(path, expected_hash=None):
    if expected_hash and file_hash(path) != expected_hash:
        raise ValueError(f"source hash changed: {path}")
    return read(path)


def task_from_dict(value):
    return Stage2Task(**{
        **value, "arguments": tuple(Stage2TaskArgument(**a)
                                   for a in value.get("arguments", [])),
    })


def template_spec(kwargs):
    return {k: kwargs[k] for k in TEMPLATE_FIELDS}


class TemplateJudge(LLMStage3UnknownJudge):
    """Compile production messages/schema without ever calling a transport."""
    compiled = None

    def _judge_prepared(self, **kwargs):
        self.compiled = template_spec(kwargs)
        return Stage3UnknownDecision()


class FrozenJudge(LLMStage3UnknownJudge):
    """Refuse a changed prompt or schema before the production client is called."""
    expected_template = None

    def _judge_prepared(self, **kwargs):
        if template_spec(kwargs) != self.expected_template:
            raise ValueError("production request template differs from prepared input")
        return super()._judge_prepared(**kwargs)


def compile_template(payload, binary=False):
    compiler = TemplateJudge(SimpleNamespace(model=config.MODEL),
                             max_tokens=config.MAX_TOKENS,
                             max_validation_retries=0)
    # Text and schema depend on the claim and configured frame limit, not pixels.
    dummy = Stage3JudgeFrame("s3f_000001", np.zeros((1, 1, 3), dtype=np.uint8))
    method = compiler.judge_binary if binary else compiler.judge
    decision = method(payload, (dummy,))
    if compiler.compiled is None:
        raise ValueError(f"could not compile production request: {decision.error}")
    return compiler.compiled


def authoritative_rows(bundle, report):
    """Reconstruct raw per-requirement scores, never use effective parent scores."""
    metrics = report["metrics"]
    bindings = {r["node_id"]: r for r in bundle["requirements"]}
    evaluations = {r["node_id"]: r for r in bundle["evaluations"]}
    checks = {r["id"]: r for r in metrics["checks"]}
    rows = []
    for audit in metrics["semantic_requirement_aggregation"]:
        node = audit["node_id"]
        binding = bindings[node]
        check = checks[binding["requirement_id"]]
        rows.append({
            "node_id": node, "requirement_id": binding["requirement_id"],
            "text": binding["source_text"], "source_span": binding["source_span"],
            "predicate_type": audit["predicate_type"],
            "evaluation_status": audit["evaluation_status"],
            "entity_scopes": binding.get("entity_scopes", {}),
            "evaluation_binding": evaluations[node],
            "score": check.get("score"),
            "historical_resolved_by": (check.get("observed") or {}).get("resolved_by"),
        })
    if len({r["node_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate authoritative score row")
    return attach_semantic_families(bundle["graph"], rows)


def load_frame(spec, *, verify=True):
    path = Path(spec["path"])
    if verify and file_hash(path) != spec["png_sha256"]:
        raise ValueError(f"frozen frame file changed: {path}")
    with Image.open(path) as image:
        image.load()
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, order="C")
    actual = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
    if actual != spec["rgb_sha256"]:
        raise ValueError(f"frozen RGB hash changed: {path}")
    return Stage3JudgeFrame(spec["frame_id"], rgb)


def schedule_sources(report, primary, bundle):
    """Resolve final score provenance; frame IDs are local to each source run."""
    sources = [{"id": "primary", "path": Path(primary), "only_nodes": None}]
    recovery = report.get("evidence", {}).get("incremental_unknown_recovery")
    if recovery:
        artifact = report["artifacts"]["incremental_unknown_recovery"]
        recovered = checked_json(artifact)
        filled = set(recovery["filled_node_ids"])
        if filled != {x["node_id"] for x in recovered["fills"]}:
            raise ValueError("incremental recovery node provenance mismatch")
        sources.append({
            "id": "recovery",
            "path": Path(report["artifacts"]["incremental_unknown_replacement_graph"]),
            "only_nodes": filled,
        })
    selected = {}
    provenance = []
    for source in sources:
        root = source["path"]
        source_bundle = read(root / "bundle.json")
        if json_hash(source_bundle["graph"]) != json_hash(bundle["graph"]):
            raise ValueError(f"replay source uses a different requirement graph: {root}")
        stage2_manifest = root / "stage2/stage2_vlm_request_manifest.json"
        if stage2_manifest.exists() and read(stage2_manifest):
            raise ValueError("Stage 2 VLM judgments are not supported by this protocol")
        schedule_path = root / "stage3/stage3_batch_results.json"
        tasks_path = root / "stage2/stage2_tasks.json"
        manifest_path = root / "stage3/stage3_vlm_request_manifest.json"
        index_path = root / "stage3/stage3_frame_index.json"
        schedule = read(schedule_path)
        tasks = {x["node_id"]: x for x in read(tasks_path)}
        manifest = {x["request_id"]: x for x in read(manifest_path)}
        index = {x["frame_id"]: x for x in read(index_path)["frames"]}
        mapped = set()
        covered = set()
        for claim in schedule["claims"]:
            calls = [{
                "historical_request_id": b["request_id"], "kind": "unknown_resolution",
                "frame_ids": b["frame_ids"],
            } for b in claim["batches"]]
            binary = claim.get("binary_arbitration") or {}
            if binary.get("attempted"):
                calls.append({
                    "historical_request_id": binary["request_id"],
                    "kind": "binary_resolution", "frame_ids": binary["selected_frame_ids"],
                })
            for call in calls:
                rid = call["historical_request_id"]
                if rid in mapped:
                    raise ValueError(f"duplicate request in source {source['id']}: {rid}")
                mapped.add(rid)
            node = claim["node_id"]
            if source["only_nodes"] is not None and node not in source["only_nodes"]:
                continue
            covered.add(node)
            selected[node] = {
                "claim": claim, "task": tasks[node], "calls": calls,
                "source_id": source["id"], "root": root,
                "manifest": manifest, "index": index,
            }
        if mapped != set(manifest):
            raise ValueError(f"unmapped historical VLM calls in source {root}")
        if source["only_nodes"] is not None and covered != source["only_nodes"]:
            raise ValueError("incremental recovery is missing scored claim requests")
        provenance.append({
            "id": source["id"], "graph_dir": str(root),
            "node_filter": sorted(source["only_nodes"]) if source["only_nodes"] else None,
            "schedule_sha256": file_hash(schedule_path),
            "tasks_sha256": file_hash(tasks_path),
            "request_manifest_sha256": file_hash(manifest_path),
            "frame_index_sha256": file_hash(index_path),
        })
    return list(selected.values()), provenance


def prepare_candidate(row, *, frame_workers=4):
    result = checked_json(row["result_path"], row["result_sha256"])
    bundle = checked_json(row["frozen_bundle"], row["semantic_bundle_sha256"])
    report = next(r for r in result["reports"]
                  if r["report_id"] == "semantic_requirements")
    if report["status"] != "measured":
        raise ValueError("requires a measured source Semantic report")
    for manifest in row["semantic_request_manifests"]:
        if manifest["stage"] == "stage2" and manifest["count"]:
            raise ValueError("Stage 2 VLM judgments are not supported by this protocol")
    graph = RequirementGraph.from_dict(bundle["graph"])
    source = Path(row["semantic_bundle"]).parent
    scheduled, provenance = schedule_sources(report, source, bundle)
    scores = authoritative_rows(bundle, report)
    score_nodes = {v["node_id"] for v in scores}
    weights = report["metrics"]["semantic_family_weights"]
    baseline = score_semantic_case(bundle["graph"]["prompt"], scores,
                                   family_weights=weights)
    # Changes in parent/dedup policy are visible, rather than hidden as judge noise.
    historical_delta = round(baseline["score"] - report["score"], 6)
    frame_specs = {}
    claims = []
    used_requests = set()
    for item in scheduled:
        c = item["claim"]
        node = c["node_id"]
        if node not in score_nodes:
            raise ValueError(f"unmapped replay claim: {node}")
        payload = visual_claim_payload(graph, node)
        calls = item["calls"]
        binary = c.get("binary_arbitration") or {}
        for call in calls:
            request_id = call["historical_request_id"]
            request_key = (item["source_id"], request_id)
            if request_key in used_requests:
                raise ValueError(f"duplicate request ID: {request_id}")
            used_requests.add(request_key)
            recorded = item["manifest"][request_id]
            if (recorded["call_kind"] != call["kind"]
                    or recorded["frame_ids"] != call["frame_ids"]):
                raise ValueError(f"historical request schedule mismatch: {request_id}")
            if not call["frame_ids"] or len(set(call["frame_ids"])) != len(call["frame_ids"]):
                raise ValueError(f"empty or duplicate image list: {request_id}")
            call["source_id"] = item["source_id"]
            call["frame_keys"] = [f"{item['source_id']}:{fid}" for fid in call["frame_ids"]]
            for image in recorded["images"]:
                fid = image["frame_id"]
                if item["index"][fid]["image_hash"] != image["sha256"]:
                    raise ValueError(f"recorded RGB hash mismatch: {request_id}")
                frame_specs[f"{item['source_id']}:{fid}"] = {
                    "frame_id": fid,
                    "path": str(item["root"] / "stage3/stage3_frames" / f"{fid}.png"),
                    "rgb_sha256": image["sha256"],
                }
        claims.append({
            "node_id": node, "task": item["task"], "payload": payload, "calls": calls,
            "unknown_template": compile_template(payload),
            "binary_template": compile_template(payload, binary=True)
            if binary.get("attempted") else None,
        })
    claim_nodes = {c["node_id"] for c in claims}
    if len(claim_nodes) != len(claims):
        raise ValueError("duplicate replay claim")
    unhandled = [r["node_id"] for r in scores
                 if str(r["historical_resolved_by"]).startswith("stage3")
                 and r["node_id"] not in claim_nodes]
    if unhandled:
        raise ValueError(f"Stage 3 rows not covered by replay: {unhandled}")
    needed = {fid for c in claims for call in c["calls"] for fid in call["frame_keys"]}
    def validate_frame(fid):
        spec = frame_specs[fid]
        spec["png_sha256"] = file_hash(spec["path"])
        load_frame(spec)  # Full PNG decode and independent recorded-RGB hash check.
    # Saved evidence can live on shared storage. Bound parallel reads, discard
    # decoded pixels immediately, and retain deterministic ordering in the plan.
    with ThreadPoolExecutor(max_workers=frame_workers) as pool:
        for _ in pool.map(validate_frame, sorted(needed)):
            pass
    return {
        "model_id": row["model_id"], "case_id": row["case_id"], "bundle": bundle,
        "source_result_path": row["result_path"],
        "source_result_sha256": row["result_sha256"],
        "schedule_sources": provenance,
        "historical_score": report["score"], "frozen_score_rows": scores,
        "current_code_historical_judgments_score": baseline["score"],
        "aggregation_delta_from_historical": historical_delta,
        "family_weights": weights, "claims": claims,
        "frames": {k: frame_specs[k] for k in sorted(needed)},
        "request_count": sum(len(c["calls"]) for c in claims),
        "fixed_non_stage3_row_count": len(scores) - len(claims),
    }


def prepare(selection, output, *, models=(), cases=(), repeats=3, workers=4):
    if repeats < 2 or not 1 <= workers <= 32:
        raise ValueError("repeats must be >=2 and workers between 1 and 32")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepare requires an empty output directory")
    selected = read(selection)["rows"]
    model_pool = {r["model_id"] for r in selected}
    case_pool = {r["case_id"] for r in selected}
    if set(models) - model_pool or set(cases) - case_pool:
        raise ValueError("unknown model/case filter")
    selected = [r for r in selected if (not models or r["model_id"] in models)
                and (not cases or r["case_id"] in cases)]
    if not selected:
        raise ValueError("no selected candidates")
    model_ids = sorted({r["model_id"] for r in selected})
    case_ids = sorted({r["case_id"] for r in selected})
    if len({(r["model_id"], r["case_id"]) for r in selected}) != len(selected):
        raise ValueError("duplicate selected candidate")
    if len(selected) != len(model_ids) * len(case_ids):
        raise ValueError("selection is not a complete model-by-case panel")
    before_code = code_hashes()
    metadata = {
        "protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": file_hash(selection), "code_hashes": before_code,
        "judge": judge_config(), "repeats": repeats, "workers": workers,
        "model_ids": model_ids, "case_ids": case_ids,
        "validation_retries": 0,
        "invalid_output_policy": "use the production judge's safe UNKNOWN fallback; count and retain every validation failure; transport or unresolved parse errors invalidate the case",
        "scope": "conditional on frozen Stage 0/1/2, historical claim routing and batches",
        "aggregation": "production type gates and batch conflict resolution; binary fallback only for UNKNOWN when a binary request was historically scheduled",
        "candidates": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    for row in selected:
        plan = prepare_candidate(row, frame_workers=workers)
        if "family_weights" in metadata and metadata["family_weights"] != plan["family_weights"]:
            raise ValueError("selected candidates have different Semantic family weights")
        metadata["family_weights"] = plan["family_weights"]
        relative = f"candidates/{row['model_id']}/{row['case_id']}.json"
        write(output / relative, plan)
        metadata["candidates"].append({
            "model_id": row["model_id"], "case_id": row["case_id"],
            "plan": relative, "sha256": file_hash(output / relative),
            "request_count": plan["request_count"],
            "frame_count": len(plan["frames"]),
            "aggregation_delta_from_historical": plan["aggregation_delta_from_historical"],
        })
        print(f"prepared {row['model_id']}/{row['case_id']}: "
              f"{plan['request_count']} requests, {len(plan['frames'])} frames",
              flush=True)
    if code_hashes() != before_code:
        raise ValueError("verifier code changed during preparation; prepare again")
    metadata["request_count_per_run"] = sum(c["request_count"]
                                           for c in metadata["candidates"])
    metadata["fingerprint"] = json_hash(metadata)
    write(output / "protocol.json", metadata)
    return metadata


def load_protocol(root, *, check_environment=True):
    meta = read(Path(root) / "protocol.json")
    fingerprint = meta.pop("fingerprint")
    if meta.get("protocol") != PROTOCOL or json_hash(meta) != fingerprint:
        raise ValueError("prepared protocol fingerprint changed")
    meta["fingerprint"] = fingerprint
    if check_environment:
        if meta["code_hashes"] != code_hashes():
            raise ValueError("verifier code changed since prepare")
        if meta["judge"] != judge_config():
            raise ValueError("judge configuration changed since prepare")
    return meta


def replay_claim(plan, claim, frames, client_factory=tool_client_from_env):
    task = task_from_dict(claim["task"])
    graph = RequirementGraph.from_dict(plan["bundle"]["graph"])
    judge = FrozenJudge(client_factory(), max_tokens=config.MAX_TOKENS,
                        max_validation_retries=0)
    resolutions = []
    binary_resolution = None
    records = []
    unhealthy = False
    for call in claim["calls"]:
        is_binary = call["kind"] == "binary_resolution"
        judge.expected_template = claim["binary_template" if is_binary else "unknown_template"]
        frame_keys = call.get("frame_keys", call["frame_ids"])
        supplied = tuple(frames[fid] for fid in frame_keys)
        method = judge.judge_binary if is_binary else judge.judge
        raw_before = len(judge.raw_records)
        decision = method(claim["payload"], supplied)
        raw = list(judge.raw_records)[raw_before:]
        failed = decision.transport_status != "success" or decision.parse_status != "valid"
        # The production adapter validates evidence grounding and safely converts
        # invalid answers to UNKNOWN. Preserve that explicit decision, while
        # retaining and reporting the bad raw output; it is not a transport error.
        fallback = (not failed and decision.verdict.value == "UNKNOWN"
                    and any(r.get("validation_error") for r in raw))
        unhealthy |= failed
        resolution = (finalize_stage3_binary_decision(task, decision) if is_binary
                      else finalize_stage3_unknown_decision(graph, task, decision))
        if is_binary:
            binary_resolution = resolution
        else:
            resolutions.append(resolution)
        records.append({
            **call, "decision": decision.to_dict(), "resolution": resolution.to_dict(),
            "validation_fallback_used": fallback,
            "request_fingerprint": json_hash({
                "template": judge.expected_template,
                "frame_rgb_hashes": [plan["frames"][fid]["rgb_sha256"]
                                     for fid in frame_keys],
                "frame_ids": call["frame_ids"], "judge": judge_config(),
            }),
            "raw_records": raw,
        })
    final = _aggregate_stage3_batches(task, resolutions, omitted_frame_ids=())
    binary_used = False
    if (final.final_verdict is not None and final.final_verdict.value == "UNKNOWN"
            and binary_resolution is not None):
        final = binary_resolution
        binary_used = True
    unhealthy |= bool(final.evaluation_error)
    return {"node_id": claim["node_id"], "status": "error" if unhealthy else "complete",
            "resolution": final.to_dict(), "binary_used": binary_used,
            "validation_fallback_count": sum(r["validation_fallback_used"] for r in records),
            "calls": records}


def score_replayed(plan, claim_results):
    rows = copy.deepcopy(plan["frozen_score_rows"])
    updated = {r["node_id"]: r for r in claim_results}
    if set(updated) != {r["node_id"] for r in plan["claims"]}:
        raise ValueError("incomplete replay claim set")
    for row in rows:
        replacement = updated.get(row["node_id"])
        if replacement is None:
            continue
        verdict = replacement["resolution"]["final_verdict"]
        if replacement["status"] != "complete" or verdict is None:
            row["evaluation_status"], row["score"] = "ERROR", None
        elif verdict == "UNKNOWN":
            row["evaluation_status"], row["score"] = "NOT_EVALUATED", None
        else:
            row["evaluation_status"] = verdict
            row["score"] = 1.0 if verdict == "MATCH" else 0.0
    return score_semantic_case(plan["bundle"]["graph"]["prompt"], rows,
                               family_weights=plan["family_weights"])


def _safe_output(prepared, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output == prepared or prepared in output.parents or output in prepared.parents:
        raise ValueError("run output must be separate from prepared inputs")
    return prepared, output


def run(prepared, output, *, run_ids=(), resume=False,
        client_factory=tool_client_from_env):
    prepared, output = _safe_output(prepared, output)
    meta = load_protocol(prepared)
    chosen = list(run_ids) or list(range(1, meta["repeats"] + 1))
    if len(chosen) != len(set(chosen)) or any(
            i not in range(1, meta["repeats"] + 1) for i in chosen):
        raise ValueError("invalid or duplicate run number")
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != meta:
            raise ValueError("output belongs to a different prepared experiment")
        if not resume:
            raise ValueError("existing experiment output requires --resume")
    elif any(output.iterdir()):
        raise ValueError("refusing a nonempty unrelated output directory")
    else:
        write(protocol_path, meta)
    lock = output / ".running.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        for number in chosen:
            for candidate in meta["candidates"]:
                plan = checked_json(prepared / candidate["plan"], candidate["sha256"])
                destination = (output / f"run-{number}" / candidate["model_id"]
                               / candidate["case_id"])
                identity = json_hash({"protocol": meta["fingerprint"],
                                      "plan": candidate["sha256"], "run": number})
                done = destination / "result.json"
                if done.exists():
                    if read(done).get("identity") != identity:
                        raise ValueError("cached case belongs to a different run")
                    print(f"resume: retained {done}", flush=True)
                    continue
                # Validate every image before creating any network client.
                frames = {fid: load_frame(spec) for fid, spec in plan["frames"].items()}
                destination.mkdir(parents=True, exist_ok=True)

                def perform(item, destination=destination, identity=identity,
                            plan=plan, frames=frames):
                    index, claim = item
                    path = destination / "claims" / f"{index:04d}.json"
                    if path.exists():
                        cached = read(path)
                        if cached.get("identity") != identity or cached["node_id"] != claim["node_id"]:
                            raise ValueError("cached claim belongs to a different run")
                        return cached
                    result = replay_claim(plan, claim, frames, client_factory)
                    result["identity"] = identity
                    write(path, result)
                    return result

                with ThreadPoolExecutor(max_workers=meta["workers"]) as pool:
                    outcomes = list(pool.map(perform, enumerate(plan["claims"])))
                if code_hashes() != meta["code_hashes"]:
                    raise ValueError("verifier code changed during replay")
                aggregate = score_replayed(plan, outcomes)
                errors = sum(r["status"] != "complete" for r in outcomes)
                summary = {
                    "identity": identity, "model_id": plan["model_id"],
                    "case_id": plan["case_id"], "run": number,
                    "status": "error" if errors else "complete",
                    "score": aggregate["score"] if not errors else None,
                    "diagnostic_aggregation": aggregate, "error_claim_count": errors,
                    "validation_fallback_count": sum(r["validation_fallback_count"]
                                                     for r in outcomes),
                    "logical_request_count": sum(len(r["calls"]) for r in outcomes),
                }
                write(done, summary)
                print(f"run {number} {plan['model_id']}/{plan['case_id']}: "
                      f"{summary['status']} score={summary['score']}", flush=True)
                summarize(output)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
    return summarize(output)


def tau_b(left, right):
    concordant = discordant = tied_left = tied_right = 0
    for i, j in combinations(range(len(left)), 2):
        a = (left[i] > left[j]) - (left[i] < left[j])
        b = (right[i] > right[j]) - (right[i] < right[j])
        if a and b:
            concordant += a == b
            discordant += a != b
        elif not a and b:
            tied_left += 1
        elif a and not b:
            tied_right += 1
    denominator = math.sqrt((concordant + discordant + tied_left)
                            * (concordant + discordant + tied_right))
    return (concordant - discordant) / denominator if denominator else None


def summarize(output):
    output = Path(output)
    meta = load_protocol(output, check_environment=False)
    runs = list(range(1, meta["repeats"] + 1))
    scores = []
    cells = {}
    validation_fallbacks = 0
    for r in runs:
        for c in meta["candidates"]:
            path = output / f"run-{r}" / c["model_id"] / c["case_id"] / "result.json"
            value = read(path) if path.exists() else {"status": "pending", "score": None}
            validation_fallbacks += value.get("validation_fallback_count", 0)
            if path.exists():
                identity = json_hash({"protocol": meta["fingerprint"],
                                      "plan": c["sha256"], "run": r})
                if value.get("identity") != identity:
                    raise ValueError(f"result fingerprint mismatch: {path}")
            cell = {"model": c["model_id"], "case": c["case_id"], "run": r,
                    "status": value["status"], "score": value["score"]}
            scores.append(cell)
            cells[(c["model_id"], c["case_id"], r)] = cell
    case_stats = []
    model_runs = {}
    model_stats = []
    for model in meta["model_ids"]:
        for case in meta["case_ids"]:
            series = [cells[(model, case, r)] for r in runs]
            complete = all(v["status"] == "complete" for v in series)
            values = [v["score"] for v in series] if complete else []
            case_stats.append({"model": model, "case": case,
                               "complete_runs": sum(v["status"] == "complete" for v in series),
                               "mean": statistics.mean(values) if complete else None,
                               "sd": statistics.stdev(values) if complete else None})
        averages = []
        for r in runs:
            series = [cells[(model, case, r)] for case in meta["case_ids"]]
            average = (statistics.mean(v["score"] for v in series)
                       if all(v["status"] == "complete" for v in series) else None)
            model_runs[(model, r)] = average
            averages.append(average)
        complete = all(v is not None for v in averages)
        model_stats.append({
            "model": model, **{f"run_{r}": value for r, value in zip(runs, averages, strict=True)},
            "mean": statistics.mean(averages) if complete else None,
            "sd": statistics.stdev(averages) if complete else None,
        })
    ranking = []
    for a, b in combinations(runs, 2):
        x = [model_runs[(m, a)] for m in meta["model_ids"]]
        y = [model_runs[(m, b)] for m in meta["model_ids"]]
        complete = all(v is not None for v in x + y)
        flips = []
        if complete:
            for i, j in combinations(range(len(x)), 2):
                if (x[i] - x[j]) * (y[i] - y[j]) < 0:
                    flips.append([meta["model_ids"][i], meta["model_ids"][j]])
        ranking.append({"runs": [a, b], "tau_b": tau_b(x, y) if complete else None,
                        "complete_panel": complete, "reversed_pairs": flips})
    write_csv(output / "scores.csv", scores, ["model", "case", "run", "status", "score"])
    write_csv(output / "case_statistics.csv", case_stats,
              ["model", "case", "complete_runs", "mean", "sd"])
    write_csv(output / "model_statistics.csv", model_stats,
              ["model", *(f"run_{r}" for r in runs), "mean", "sd"])
    summary = {"protocol": PROTOCOL, "planned_scores": len(scores),
               "complete_scores": sum(v["status"] == "complete" for v in scores),
               "error_scores": sum(v["status"] == "error" for v in scores),
               "validation_fallback_count": validation_fallbacks,
               "ranking_comparisons": ranking,
               "sd_definition": "sample SD across independent runs, ddof=1",
               "inference_scope": meta["scope"]}
    write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="validate evidence and freeze requests; no VLM calls")
    prep.add_argument("--selection", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--model", action="append", default=[])
    prep.add_argument("--case", action="append", default=[])
    prep.add_argument("--repeats", type=int, default=3)
    prep.add_argument("--workers", type=int, default=4)
    execute = sub.add_parser("run", help="make fresh VLM calls for the prepared repeats")
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--out-dir", type=Path, required=True)
    execute.add_argument("--run", action="append", type=int, default=[])
    execute.add_argument("--resume", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.selection, args.out_dir, models=args.model,
                             cases=args.case, repeats=args.repeats, workers=args.workers)
            print(json.dumps({k: result[k] for k in
                              ["fingerprint", "request_count_per_run", "repeats"]}))
        elif args.command == "run":
            result = run(args.prepared, args.out_dir, run_ids=args.run, resume=args.resume)
            print(json.dumps(result))
            return int(result["error_scores"] > 0)
        else:
            print(json.dumps(summarize(args.out_dir)))
    except (ValueError, KeyError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
