"""Every report that reaches the record honours the envelope it is scored by.

The verifier tests in this suite each drive one verifier and assert what it
SAYS — the status, the reason, the number. None of them put the report back
through the thing that reads it, and that gap is where two of the audit's
findings lived:

* `content_parity` returned an ERROR carrying 0.0 on its drift branch.
  `contracts.score_for_aggregate` rejects that by design — an error holding a
  number must never be read as a bad scene — so it raised, out of the loop in
  `episode` that builds `scores_by_class`, and took the whole run's scoring
  down at the exact moment the verifier caught the drift it exists for. Its own
  test asserted the status and the reason and passed throughout.
* `bounds_discipline` returned PASS carrying 0.0 when there was no actor count
  to divide by: the worst number in the best status.

So this file tests the ENVELOPE rather than any one verifier's judgement, over
every kind the registry discovers. A verifier added tomorrow is covered the day
it is registered, which is the only way this stays true.

What it deliberately does not do is assert that a starved context produces a
refusal. Some verifiers may one day answer from the record alone. The contract
is about the SHAPE of whatever they answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code4scene.evaluation import components, contracts
from code4scene.evaluation import verifiers as verifiers_eval

IDS = {"task_bundle_id": "bundle-1", "episode_id": "ep-1"}

#: Statuses a report may carry. The first three are report-level; the last two
#: normally live on typed metric results and are accepted here because
#: `contracts.WITHHELD_STATUSES` accepts them, and a boundary that fails closed
#: has to be tested at the shape it actually admits.
STATUSES = {
    contracts.PASS,
    contracts.FAIL,
    contracts.ERROR,
    contracts.MEASURED,
    contracts.VALID,
    contracts.INVALID,
    "not_evaluated",
    "not_applicable",
}


def starved_reports(kinds: list[str]) -> list[dict]:
    """Run each kind with nothing supplied, through the production path.

    `verifiers.run` is what fills `record["verifiers"]`, and it converts a
    verifier that raises into an error report. That conversion is part of the
    contract under test: the record must never hold something the aggregate
    cannot read, whether the verifier returned it or threw on the way.
    """
    task = SimpleNamespace(
        verifiers=[{"name": kind} for kind in kinds],
        path=None,
        prompt="a scene",
        size_m=None,
        kind="scene_repair",
        case_type="image_to_scene",
        data={"source": {"reference_views": ["missing-reference.png"]}},
    )
    record = {"exit_reason": "completed", "metrics": {}, "rounds": []}
    return verifiers_eval.run(task, record, IDS)


@pytest.fixture(scope="module")
def reports() -> dict[str, dict]:
    kinds = verifiers_eval.kinds()
    produced = starved_reports(kinds)
    assert len(produced) == len(kinds), (
        "one report per declared verifier, or the record silently under-reports")
    return {report["report_id"]: report for report in produced}


def test_every_registered_verifier_produces_a_report(reports):
    assert set(reports) == set(verifiers_eval.kinds())


@pytest.mark.parametrize("kind", verifiers_eval.kinds())
def test_the_report_envelope_holds(reports, kind):
    report = reports[kind]
    status = report.get("status")

    assert status in STATUSES, f"{kind}: unknown status {status!r}"
    for key in ("report_id", "task_bundle_id", "episode_id"):
        assert report.get(key), f"{kind}: report carries no {key}"
    if status not in {contracts.PASS, contracts.MEASURED, contracts.VALID}:
        assert report.get("failure_reason"), (
            f"{kind}: {status} without a reason — the shared schema refuses a "
            f"failure that does not say why, and an auditor cannot act on it")
    if status in contracts.WITHHELD_STATUSES:
        assert report.get("score") is None, (
            f"{kind}: {status} carrying {report['score']!r}. A refusal with a "
            f"number in it is indistinguishable from a measured bad scene, and "
            f"`score_for_aggregate` raises on it")


@pytest.mark.parametrize(
    "kind",
    sorted({"physical_safety", "semantic_requirements", "source_preservation"}),
)
def test_a_scored_verifier_can_always_be_aggregated(reports, kind):
    """The call that crashed the episode, made a test.

    `episode` now catches the violation rather than dying of it, so this is
    what keeps the catch from becoming a place violations quietly accumulate.
    """
    contracts.score_for_aggregate(reports[kind])


def test_a_valid_gate_without_a_number_is_not_a_quality_score():
    report = {
        "report_id": "candidate_integrity",
        "status": contracts.VALID,
        "score": None,
    }

    assert not components.is_scored("candidate_integrity")
    assert contracts.score_for_aggregate(report) is None


# ── the two branches this file was written for ────────────────────────────


def test_bounds_discipline_does_not_pass_a_scene_it_could_not_normalize():
    """Corrections per Actor over zero Actors is not a clean plate."""
    from code4scene.evaluation.verifiers import bounds_discipline

    context = SimpleNamespace(
        record={"edge_discipline": {"passes": [{"clamped": 0, "deleted": 0}],
                                    "total_clamped": 0, "total_deleted": 0},
                "metrics": {"actors": 0}},
        ids=IDS, spec={})
    report = bounds_discipline.verify(context)

    assert report["status"] == contracts.ERROR
    assert report["score"] is None
    assert report["metrics"]["corrections_per_actor"] is None
    contracts.score_for_aggregate(report)


def test_content_parity_drift_is_withheld_and_survives_the_aggregate(tmp_path):
    """The drift branch is the one that crashed scoring; it must be readable."""
    from code4scene.evaluation.verifiers import content_parity

    saved = tmp_path / "instance" / "scenes" / "saved"
    saved.mkdir(parents=True)
    (tmp_path / "instance" / ".scenebench-pristine.json").write_text(
        '{"release": "ue58-b"}')
    context = SimpleNamespace(
        record={"scene_dependencies": {"release": "ue58-a", "package_count": 40,
                                       "unresolved": ["/Game/X"]}},
        ids=IDS, spec={},
        scoring=SimpleNamespace(saved_scenes_dir=str(saved)))
    report = content_parity.verify(context)

    assert report["status"] == contracts.ERROR
    assert report["score"] is None
    assert contracts.score_for_aggregate(report) is None
