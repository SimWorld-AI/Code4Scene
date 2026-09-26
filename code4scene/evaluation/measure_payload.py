"""Canonical in-editor scene measurement — the single source of truth.

This file is BOTH:
  - an importable module: :func:`compute` is pure and unit-tested offline;
  - the payload executed inside the UE editor python: the wrapper sends this
    file's source over the bridge with a trailing ``main(gh, out)`` call.
``import unreal`` therefore happens inside :func:`collect_rows` only.

This is the only active scene-measurement implementation. ``ir-measure.js`` reads this
file and appends the ``main(...)`` call, so the agent's ``measure_scene`` tool
and the harness's own closing measurement run the same bytes. It used to carry
its own copy of the algorithm, which meant an agent could optimise against a
different implementation than the one that scored it.

Provenance: verbatim port of the python previously embedded as strings in
an earlier JavaScript measurement tool (``_measureScript``), which was itself the
authoritative superset of the drifted copy in
``run_asset_retrieval_ab_eval.js::computeSceneMetrics`` (a standalone asset
retrieval A/B script, not benchmark scoring). Editor scoring, Studio and offline
rescoring share the fixed 0--5 cm floating rule below. No legacy detector or
threshold selector is retained. Historical reports keep their original metric
identifiers; new measurements identify the current rule explicitly.

Units: centimetres throughout (UE units); output rounds to metres.
"""

from __future__ import annotations

import bisect
import json
import math
from typing import Any

# Markers are grepped by callers; keep byte-identical to ir-measure.js.
MARK_OK = "IR_MEASURE_OK"
MARK_ERR = "IR_MEASURE_ERR"

METRIC_ID = "measure_scene_5cm"
FLOATING_METRIC_ID = "t2s-aabb-floating-contact-5cm.v2"

# GH: configurable out-of-bounds half-extent; FT: fixed support tolerance.
DEFAULT_GH_CM = 13000.0
DEFAULT_FT_CM = 5.0
TOUCH_CM = 5.0        # AABB overlap slack: all three axes must exceed this
FLAT_EZ_CM = 20.0     # half-height below which an actor is "flat" (ground sheets)
CLUTTER_R_CM = 80.0   # footprint radius at or above which an actor is "big"

# Actors excluded from measurement ("scoring infrastructure"). Matched on
# lowercased label prefix (INFRA) or exact lowercased class name (INFRA_CLS).
INFRA = ('floor', 'sky', 'light', 'atmo', 'fog', 'post', 'sphere', 'world',
         'brush', 'default', 'player', 'gamemode', 'nav', 'levelbounds',
         'landscape', 'volume', 'note', 'camera', 'directional', 'exponential',
         'exprunner', 'arena_env', 'ground_plane')
INFRA_CLS = ('directionallight', 'skylight', 'skyatmosphere',
             'exponentialheightfog', 'worldsettings', 'playerstart',
             'cameraactor', 'reflectioncapture', 'brush',
             'navmeshboundingvolume', 'postprocessvolume')


def _measurement_row(label, cls, origin, extent, pivot):
    """Apply the same population and geometry contract to UE and saved scenes."""
    if cls.lower() in INFRA_CLS or str(label).lower().startswith(INFRA):
        return None
    vectors = (origin, extent, pivot)
    if not all(len(vector) == 3 and all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in vector
    ) for vector in vectors):
        raise ValueError("measurement geometry must contain finite 3-vectors")
    if all(value < 1 for value in extent):
        return None
    return {'n': label, 'cls': cls,
            'ox': origin[0], 'oy': origin[1], 'oz': origin[2],
            'ex': max(extent[0], 1.0), 'ey': max(extent[1], 1.0),
            'ez': max(extent[2], 1.0), 'lx': pivot[0], 'ly': pivot[1],
            'bot': origin[2] - extent[2], 'top': origin[2] + extent[2]}


def rows_from_scene(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect measurement rows from an independently exported scene graph."""
    rows = []
    for actor in scene['actors']:
        label = actor.get('label', actor.get('name'))
        cls = actor.get('class', '').rsplit('.', 1)[-1]
        if cls.lower() in INFRA_CLS or str(label).lower().startswith(INFRA):
            continue
        row = _measurement_row(label, cls, actor['bounds']['origin_cm'],
                               actor['bounds']['extent_cm'],
                               actor['transform']['location_cm'])
        if row is not None:
            rows.append(row)
    return rows


def collect_rows():
    """Gather one measurement row per non-infra actor. Editor-only."""
    import unreal
    acts = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()

    def infra(a):
        if a.get_class().get_name().lower() in INFRA_CLS:
            return True
        try:
            lbl = a.get_actor_label().lower()
        except Exception:
            lbl = a.get_name().lower()
        return lbl.startswith(INFRA)

    rows = []
    for a in acts:
        if infra(a):
            continue
        try:
            o, e = a.get_actor_bounds(False)
        except Exception:
            continue
        try:
            lbl = a.get_actor_label()
        except Exception:
            lbl = a.get_name()
        loc = a.get_actor_location()
        row = _measurement_row(lbl, a.get_class().get_name(),
                               (o.x, o.y, o.z), (e.x, e.y, e.z),
                               (loc.x, loc.y, loc.z))
        if row is not None:
            rows.append(row)
    return rows


def floating_actors(rows: list[dict[str, Any]]) -> list[str]:
    """Unsupported actors under the sole 0--5 cm, inclusive-contact rule.

    Keep the established asymmetric horizontal pivot/box support test. Sorting
    support tops avoids rescanning every actor for offline floating-only work.
    """
    ordered = sorted(enumerate(rows), key=lambda item: item[1]['top'])
    tops = [actor['top'] for _, actor in ordered]
    failed = []
    for i, actor in enumerate(rows):
        if actor['bot'] <= DEFAULT_FT_CM:
            continue
        start = bisect.bisect_left(tops, actor['bot'] - DEFAULT_FT_CM - 1e-8)
        end = bisect.bisect_right(tops, actor['bot'] + 1e-8)
        supported = any(
            j != i and support['top'] <= actor['bot']
            and actor['bot'] - support['top'] <= DEFAULT_FT_CM
            and abs(support['ox'] - actor['lx']) < actor['ex'] * 0.5 + support['ex']
            and abs(support['oy'] - actor['ly']) < actor['ey'] * 0.5 + support['ey']
            for j, support in ordered[start:end]
        )
        if not supported:
            failed.append(actor['n'])
    return failed


def compute(rows: list[dict[str, Any]],
            gh_cm: float = DEFAULT_GH_CM) -> dict[str, Any]:
    """Pure metric computation over collected rows: ``{'actors', 'report'}``."""
    B = rows
    n = len(B)

    def flat(b):
        return b['ez'] < FLAT_EZ_CM

    def big(b):
        return (b['ex'] * b['ex'] + b['ey'] * b['ey']) ** 0.5 >= CLUTTER_R_CM

    # Pairwise AABB collisions. Flat participants are skipped; a pair is
    # "structural" when either side is big. Every axis overlap must exceed
    # TOUCH_CM, so mere touching never counts.
    coll = set(); pairs = 0; scoll = set(); spairs = 0; cw: dict[str, list[str]] = {}
    for i in range(n):
        a = B[i]
        for j in range(i + 1, n):
            b = B[j]
            if flat(a) or flat(b):
                continue
            ox = min(a['ox'] + a['ex'], b['ox'] + b['ex']) - max(a['ox'] - a['ex'], b['ox'] - b['ex'])
            if ox <= TOUCH_CM:
                continue
            oy = min(a['oy'] + a['ey'], b['oy'] + b['ey']) - max(a['oy'] - a['ey'], b['oy'] - b['ey'])
            if oy <= TOUCH_CM:
                continue
            oz = min(a['top'], b['top']) - max(a['bot'], b['bot'])
            if oz <= TOUCH_CM:
                continue
            pairs += 1; coll.add(i); coll.add(j)
            if big(a) or big(b):
                spairs += 1; scoll.add(i); scoll.add(j)
                cw.setdefault(a['n'], []).append(b['n'])
                cw.setdefault(b['n'], []).append(a['n'])

    fl = floating_actors(B)

    # Out-of-bounds is judged on the actor PIVOT (lx/ly), not its bounds.
    oob = [a['n'] for a in B if abs(a['lx']) > gh_cm or abs(a['ly']) > gh_cm]

    actors = [{'label': b['n'],
               'x_m': round(b['ox'] / 100.0, 3), 'y_m': round(b['oy'] / 100.0, 3),
               'z_min': round(b['bot'] / 100.0, 3),
               'w_m': round(2 * b['ex'] / 100.0, 3), 'd_m': round(2 * b['ey'] / 100.0, 3),
               'h_m': round(2 * b['ez'] / 100.0, 3),
               'cls': b['cls'], 'flat': flat(b), 'big': big(b)} for b in B]
    report = {'checked': n,
              'collision_pairs': pairs,
              'structural_collision_pairs': spairs,
              'structural_collision_actors': sorted(set(B[i]['n'] for i in scoll)),
              'collision_actors': len(coll),
              'floating': fl,
              'out_of_bounds': oob,
              'colliding_with': cw}
    return {'actors': actors, 'report': report}


def main(gh_cm: float, out_path: str) -> None:
    """In-editor entry point: collect, compute, write JSON, print marker.

    Per-actor data for a dense scene is hundreds of KB — printing it as one
    log line overflows the editor-log capture, hence the file + tiny marker.
    """
    try:
        payload = compute(collect_rows(), gh_cm)
        with open(out_path, 'w') as f:
            f.write(json.dumps(payload))
        print(MARK_OK + ' n=' + str(payload['report']['checked']) + ' path=' + out_path)
    except Exception as e:
        print(MARK_ERR + ' ' + str(e))
