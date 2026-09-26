"""Match two sets of Actors one-to-one, at the lowest total cost.

Comparing a built scene with a canonical one needs a correspondence first, and
greedy nearest-neighbour gives the wrong one: the first Actor takes the match
a later Actor needed more, and the resulting "error" is an artefact of the
iteration order. This is the Hungarian algorithm (Jonker-Volgenant form), so
the correspondence is the globally cheapest one and is independent of the
order the scenes were exported in.

No SciPy. The imaging extra is optional and this is not in it; a scoring pass
that silently needs a numerical stack it was never given is a scoring pass
that does not run in the container.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class AssignmentError(ValueError):
    """The cost matrix is not something an assignment can be solved over."""


def _validate(costs: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not isinstance(costs, (list, tuple)):
        raise AssignmentError("assignment costs must be a matrix")
    if not costs:
        return 0, 0
    columns = len(costs[0]) if isinstance(costs[0], (list, tuple)) else 0
    if not columns or any(not isinstance(row, (list, tuple)) or len(row) != columns
                          for row in costs):
        raise AssignmentError("the cost matrix must be rectangular and non-empty")
    if any(not math.isfinite(float(value)) for row in costs for value in row):
        raise AssignmentError("assignment costs must be finite numbers")
    return len(costs), columns


def solve(costs: Sequence[Sequence[float]]) -> list[int]:
    """Column chosen for each row, minimising the total cost.

    Rows must not outnumber columns; `match` handles the general case by
    transposing. Returns -1 for a row left unassigned, which only happens for
    an empty matrix.
    """
    rows, columns = _validate(costs)
    if not rows or not columns:
        return [-1] * rows
    if rows > columns:
        raise AssignmentError("solve needs rows <= columns; transpose first")

    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for i in range(1, rows + 1):
        p[0] = i
        column0 = 0
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta, column1 = math.inf, 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = float(costs[row0 - 1][column - 1]) - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column], way[column] = current, column0
                if minimum[column] < delta:
                    delta, column1 = minimum[column], column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while column0:
            previous = way[column0]
            p[column0] = p[previous]
            column0 = previous

    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def match(costs: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """(row, column) pairs of the cheapest one-to-one correspondence."""
    rows, columns = _validate(costs)
    if not rows or not columns:
        return []
    if rows <= columns:
        return [(row, column) for row, column in enumerate(solve(costs))
                if column >= 0]
    transposed = [[costs[row][column] for row in range(rows)]
                  for column in range(columns)]
    return [(row, column) for column, row in enumerate(solve(transposed))
            if row >= 0]


__all__ = ["AssignmentError", "match", "solve"]
