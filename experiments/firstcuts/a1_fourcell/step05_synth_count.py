#!/usr/bin/env python3
"""Count synthetic shortest-route pairs for A1 and check their XY geometry.

This is a topology/data-shape probe only: it does not create training samples,
load a checkpoint, or invoke E(tau). BFS is delegated to step0_count's helpers
and lacot.subgoal.grid_bfs; route-to-anchor conversion uses lacot.intent.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lacot.intent import anchors_resample, cells_to_anchors, route_cells
from lacot.subgoal import grid_bfs
from step0_count import (
    DEFAULT_ENV,
    _cell_mapper,
    _default_dataset_dir,
    _free_cells,
    _route_code,
    _shortest_next_steps,
    load_official_data,
)


EXPECTED_DISTANCE_4_BRANCHES = 8
EXPECTED_DISTANCE_5_19_BRANCHES = 107
DEFAULT_T_CAP = 128


def _neighbors(cell: tuple[int, int]):
    """Yield the same four cardinal neighbors used by step0_count's BFS helper."""
    i, j = cell
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield i + di, j + dj


def _enumerate_shortest_routes(
    occ: np.ndarray,
    src: tuple[int, int],
    dst: tuple[int, int],
    dist_to_goal: dict[tuple[int, int], int],
    width: int,
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate all paths that monotonically descend the shared BFS map."""
    distance = dist_to_goal.get(src)
    if distance is None:
        return []

    routes: list[tuple[tuple[int, int], ...]] = []

    def visit(cell: tuple[int, int], route: list[tuple[int, int]]) -> None:
        if cell == dst:
            routes.append(tuple(route))
            return
        remaining = dist_to_goal[cell]
        for nxt in _neighbors(cell):
            if (
                0 <= nxt[0] < occ.shape[0]
                and 0 <= nxt[1] < occ.shape[1]
                and occ[nxt]
                and dist_to_goal.get(nxt) == remaining - 1
            ):
                visit(nxt, [*route, nxt])

    visit(src, [src])
    # Cell-sequence identity follows step0_count's corridor-code convention.
    unique = {_route_code(list(route), width): route for route in routes}
    return [unique[code] for code in sorted(unique)]


def _cardinal_cell_width(occ: np.ndarray, cell_to_xy) -> tuple[float, float]:
    lengths = []
    for cell in _free_cells(occ):
        here = np.asarray(cell_to_xy(cell), dtype=np.float64)
        for nxt in _neighbors(cell):
            if 0 <= nxt[0] < occ.shape[0] and 0 <= nxt[1] < occ.shape[1] and occ[nxt]:
                there = np.asarray(cell_to_xy(nxt), dtype=np.float64)
                lengths.append(float(np.linalg.norm(there - here)))
    if not lengths:
        raise ValueError("maze has no cardinal free-cell edges")
    return float(np.median(lengths)), float(max(lengths) - min(lengths))


def _route_first_cell(route: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return route[1]


def _branch_record(
    occ: np.ndarray,
    src: tuple[int, int],
    dst: tuple[int, int],
    distance: int,
    dist_to_goal: dict[tuple[int, int], int],
    width: int,
    cell_to_xy,
    xy_to_cell,
    t_cap: int,
    cell_width: float,
) -> dict[str, Any]:
    _, legal_first = _shortest_next_steps(occ, src, dst)
    if len(legal_first) < 2:
        raise AssertionError(f"not a branch according to step0_count: {src} -> {dst}")

    routes = _enumerate_shortest_routes(occ, src, dst, dist_to_goal, width)
    if len(routes) < 2:
        raise AssertionError(f"branch has fewer than two shortest routes: {src} -> {dst}")

    route_codes = [_route_code(list(route), width) for route in routes]
    # route_cells is the repo's canonical deterministic shortest route. It must
    # be one of the fully enumerated shortest corridor-cell sequences.
    canonical = tuple(map(tuple, route_cells(occ, src, dst) or []))
    canonical_code = _route_code(list(canonical), width)
    if canonical_code not in route_codes:
        raise AssertionError(f"intent.route_cells route missing from enumeration: {src} -> {dst}")

    geometric_pair_count = len(routes) * (len(routes) - 1) // 2
    first_step_pair_count = sum(
        1
        for a, b in itertools.combinations(routes, 2)
        if _route_first_cell(a) != _route_first_cell(b)
    )

    # For each branch point, convert a canonical route and an alternative with
    # a different legal first step using the established intent geometry API.
    other = next(
        route for route in routes
        if _route_first_cell(route) != _route_first_cell(canonical)
    )
    anchors_a = cells_to_anchors(canonical, cell_to_xy)
    anchors_b = cells_to_anchors(other, cell_to_xy)
    trajectory_a = anchors_resample(anchors_a, t_cap)
    trajectory_b = anchors_resample(anchors_b, t_cap)
    if trajectory_a.shape != (t_cap, 2) or trajectory_b.shape != (t_cap, 2):
        raise AssertionError("anchors_resample did not return [T,2]")
    if trajectory_a.dtype != np.float32 or trajectory_b.dtype != np.float32:
        raise AssertionError("intent anchor contract requires float32")

    # Both shortest routes have equal arc length; compare equal normalized
    # progress samples from anchors_resample as the geometric branch distance.
    max_progress_separation = float(
        np.linalg.norm(trajectory_a.astype(np.float64) - trajectory_b.astype(np.float64), axis=1).max()
    )
    geometry_pass = max_progress_separation + 1e-6 >= cell_width
    if not geometry_pass:
        raise AssertionError(
            f"XY paths do not separate by a cell width: {src}->{dst}; "
            f"max={max_progress_separation:.6f}, cell={cell_width:.6f}"
        )

    first_step_counts = Counter(
        f"{route[1][0]},{route[1][1]}" for route in routes
    )
    path_length_cells = distance + 1
    return {
        "src": list(src),
        "dst": list(dst),
        "distance_edges": int(distance),
        "route_length_cells": int(path_length_cells),
        "n_distinct_shortest_routes": len(routes),
        "n_geometric_route_pairs": geometric_pair_count,
        "n_first_step_distinct_swap_pairs": first_step_pair_count,
        "route_pair_length_distribution_cells": {str(path_length_cells): geometric_pair_count},
        "swap_pair_length_distribution_cells": {str(path_length_cells): first_step_pair_count},
        "routes": [
            {
                "cells": [list(cell) for cell in route],
                "corridor_code": list(code),
                "first_next_cell": list(route[1]),
            }
            for route, code in zip(routes, route_codes)
        ],
        "geometry_check": {
            "route_a_corridor_code": list(canonical_code),
            "route_b_corridor_code": list(_route_code(list(other), width)),
            "route_a_xy_anchors": anchors_a.astype(float).tolist(),
            "route_b_xy_anchors": anchors_b.astype(float).tolist(),
            "resampled_shape": list(trajectory_a.shape),
            "resampled_dtype": str(trajectory_a.dtype),
            "comparison": "max pointwise XY separation at equal normalized arc progress",
            "max_separation_xy": max_progress_separation,
            "cell_width_xy": cell_width,
            "pass": geometry_pass,
        },
        "routes_per_first_next_cell": dict(sorted(first_step_counts.items())),
        "legal_first_next_cells": [list(cell) for cell in sorted(legal_first)],
        "all_route_endpoints_roundtrip_to_their_cells": bool(
            all(xy_to_cell(cell_to_xy(cell)) == cell for route in routes for cell in (route[0], route[-1]))
        ),
    }


def analyze(env_name: str, dataset_dir: str | Path, t_cap: int) -> dict[str, Any]:
    env, _, _ = load_official_data(env_name, dataset_dir)
    maze = np.asarray(env.unwrapped.maze_map)
    occ = maze == 0
    if not occ.any():
        raise ValueError("OGBench maze_map contains no passable cells (expected value 0)")
    width = int(maze.shape[1])
    xy_to_cell = _cell_mapper(env, occ)
    cell_to_xy = env.unwrapped.ij_to_xy
    cell_width, cell_width_spread = _cardinal_cell_width(occ, cell_to_xy)
    if cell_width_spread > 1e-5:
        raise ValueError(f"maze cardinal cell widths are not uniform: spread={cell_width_spread}")

    cells = _free_cells(occ)
    bfs_maps = {cell: grid_bfs(occ, cell) for cell in cells}
    branches = []
    for src in cells:
        for dst, distance in bfs_maps[src].items():
            if distance < 4 or distance > 19:
                continue
            _, legal_first = _shortest_next_steps(occ, src, dst)
            if len(legal_first) < 2:
                continue
            branches.append(
                _branch_record(
                    occ=occ,
                    src=src,
                    dst=dst,
                    distance=distance,
                    dist_to_goal=bfs_maps[dst],
                    width=width,
                    cell_to_xy=cell_to_xy,
                    xy_to_cell=xy_to_cell,
                    t_cap=t_cap,
                    cell_width=cell_width,
                )
            )
    branches.sort(key=lambda row: (row["distance_edges"], row["src"], row["dst"]))

    by_tier = {
        "distance_4": sum(row["distance_edges"] == 4 for row in branches),
        "distance_5_to_19": sum(5 <= row["distance_edges"] <= 19 for row in branches),
    }
    if by_tier != {
        "distance_4": EXPECTED_DISTANCE_4_BRANCHES,
        "distance_5_to_19": EXPECTED_DISTANCE_5_19_BRANCHES,
    }:
        raise AssertionError(f"maze branch counts differ from accepted topology facts: {by_tier}")

    pair_count_by_length: Counter[str] = Counter()
    swap_count_by_length: Counter[str] = Counter()
    route_count_distribution: Counter[str] = Counter()
    for row in branches:
        length = str(row["route_length_cells"])
        pair_count_by_length[length] += row["n_geometric_route_pairs"]
        swap_count_by_length[length] += row["n_first_step_distinct_swap_pairs"]
        route_count_distribution[str(row["n_distinct_shortest_routes"])] += 1

    return {
        "env": env_name,
        "maze_shape": list(maze.shape),
        "free_cells": int(occ.sum()),
        "bfs_source": "step0_count._shortest_next_steps + lacot.subgoal.grid_bfs",
        "cell_route_identity": "step0_count._route_code(row * maze_width + column)",
        "branch_groups_by_distance": by_tier,
        "production": {
            "n_branch_groups": len(branches),
            "n_distinct_shortest_routes_across_groups": sum(
                row["n_distinct_shortest_routes"] for row in branches
            ),
            "n_geometric_route_pairs": sum(row["n_geometric_route_pairs"] for row in branches),
            "n_first_step_distinct_swap_pairs": sum(
                row["n_first_step_distinct_swap_pairs"] for row in branches
            ),
            "geometric_route_pair_count_by_route_length_cells": dict(sorted(pair_count_by_length.items(), key=lambda x: int(x[0]))),
            "first_step_distinct_pair_count_by_route_length_cells": dict(sorted(swap_count_by_length.items(), key=lambda x: int(x[0]))),
            "branch_groups_by_number_of_shortest_routes": dict(sorted(route_count_distribution.items(), key=lambda x: int(x[0]))),
        },
        "geometry_validation": {
            "branches_converted": len(branches),
            "branches_passing_one_cell_width_separation": sum(
                bool(row["geometry_check"]["pass"]) for row in branches
            ),
            "resample_points": t_cap,
            "resample_api": "lacot.intent.anchors_resample (equal arc length, [T,2] float32)",
            "anchor_api": "lacot.intent.cells_to_anchors (cell centers, [K,2] float32)",
            "route_api_role": "lacot.intent.route_cells supplies/validates the canonical tie-break shortest route; alternative routes come from the full BFS distance DAG enumeration",
            "cell_width_xy": cell_width,
            "cell_width_spread_xy": cell_width_spread,
            "minimum_branch_separation_xy": min(row["geometry_check"]["max_separation_xy"] for row in branches),
            "maximum_branch_separation_xy": max(row["geometry_check"]["max_separation_xy"] for row in branches),
        },
        "e_tau_input_compatibility": {
            "source": "experiments/scratch_lacot_rollout.py:etarget(traj, mask)",
            "input_trajectory": "[B,T,2] float32 XY only; Dc must equal XY_DIM=2",
            "mask": "[B,T] bool key_padding_mask; current training batch uses all-False mask for fixed-length real points",
            "preprocessing": "fixed T_CAP ordered XY points, linearly resampled from a temporal row-index window, then normalized with training XY mean/std; Perceiver sees point order/position but no explicit timestamp",
            "output": "[B,K,D_MODEL] E tokens",
            "not_required_by_encoder": ["velocity", "action", "timestamp"],
            "synthetic_chain_missing_for_direct_encoder_call": [
                "batch axis (wrap [T,2] as [1,T,2])",
                "fixed T_CAP length (anchors_resample can provide this)",
                "training-set XY normalization",
                "[1,T_CAP] all-False bool mask",
            ],
            "candidate_minimal_completion_options_for_lead": [
                "Option A: cells_to_anchors -> anchors_resample(saved T_CAP) -> normalize with train XY mean/std -> add batch axis and all-False mask. This matches the repo's synthetic teacher route pipeline, but parameterizes by arc length.",
                "Option B: assign per-edge durations/speeds from a declared real-data profile, interpolate the polyline uniformly in time to T_CAP, then normalize and use an all-False mask. Keep E input at 2D XY; finite-difference velocity is diagnostic only.",
                "Option C: append finite-difference velocity or timestamps only in a separately retrained/changed encoder contract; current etarget asserts exactly 2 XY dimensions, so these cannot be appended to this E input.",
            ],
        },
        "distribution_risk_and_cheapest_sanity_check": {
            "risk": "Equal-arc synthetic routes have nearly constant displacement per sample and cell-center corners. Native real windows are time/row-index sampled, so speed changes, pauses, inertia, overshoot, within-cell motion, and smooth turns alter adjacent XY distances and curvature. E gets no explicit speed/time fields, but ordered positions and positional embeddings expose these differences.",
            "design_only_not_run": "Select one held-out real segment with a known cell route. Build (1) the rollout-native fixed-T_CAP row-index-resampled XY and (2) the same segment reconstructed through traj_to_cells -> cells_to_anchors -> anchors_resample; normalize both with training XY mean/std and all-False masks. After lead authorizes an encoder comparison, compute E for each and report cosine distance plus normalized L2 over flattened [K,D_MODEL]. This isolates quantization/arc-resampling shift before comparing synthetic branch alternatives. No model is loaded or called in this step.",
            "comparison_metric": "cosine distance and normalized L2 between flattened E token arrays [K,D_MODEL]",
        },
        "branches": branches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--t-cap", type=int, default=DEFAULT_T_CAP)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).with_name("step05_synth_count.json"),
    )
    args = parser.parse_args(argv)
    if args.t_cap < 2:
        parser.error("--t-cap must be at least 2")
    dataset_dir = args.dataset_dir or _default_dataset_dir(args.env)
    result = analyze(args.env, dataset_dir, args.t_cap)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("A1 STEP 0.5 — SYNTHETIC SHORTEST-ROUTE BRANCH COUNT")
    print(f"env={result['env']} maze={result['maze_shape']} free_cells={result['free_cells']}")
    print(f"branch groups={result['branch_groups_by_distance']}")
    p = result["production"]
    print(
        "production: "
        f"routes={p['n_distinct_shortest_routes_across_groups']} "
        f"geometric_route_pairs={p['n_geometric_route_pairs']} "
        f"first_step_distinct_swap_pairs={p['n_first_step_distinct_swap_pairs']}"
    )
    print(f"geometric pairs by route length in cells={p['geometric_route_pair_count_by_route_length_cells']}")
    print(f"first-step-distinct pairs by route length in cells={p['first_step_distinct_pair_count_by_route_length_cells']}")
    print("\n# src -> dst | dist | routes | all pairs | swap pairs | route length cells | max separation / cell width")
    for row in result["branches"]:
        geom = row["geometry_check"]
        print(
            f"{row['src']} -> {row['dst']} | {row['distance_edges']} | "
            f"{row['n_distinct_shortest_routes']} | {row['n_geometric_route_pairs']} | "
            f"{row['n_first_step_distinct_swap_pairs']} | {row['route_length_cells']} | "
            f"{geom['max_separation_xy']:.4f}/{geom['cell_width_xy']:.4f}"
        )
    g = result["geometry_validation"]
    print(
        f"geometry converted/pass={g['branches_converted']}/{g['branches_passing_one_cell_width_separation']} "
        f"T={g['resample_points']} cell_width={g['cell_width_xy']:.4f} "
        f"max_separation_range=[{g['minimum_branch_separation_xy']:.4f}, "
        f"{g['maximum_branch_separation_xy']:.4f}]"
    )
    print("E(tau) compatibility and risk/sanity-check design are included in the JSON report.")
    print(f"json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
