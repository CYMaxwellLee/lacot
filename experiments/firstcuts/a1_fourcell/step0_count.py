#!/usr/bin/env python3
"""Count matched val-trajectory pairs that take different BFS-shortest branches.

This is the A1 gate. It reads the maze from OGBench's environment API and reads
the official validation split through the same OGBench loader used by the repo.
It does not load a model or run the four-cell table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lacot.intent import traj_to_cells
from lacot.subgoal import grid_bfs


DEFAULT_ENV = "pointmaze-large-stitch-v0"
DEFAULT_CONDITION_RADIUS = 2.0
PREREGISTERED_MIN_PAIRS = 30
MIN_BRANCH_DISTANCE = 2


@dataclass(frozen=True)
class RouteSample:
    episode: int
    start_row: int
    goal_row: int
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    route: tuple[int, ...]
    first_step: tuple[int, int]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]


@dataclass(frozen=True)
class RoutePair:
    pair_id: int
    sample_a: RouteSample
    sample_b: RouteSample
    start_delta: float
    goal_delta: float
    n_routes_in_condition_group: int


def _default_dataset_dir(env_name: str) -> Path:
    candidates = []
    if os.environ.get("OGBENCH_DATA_DIR"):
        candidates.append(Path(os.environ["OGBENCH_DATA_DIR"]).expanduser())
    candidates.extend(
        [
            Path.home() / "data" / "ogbench",
            Path.home() / ".ogbench" / "data",
            Path("/archive/cymaxwelllee/data/ogbench"),
        ]
    )
    for candidate in candidates:
        if (candidate / f"{env_name}.npz").is_file() and (
            candidate / f"{env_name}-val.npz"
        ).is_file():
            return candidate
    return candidates[0] if candidates else Path.home() / ".ogbench" / "data"


def load_official_data(env_name: str, dataset_dir: str | Path):
    """Return env, train, val via ogbench.make_env_and_datasets."""
    import ogbench

    return ogbench.make_env_and_datasets(env_name, dataset_dir=str(dataset_dir))


def _episode_spans(terminals: np.ndarray) -> list[tuple[int, int]]:
    ends = np.flatnonzero(np.asarray(terminals, dtype=bool))
    if not len(ends):
        raise ValueError("official validation split has no terminal rows")
    if ends[-1] != len(terminals) - 1:
        raise ValueError(
            f"validation tail is unterminated: final row={len(terminals)-1}, "
            f"last terminal={ends[-1]}"
        )
    starts = np.concatenate(([0], ends[:-1] + 1))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _free_cells(occ: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, c)) for c in np.argwhere(occ)]


def _cell_mapper(env: Any, occ: np.ndarray):
    cells = _free_cells(occ)
    centers = np.asarray([env.unwrapped.ij_to_xy(c) for c in cells], dtype=np.float64)

    def xy_to_cell(xy) -> tuple[int, int]:
        point = np.asarray(xy, dtype=np.float64)[:2]
        idx = int(np.square(centers - point).sum(axis=1).argmin())
        return cells[idx]

    return xy_to_cell


def _shortest_next_steps(
    occ: np.ndarray, src: tuple[int, int], dst: tuple[int, int]
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    dist_to_goal = grid_bfs(occ, dst)
    dist_from_start = grid_bfs(occ, src)
    distance = dist_to_goal.get(src)
    if distance is None:
        return dist_to_goal, set()
    legal = set()
    i, j = src
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (i + di, j + dj)
        if dist_to_goal.get(nxt) == distance - 1:
            legal.add(nxt)
    # Referencing both BFS maps also verifies the endpoint is in the same component.
    assert src in dist_from_start and dst in dist_to_goal
    return dist_to_goal, legal


def _route_code(route_cells: list[tuple[int, int]], width: int) -> tuple[int, ...]:
    """Maze corridor ids, with consecutive duplicate cells removed by intent.traj_to_cells."""
    return tuple(int(i * width + j) for i, j in route_cells)


def collect_route_pairs(
    env_name: str = DEFAULT_ENV,
    dataset_dir: str | Path | None = None,
    condition_radius: float = DEFAULT_CONDITION_RADIUS,
) -> tuple[Any, dict, dict, list[RoutePair], dict]:
    """Extract episode-disjoint route pairs from official validation trajectories.

    Conditions are grouped by the same ordered start/goal maze cells. Candidate
    trajectories must follow an observed, BFS-shortest route whose first step is
    one of at least two equally short choices. Matched starts and goals must each
    be within ``condition_radius`` in environment XY units. A validation episode
    can appear in at most one selected pair.
    """
    if dataset_dir is None:
        dataset_dir = _default_dataset_dir(env_name)
    env, train_ds, val_ds = load_official_data(env_name, dataset_dir)
    maze = np.asarray(env.unwrapped.maze_map)
    occ = maze == 0
    if not occ.any():
        raise ValueError("OGBench maze_map contains no passable cells (expected value 0)")

    observations = np.asarray(val_ds["observations"], dtype=np.float32)
    terminals = np.asarray(val_ds["terminals"], dtype=bool)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError(f"expected val observations [N, >=2], got {observations.shape}")
    xy_to_cell = _cell_mapper(env, occ)
    width = int(maze.shape[1])

    # group_key -> episode -> route code -> best sample
    candidates: dict[
        tuple[tuple[int, int], tuple[int, int]],
        dict[int, dict[tuple[int, ...], RouteSample]],
    ] = {}
    route_histogram: dict[int, int] = {}
    episode_spans = _episode_spans(terminals)

    for episode, (lo, hi) in enumerate(episode_spans):
        episode_xy = observations[lo : hi + 1, :2]
        cells = traj_to_cells(episode_xy, xy_to_cell)
        # Recover the first row index for each consecutive-deduplicated cell.
        run_cells: list[tuple[int, int]] = []
        run_rows: list[int] = []
        prior = None
        for local_row, xy in enumerate(episode_xy):
            cell = tuple(xy_to_cell(xy))
            if cell != prior:
                run_cells.append(cell)
                run_rows.append(lo + local_row)
                prior = cell
        # Use the existing route conversion as the route-mark source and guard
        # that it agrees with the indices retained for segment reconstruction.
        if cells != run_cells:
            raise AssertionError("intent.traj_to_cells route marker and row indices diverged")

        # Distinct sample per episode/endpoint/route. Prefer endpoints nearest
        # the corresponding maze-cell centers, then shorter recorded windows.
        for i in range(len(run_cells) - MIN_BRANCH_DISTANCE):
            src = run_cells[i]
            ds = grid_bfs(occ, src)
            for j in range(i + MIN_BRANCH_DISTANCE, len(run_cells)):
                dst = run_cells[j]
                distance = ds.get(dst)
                if distance is None or distance != j - i:
                    continue  # observed route is not a BFS-shortest route
                _, legal_next = _shortest_next_steps(occ, src, dst)
                if len(legal_next) < 2:
                    continue  # no genuine shortest-path branch at this start
                route_cells = run_cells[i : j + 1]
                first_step = (route_cells[1][0] - src[0], route_cells[1][1] - src[1])
                if (src[0] + first_step[0], src[1] + first_step[1]) not in legal_next:
                    raise AssertionError("observed shortest route starts outside BFS legal set")
                route = _route_code(route_cells, width)
                start_row, goal_row = run_rows[i], run_rows[j]
                start_xy = tuple(float(x) for x in observations[start_row, :2])
                goal_xy = tuple(float(x) for x in observations[goal_row, :2])
                sample = RouteSample(
                    episode=episode,
                    start_row=start_row,
                    goal_row=goal_row,
                    start_cell=src,
                    goal_cell=dst,
                    route=route,
                    first_step=first_step,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                )
                key = (src, dst)
                route_records = candidates.setdefault(key, {}).setdefault(episode, {})
                prior_sample = route_records.get(route)
                if prior_sample is None or (goal_row - start_row, start_row) < (
                    prior_sample.goal_row - prior_sample.start_row,
                    prior_sample.start_row,
                ):
                    route_records[route] = sample

    # Report route richness for all condition groups with at least one observed
    # shortest route and a genuine BFS branch, including groups with only one
    # observed route (the key diagnostic when the gate has no qualifying pairs).
    for by_episode in candidates.values():
        routes = {route for by_route in by_episode.values() for route in by_route}
        if routes:
            route_histogram[len(routes)] = route_histogram.get(len(routes), 0) + 1

    # One best compatible pair per ordered maze-cell condition, then a global
    # episode-disjoint selection. This keeps the gate count from multiplying
    # near-duplicate windows or reusing one validation episode many times.
    group_options = []
    centers = {c: np.asarray(env.unwrapped.ij_to_xy(c), dtype=np.float64) for c in _free_cells(occ)}
    for key, by_episode in candidates.items():
        by_route: dict[tuple[int, ...], list[RouteSample]] = {}
        for records in by_episode.values():
            for route, sample in records.items():
                by_route.setdefault(route, []).append(sample)
        route_count = len(by_route)
        if route_count < 2:
            continue
        options = []
        routes = sorted(by_route)
        for ri, route_a in enumerate(routes):
            for route_b in routes[ri + 1 :]:
                sample_a_step = by_route[route_a][0].first_step
                sample_b_step = by_route[route_b][0].first_step
                if sample_a_step == sample_b_step:
                    continue
                for a in by_route[route_a]:
                    for b in by_route[route_b]:
                        if a.episode == b.episode:
                            continue
                        start_delta = float(np.linalg.norm(np.asarray(a.start_xy) - b.start_xy))
                        goal_delta = float(np.linalg.norm(np.asarray(a.goal_xy) - b.goal_xy))
                        if start_delta > condition_radius or goal_delta > condition_radius:
                            continue
                        # Prefer close continuous conditions and then stable ids.
                        options.append(
                            (
                                start_delta + goal_delta,
                                start_delta,
                                goal_delta,
                                min(a.episode, b.episode),
                                max(a.episode, b.episode),
                                a,
                                b,
                                route_count,
                            )
                        )
        if options:
            options.sort(key=lambda x: x[:5])
            candidate_episodes = {episode for option in options for episode in option[3:5]}
            group_options.append((len(candidate_episodes), key, options))

    # Constrained endpoint groups first; use no source episode more than once.
    group_options.sort(key=lambda x: (x[0], x[1]))
    used_episodes: set[int] = set()
    pairs: list[RoutePair] = []
    for _, _, options in group_options:
        chosen = next(
            (row for row in options if row[5].episode not in used_episodes and row[6].episode not in used_episodes),
            None,
        )
        if chosen is None:
            continue
        _, start_delta, goal_delta, _, _, a, b, route_count = chosen
        used_episodes.update((a.episode, b.episode))
        if a.first_step > b.first_step:
            a, b = b, a
        pairs.append(
            RoutePair(
                pair_id=len(pairs),
                sample_a=a,
                sample_b=b,
                start_delta=start_delta,
                goal_delta=goal_delta,
                n_routes_in_condition_group=route_count,
            )
        )

    metadata = {
        "env": env_name,
        "dataset_dir": str(dataset_dir),
        "official_maze_shape": list(maze.shape),
        "official_maze_free_cells": int(occ.sum()),
        "val_rows": int(len(observations)),
        "val_episodes": int(len(episode_spans)),
        "condition_radius_xy": float(condition_radius),
        "min_bfs_distance_cells": MIN_BRANCH_DISTANCE,
        "distinct_route_groups_by_route_count": {
            str(k): int(v) for k, v in sorted(route_histogram.items())
        },
        "eligible_condition_groups_before_episode_disjoint_matching": len(group_options),
        "selected_episode_disjoint_pairs": len(pairs),
        "selected_unique_episodes": len(used_episodes),
    }
    return env, train_ds, val_ds, pairs, metadata


def report(pairs: list[RoutePair], metadata: dict) -> dict:
    by_pair_diversity: dict[str, int] = {}
    for pair in pairs:
        key = str(pair.n_routes_in_condition_group)
        by_pair_diversity[key] = by_pair_diversity.get(key, 0) + 1
    out = {
        "gate": "PREREGISTERED: n_pairs >= 30; n < 30 means STOP AND REPORT",
        "n_qualified_pairs": len(pairs),
        "route_diversity_distribution": metadata["distinct_route_groups_by_route_count"],
        "selected_pairs_by_condition_route_count": dict(sorted(by_pair_diversity.items())),
        "metadata": metadata,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--condition-radius", type=float, default=DEFAULT_CONDITION_RADIUS)
    parser.add_argument("--json", action="store_true", help="print compact JSON only")
    args = parser.parse_args(argv)
    dataset_dir = args.dataset_dir or _default_dataset_dir(args.env)
    _, _, _, pairs, metadata = collect_route_pairs(args.env, dataset_dir, args.condition_radius)
    out = report(pairs, metadata)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    else:
        print("A1 STEP 0 — OFFICIAL OGBENCH VALIDATION ROUTE-PAIR GATE")
        print(out["gate"])
        print(f"env={metadata['env']}  validation episodes={metadata['val_episodes']}  rows={metadata['val_rows']}")
        print(
            f"official maze={metadata['official_maze_shape']}  free cells={metadata['official_maze_free_cells']}"
        )
        print(
            f"condition rule: same ordered start/goal maze cells; each XY endpoint delta <= "
            f"{metadata['condition_radius_xy']:.3f}; minimum BFS distance="
            f"{metadata['min_bfs_distance_cells']} cells"
        )
        print(f"eligible condition groups before episode-disjoint matching={metadata['eligible_condition_groups_before_episode_disjoint_matching']}")
        print(f"n_qualified_pairs={out['n_qualified_pairs']}")
        print(f"route_diversity_distribution={json.dumps(out['route_diversity_distribution'], sort_keys=True)}")
        print(f"selected_pairs_by_condition_route_count={json.dumps(out['selected_pairs_by_condition_route_count'], sort_keys=True)}")
        print(f"decision={'PASS GATE' if len(pairs) >= PREREGISTERED_MIN_PAIRS else 'STOP AND REPORT'}")
        print(f"dataset_dir={metadata['dataset_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
