#!/usr/bin/env python3
"""Approved synthetic A1 branch run and held-out E-input sanity comparison."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lacot.intent import anchors_resample, cells_to_anchors, traj_to_cells
from run_a1 import A1Model, DEFAULT_CKPT, _normalization, _trajectory_window, run_experiment
from step0_count import RoutePair, RouteSample, _cell_mapper, _episode_spans, load_official_data

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "step05_synth_count.json"
ENV = "pointmaze-large-stitch-v0"
DATA_DIR = Path(os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/data/ogbench"))


def make_synthetic_pairs(env, source: dict, t_cap: int, limit: int | None = None):
    """Materialize all first-step-distinct JSON route pairs as 128-row XY windows."""
    observations: list[np.ndarray] = []
    pairs: list[RoutePair] = []
    width = int(np.asarray(env.unwrapped.maze_map).shape[1])
    for branch_index, branch in enumerate(source["branches"]):
        src, dst = tuple(branch["src"]), tuple(branch["dst"])
        routes = branch["routes"]
        for route_a, route_b in itertools.combinations(routes, 2):
            if route_a["first_next_cell"] == route_b["first_next_cell"]:
                continue
            samples = []
            for route in (route_a, route_b):
                cells = [tuple(cell) for cell in route["cells"]]
                assert cells[0] == src and cells[-1] == dst
                assert tuple(route["corridor_code"]) == tuple(i * width + j for i, j in cells)
                xy = anchors_resample(cells_to_anchors(cells, env.unwrapped.ij_to_xy), t_cap)
                first = (cells[1][0] - src[0], cells[1][1] - src[1])
                start = len(observations)
                observations.extend(xy)
                samples.append(RouteSample(
                    episode=2 * len(pairs) + len(samples), start_row=start,
                    goal_row=start + t_cap - 1, start_cell=src, goal_cell=dst,
                    route=tuple(route["corridor_code"]), first_step=first,
                    start_xy=tuple(map(float, xy[0])), goal_xy=tuple(map(float, xy[-1])),
                ))
            pairs.append(RoutePair(
                pair_id=len(pairs), sample_a=samples[0], sample_b=samples[1],
                start_delta=0.0, goal_delta=0.0,
                n_routes_in_condition_group=len(routes),
            ))
            if limit is not None and len(pairs) >= limit:
                return {"observations": np.asarray(observations, dtype=np.float32)}, pairs
    assert len(pairs) == source["production"]["n_first_step_distinct_swap_pairs"]
    return {"observations": np.asarray(observations, dtype=np.float32)}, pairs


def _cos_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.reshape(len(a), -1).astype(np.float64)
    bb = b.reshape(len(b), -1).astype(np.float64)
    return 1.0 - np.sum(aa * bb, axis=-1) / (
        np.linalg.norm(aa, axis=-1) * np.linalg.norm(bb, axis=-1)
    )


def _distribution(values: np.ndarray) -> dict:
    return {"min": float(np.min(values)), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values))}


@torch.no_grad()
def sanity(env, train_ds: dict, val_ds: dict, model: A1Model) -> dict:
    observations = np.asarray(val_ds["observations"], dtype=np.float32)
    xy_to_cell = _cell_mapper(env, np.asarray(env.unwrapped.maze_map) == 0)
    mean, std = _normalization(train_ds)
    native, rebuilt, selected = [], [], []
    # One reproducible segment from each of 20 distinct held-out val episodes.
    # Use the first 5-cell window with enough native time samples.
    for episode, (lo, hi) in enumerate(_episode_spans(val_ds["terminals"])):
        first_rows = []
        last = None
        for row in range(lo, hi + 1):
            cell = tuple(xy_to_cell(observations[row, :2]))
            if cell != last:
                first_rows.append(row)
                last = cell
        window = next(((first_rows[i], first_rows[i + 4])
                       for i in range(len(first_rows) - 4)
                       if first_rows[i + 4] - first_rows[i] >= 20), None)
        if window is None:
            continue
        start, end = window
        segment = observations[start:end + 1, :2]
        cells = traj_to_cells(segment, xy_to_cell)
        if len(cells) < 5:
            continue
        native_xy = _trajectory_window(observations,
            type("Window", (), {"start_row": start, "goal_row": end})(),
            mean[:2], std[:2], model.t_cap)
        rebuilt_xy = anchors_resample(cells_to_anchors(cells, env.unwrapped.ij_to_xy), model.t_cap)
        native.append(native_xy)
        rebuilt.append(((rebuilt_xy - mean[:2]) / std[:2]).astype(np.float32))
        selected.append({"episode": episode, "start_row": start, "end_row": end,
                         "n_cells": len(cells)})
        if len(selected) == 20:
            break
    if len(selected) != 20:
        raise RuntimeError(f"needed 20 held-out real segments, found {len(selected)}")
    native_e = model.encode_trajectory(torch.from_numpy(np.stack(native)).to(model.device)).cpu().numpy()
    rebuilt_e = model.encode_trajectory(torch.from_numpy(np.stack(rebuilt)).to(model.device)).cpu().numpy()
    within = _cos_distance(native_e, rebuilt_e)
    between = np.asarray([_cos_distance(native_e[i:i+1], native_e[j:j+1])[0]
                          for i in range(20) for j in range(i+1, 20)])
    ratio = float(np.median(within) / np.median(between))
    return {"status": "NUMBERS_ONLY_LEAD_JUDGES", "n_segments": 20,
            "n_between_pairs": len(between), "same_segment_cosine_distance": _distribution(within),
            "different_segment_cosine_distance": _distribution(between),
            "median_ratio": ratio, "segments": selected,
            "provenance": {"split": "official validation (held-out)",
                "native": "row-index linear resample to T=128",
                "rebuilt": "traj_to_cells -> cells_to_anchors -> anchors_resample(T=128)",
                "normalization": "training observations XY mean/std", "weights": model.weight_source}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sanity", "main"))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--dataset-dir", default=str(DATA_DIR))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    env, train_ds, val_ds = load_official_data(ENV, args.dataset_dir)
    model = A1Model(args.ckpt, torch.device("cpu"))
    if args.mode == "sanity":
        output = sanity(env, train_ds, val_ds, model)
        print(json.dumps({k: output[k] for k in ("n_segments", "n_between_pairs", "same_segment_cosine_distance", "different_segment_cosine_distance", "median_ratio")}, indent=2))
    else:
        source = json.loads(SOURCE.read_text())
        synthetic_ds, pairs = make_synthetic_pairs(env, source, model.t_cap)
        output = run_experiment(env, train_ds, synthetic_ds, pairs, model, None, 20260927)
        output["provenance"] = {"trajectory_source": "synthetic-arclength",
            "source_json": str(SOURCE), "source_pair_count": source["production"]["n_first_step_distinct_swap_pairs"],
            "condition": "shared source/goal maze cell centers for both routes",
            "scoring": "route-specific first-step match for S_c; all BFS shortest next steps separately counted in bfs_legal"}
        output["pair_routes"] = [{"pair_id": p.pair_id, "route_a": list(p.sample_a.route),
                                  "route_b": list(p.sample_b.route), "first_step_a": list(p.sample_a.first_step),
                                  "first_step_b": list(p.sample_b.first_step)} for p in pairs]
        rates = {cell: {field: float(np.mean([r[field] for r in output["four_cell"] if r["cell"] == cell]))
                 for field in ("route_match_to_e", "route_match_to_c", "bfs_legal")}
                 for cell in ("H(c_A,e_A)", "H(c_A,e_B)", "H(c_B,e_A)", "H(c_B,e_B)")}
        output["summary"]["four_cell_rates"] = rates
        print(f"n_pairs={output['n_pairs']} n_cells={output['n_cells']}")
        print(json.dumps({"S_c": output["summary"]["S_c"], "four_cell_rates": rates,
                          "read_heads": output["summary"]["read_heads"]}, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        print(f"json={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
