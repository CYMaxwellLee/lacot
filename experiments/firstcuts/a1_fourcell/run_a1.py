#!/usr/bin/env python3
"""Run the preregistered A1 teacher-code four-cell and read-head comparison."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.subgoal import grid_bfs
from step0_count import (
    DEFAULT_CONDITION_RADIUS,
    DEFAULT_ENV,
    PREREGISTERED_MIN_PAIRS,
    RoutePair,
    _default_dataset_dir,
    collect_route_pairs,
)


DEFAULT_CKPT = (
    REPO_ROOT
    / "results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu"
    "_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s0.pt"
)
SMOKE_MAX_PAIRS = 8
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 10_000


def sota_mlp(in_dim: int, hidden_dim: int, out_dim: int, n: int = 2) -> nn.Sequential:
    """Same MLP builder and layer order as scratch_lacot_rollout.py:sota_mlp."""
    layers: list[nn.Module] = []
    width = in_dim
    for _ in range(n):
        linear = nn.Linear(width, hidden_dim)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.GELU(), nn.LayerNorm(hidden_dim)))
        width = hidden_dim
    linear = nn.Linear(width, out_dim)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    layers.append(linear)
    return nn.Sequential(*layers)


class Ahead(nn.Module):
    """The trained H signature: ahead(cond, u) -> [B, CHUNK, ADIM]."""

    def __init__(self, cond_dim: int, latent_dim: int, chunk: int, action_dim: int):
        super().__init__()
        self.net = sota_mlp(cond_dim + latent_dim, 512, chunk * action_dim, n=3)
        self.chunk = chunk
        self.action_dim = action_dim

    def forward(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.cat((cond, u.reshape(u.shape[0], -1)), dim=-1)
        return self.net(x).reshape(-1, self.chunk, self.action_dim)


class BCHead(nn.Module):
    """The trained bc_head signature: bc_head(cond) -> [B, CHUNK, ADIM]."""

    def __init__(self, cond_dim: int, chunk: int, action_dim: int):
        super().__init__()
        self.net = sota_mlp(cond_dim, 512, chunk * action_dim, n=3)
        self.chunk = chunk
        self.action_dim = action_dim

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).reshape(-1, self.chunk, self.action_dim)


class FrozenTeacherEncoder(nn.Module):
    """The state-track E(τ) from scratch_lacot_rollout.py:etarget."""

    def __init__(self, d_model: int, k: int, t_cap: int):
        super().__init__()
        self.traj_enc = sota_mlp(2, 512, 512)
        self.e_pooler = PerceiverPooler(
            512, d_model, k, num_layers=2, num_heads=4, max_len=max(512, t_cap)
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        batch, steps, xy_dim = trajectory.shape
        if xy_dim != 2:
            raise ValueError(f"E(τ) requires normalized XY trajectories [B,T,2], got {trajectory.shape}")
        features = self.traj_enc(trajectory.reshape(batch * steps, 2)).reshape(batch, steps, 512)
        mask = torch.zeros((batch, steps), dtype=torch.bool, device=trajectory.device)
        return self.e_pooler(features, key_padding_mask=mask)


class A1Model(nn.Module):
    def __init__(self, ckpt_path: str | Path, device: torch.device, use_ema: bool = False):
        super().__init__()
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.ckpt_path}; pass --ckpt")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {})
        if cfg.get("ENC_OBJ") != "recon_ictr":
            raise ValueError(f"A1 requires ENC_OBJ=recon_ictr, got {cfg.get('ENC_OBJ')!r}")
        if "intent_ad" in ckpt:
            raise ValueError("A1 four-cell requires the no-intent recon_ictr head; checkpoint has intent_ad")
        if "vq" in ckpt:
            raise ValueError("A1 four-cell requires continuous E(τ); checkpoint has VQ enabled")

        self.k = int(cfg["K"])
        self.cond_dim = int(cfg["COND"])
        self.chunk = int(cfg["CHUNK"])
        self.d_model = int(cfg["D_MODEL"])
        self.t_cap = int(cfg["T_CAP"])
        obs_dim = int(ckpt["cond_enc"]["0.weight"].shape[1])
        # Derive the action dimension from the checkpoint head output and saved
        # CHUNK, then validate against bc_head's output; no dimensions are guessed.
        output_width = int(ckpt["ahead"]["net.9.weight"].shape[0])
        if output_width % self.chunk:
            raise ValueError("ahead output width is not divisible by saved CHUNK")
        action_dim = output_width // self.chunk
        bc_width = int(ckpt["bc_head"]["net.9.weight"].shape[0])
        if bc_width != output_width:
            raise ValueError("ahead and bc_head checkpoint action widths differ")
        if obs_dim != 2 or action_dim != 2:
            raise ValueError(
                f"this A1 implementation uses pointmaze XY actions; checkpoint has obs={obs_dim}, actions={action_dim}"
            )

        self.cond_enc = sota_mlp(obs_dim, 512, 512)
        self.cond_head = sota_mlp(1024, 512, self.cond_dim)
        self.flow = Flow(token_dim=self.d_model, seq_len=self.k, n_blocks=4, cond_dim=self.cond_dim)
        self.ahead = Ahead(self.cond_dim, self.k * self.d_model, self.chunk, action_dim)
        self.bc_head = BCHead(self.cond_dim, self.chunk, action_dim)
        self.encoder = FrozenTeacherEncoder(self.d_model, self.k, self.t_cap)

        for key, module in (
            ("cond_enc", self.cond_enc),
            ("cond_head", self.cond_head),
            ("flow", self.flow),
            ("ahead", self.ahead),
            ("bc_head", self.bc_head),
            ("traj_enc", self.encoder.traj_enc),
            ("e_pooler", self.encoder.e_pooler),
        ):
            module.load_state_dict(ckpt[key])

        # The rollout checkpoint's optional EMA is explicitly the five-module
        # deployed flow/head path. Its frozen E(τ) remains the saved raw encoder.
        self.weight_source = "raw"
        if use_ema:
            ema = ckpt.get("ema")
            if not ema:
                raise ValueError("--ema requested but checkpoint contains no EMA weights")
            ema_modules = {
                "cond_enc": self.cond_enc,
                "cond_head": self.cond_head,
                "flow": self.flow,
                "ahead": self.ahead,
                "bc_head": self.bc_head,
            }
            for name, module in ema_modules.items():
                if name not in ema:
                    raise ValueError(f"checkpoint EMA is missing {name}")
                module.load_state_dict(ema[name])
            self.weight_source = "EMA heads/conditioner; raw frozen E(τ)"

        self.to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.action_dim = action_dim

    @torch.no_grad()
    def encode_condition(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.cond_head(torch.cat((self.cond_enc(states), self.cond_enc(goals)), dim=-1))

    @torch.no_grad()
    def encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.encoder(trajectory)


def _normalization(train_ds: dict) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(train_ds["observations"], dtype=np.float32)
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = (observations.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
    return mean, std


def _trajectory_window(
    observations: np.ndarray, sample, mean_xy: np.ndarray, std_xy: np.ndarray, t_cap: int
) -> np.ndarray:
    times = np.linspace(sample.start_row, sample.goal_row, t_cap, dtype=np.float64)
    low = np.floor(times).astype(np.int64)
    high = np.minimum(low + 1, sample.goal_row)
    weight = (times - low)[:, None].astype(np.float32)
    xy = observations[low, :2] * (1.0 - weight) + observations[high, :2] * weight
    return ((xy - mean_xy) / std_xy).astype(np.float32)


def _direction_vectors(env) -> dict[tuple[int, int], np.ndarray]:
    occ = np.asarray(env.unwrapped.maze_map) == 0
    vectors: dict[tuple[int, int], np.ndarray] = {}
    for cell in map(tuple, np.argwhere(occ)):
        cell = tuple(map(int, cell))
        here = np.asarray(env.unwrapped.ij_to_xy(cell), dtype=np.float64)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + di, cell[1] + dj)
            if 0 <= nxt[0] < occ.shape[0] and 0 <= nxt[1] < occ.shape[1] and occ[nxt]:
                vec = np.asarray(env.unwrapped.ij_to_xy(nxt), dtype=np.float64) - here
                vec /= np.linalg.norm(vec)
                if (di, dj) in vectors and not np.allclose(vectors[(di, dj)], vec, atol=1e-7):
                    raise ValueError("official ij_to_xy is not a consistent cardinal maze transform")
                vectors[(di, dj)] = vec
    expected = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    if set(vectors) != expected:
        raise ValueError(f"official maze does not expose all four directions: {sorted(vectors)}")
    return vectors


def _predicted_directions(action: np.ndarray, vectors: dict[tuple[int, int], np.ndarray]) -> set[tuple[int, int]]:
    action = np.asarray(action, dtype=np.float64).reshape(-1)[:2]
    if np.linalg.norm(action) < 1e-8:
        return set()
    scores = {key: float(action @ vector) for key, vector in vectors.items()}
    best = max(scores.values())
    # A geometric tie is treated as a set of equally supported headings.
    return {key for key, score in scores.items() if best - score <= 1e-7 * max(1.0, abs(best))}


def _legal_next_dirs(occ: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> set[tuple[int, int]]:
    from_goal = grid_bfs(occ, dst)
    distance = from_goal.get(src)
    if distance is None:
        raise ValueError(f"unreachable pair in gate data: {src} -> {dst}")
    return {
        (di, dj)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if from_goal.get((src[0] + di, src[1] + dj)) == distance - 1
    }


def _route_score(
    action: np.ndarray,
    target_step: tuple[int, int],
    legal_steps: set[tuple[int, int]],
    vectors: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    predicted = _predicted_directions(action, vectors)
    return {
        "route_match": bool(target_step in predicted),
        "bfs_legal": bool(predicted.intersection(legal_steps)),
        "predicted_dirs": [list(d) for d in sorted(predicted)],
        "legal_dirs": [list(d) for d in sorted(legal_steps)],
        "chance_route_match": 1.0 / len(legal_steps),
    }


def _base_inverse(flow: Flow, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """F(cond,z): the same reverse-block path as Flow.sample, with supplied z."""
    u = z
    for i in reversed(range(len(flow.blocks))):
        if i < len(flow.blocks) - 1:
            u = flow.perm.inverse(u)
        u = flow.blocks[i].inverse(u, flow._cond_for_block(cond, i))
    return u


def _bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _tensor_rows(rows: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=device)


@torch.no_grad()
def run_experiment(
    env,
    train_ds: dict,
    val_ds: dict,
    pairs: list[RoutePair],
    model: A1Model,
    pair_limit: int | None,
    xi_seed: int,
) -> dict:
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    if not pairs:
        raise ValueError("cannot run four-cell without selected route pairs")

    observations = np.asarray(val_ds["observations"], dtype=np.float32)
    mean, std = _normalization(train_ds)
    state_rows, goal_rows, traj_rows = [], [], []
    route_steps_a, route_steps_b, legal_steps = [], [], []
    occ = np.asarray(env.unwrapped.maze_map) == 0
    for pair in pairs:
        a, b = pair.sample_a, pair.sample_b
        state_rows.extend(((observations[a.start_row] - mean) / std, (observations[b.start_row] - mean) / std))
        goal_rows.extend(((observations[a.goal_row] - mean) / std, (observations[b.goal_row] - mean) / std))
        traj_rows.extend(
            (
                _trajectory_window(observations, a, mean[:2], std[:2], model.t_cap),
                _trajectory_window(observations, b, mean[:2], std[:2], model.t_cap),
            )
        )
        route_steps_a.append(a.first_step)
        route_steps_b.append(b.first_step)
        legal_steps.append(_legal_next_dirs(occ, a.start_cell, a.goal_cell))
        if legal_steps[-1] != _legal_next_dirs(occ, b.start_cell, b.goal_cell):
            raise AssertionError("paired conditions must have the same BFS legal next-step set")
        if a.first_step == b.first_step:
            raise AssertionError("paired routes must choose different initial shortest-path branches")

    state_t = _tensor_rows(state_rows, model.device)
    goal_t = _tensor_rows(goal_rows, model.device)
    traj_t = _tensor_rows(traj_rows, model.device)
    cond = model.encode_condition(state_t, goal_t)
    e = model.encode_trajectory(traj_t)
    n = len(pairs)
    c_a, c_b = cond[0::2], cond[1::2]
    e_a, e_b = e[0::2], e[1::2]

    # Cell order: AA, AB, BA, BB. Only e is exchanged: the two c inputs remain
    # exactly c_A/c_B in all corresponding cells.
    cond_four = torch.stack((c_a, c_a, c_b, c_b), dim=1).reshape(4 * n, model.cond_dim)
    e_four = torch.stack((e_a, e_b, e_a, e_b), dim=1).reshape(4 * n, model.k, model.d_model)
    h_four = model.ahead(cond_four, e_four).cpu().numpy()[:, 0, :]
    h_four = h_four.reshape(n, 4, model.action_dim)

    # H(c,0) is the actual negative wiring control; the same all-zero e is used
    # for each fixed condition. It is kept distinct from F(cond,0).
    zero_e = torch.zeros_like(e)
    h_zero_e = model.ahead(cond, zero_e).cpu().numpy()[:, 0, :].reshape(n, 2, model.action_dim)

    # Read-head targets below are ordered A,B for each pair. Keep the
    # conditioner and all flow outputs in that same interleaved order.
    all_cond = cond
    bc_actions = model.bc_head(all_cond).cpu().numpy()[:, 0, :]
    true_e_actions = h_four[:, (0, 3), :].reshape(2 * n, model.action_dim)
    zero_base = torch.zeros((2 * n, model.k, model.d_model), dtype=torch.float32, device=model.device)
    f_zero = _base_inverse(model.flow, zero_base, all_cond)
    f_zero_actions = model.ahead(all_cond, f_zero).cpu().numpy()[:, 0, :]
    generator = torch.Generator(device=model.device).manual_seed(xi_seed)
    xi = torch.randn(
        (2 * n, model.k, model.d_model), dtype=torch.float32, device=model.device, generator=generator
    )
    f_xi = _base_inverse(model.flow, xi, all_cond)
    f_xi_actions = model.ahead(all_cond, f_xi).cpu().numpy()[:, 0, :]

    vectors = _direction_vectors(env)
    cell_rows = []
    pair_sc = []
    positive_matches, null_matches, chance_rows = [], [], []
    readout_hits = {key: [] for key in ("bc_head(cond)", "ahead(cond,E(τ))", "ahead(cond,F(cond,0))", "ahead(cond,F(cond,ξ))")}
    readout_legal = {key: [] for key in readout_hits}
    for i, pair in enumerate(pairs):
        a, b = pair.sample_a, pair.sample_b
        ra, rb = route_steps_a[i], route_steps_b[i]
        legal = legal_steps[i]
        # Four-cell predictions are scored against the e-owned observed route;
        # BFS legality is reported separately, with every equally-short next
        # step included (no deterministic path tie-break is used).
        specs = (
            ("H(c_A,e_A)", h_four[i, 0], ra, ra, "A", "A"),
            ("H(c_A,e_B)", h_four[i, 1], rb, ra, "A", "B"),
            ("H(c_B,e_A)", h_four[i, 2], ra, rb, "B", "A"),
            ("H(c_B,e_B)", h_four[i, 3], rb, rb, "B", "B"),
        )
        for name, action, e_target, c_target, c_name, e_name in specs:
            scored = _route_score(action, e_target, legal, vectors)
            scored["cond_route_match"] = bool(c_target in _predicted_directions(action, vectors))
            cell_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "cell": name,
                    "c_owner": c_name,
                    "e_owner": e_name,
                    "route_match_to_e": scored["route_match"],
                    "route_match_to_c": scored["cond_route_match"],
                    "bfs_legal": scored["bfs_legal"],
                    "predicted_dirs": scored["predicted_dirs"],
                    "legal_dirs": scored["legal_dirs"],
                }
            )

        ab = _route_score(h_four[i, 1], rb, legal, vectors)
        ab_c = _route_score(h_four[i, 1], ra, legal, vectors)
        ba = _route_score(h_four[i, 2], ra, legal, vectors)
        ba_c = _route_score(h_four[i, 2], rb, legal, vectors)
        pair_sc.append((int(ab["route_match"]) - int(ab_c["route_match"]) + int(ba["route_match"]) - int(ba_c["route_match"])) / 2)
        diagonal = (
            _route_score(h_four[i, 0], ra, legal, vectors)["route_match"],
            _route_score(h_four[i, 3], rb, legal, vectors)["route_match"],
        )
        null = (
            _route_score(h_zero_e[i, 0], ra, legal, vectors)["route_match"],
            _route_score(h_zero_e[i, 1], rb, legal, vectors)["route_match"],
        )
        positive_matches.extend(diagonal)
        null_matches.extend(null)
        chance_rows.extend((1.0 / len(legal), 1.0 / len(legal)))

        for j, (target_step, action_idx) in enumerate(((ra, 2 * i), (rb, 2 * i + 1))):
            actions_by_head = {
                "bc_head(cond)": bc_actions[action_idx],
                "ahead(cond,E(τ))": true_e_actions[action_idx],
                "ahead(cond,F(cond,0))": f_zero_actions[action_idx],
                "ahead(cond,F(cond,ξ))": f_xi_actions[action_idx],
            }
            for name, action in actions_by_head.items():
                score = _route_score(action, target_step, legal, vectors)
                readout_hits[name].append(score["route_match"])
                readout_legal[name].append(score["bfs_legal"])

    sc_mean, sc_lo, sc_hi = _bootstrap_mean_ci(np.asarray(pair_sc), BOOTSTRAP_SEED)
    chance = float(np.mean(chance_rows))
    output = {
        "status": "SMOKE" if pair_limit is not None else "FULL",
        "n_pairs": n,
        "n_cells": 4 * n,
        "ckpt": str(model.ckpt_path),
        "weights": model.weight_source,
        "xi_seed": xi_seed,
        "preregistered": {
            "gate_n_pairs_min": PREREGISTERED_MIN_PAIRS,
            "primary": "S_c = mean over pairs of swapped e-route match minus same-c condition-route match",
            "pass": "paired bootstrap percentile 95% CI lower bound > 0",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "route_accuracy": "match e-owned observed first step; report BFS legal-set hit separately",
        },
        "chance_route_match": chance,
        "four_cell": cell_rows,
        "summary": {
            "S_c": {"mean": sc_mean, "paired_bootstrap_95ci": [sc_lo, sc_hi]},
            "diagonal_E_route_match_rate": float(np.mean(positive_matches)),
            "zero_e_route_match_rate": float(np.mean(null_matches)),
            "four_cell_bfs_legal_rate": float(np.mean([r["bfs_legal"] for r in cell_rows])),
            "read_heads": {
                name: {
                    "route_match_rate": float(np.mean(readout_hits[name])),
                    "bfs_legal_rate": float(np.mean(readout_legal[name])),
                }
                for name in readout_hits
            },
        },
    }
    return output


def print_output(output: dict) -> None:
    print(f"A1 FOUR-CELL — {output['status']}")
    print(output["preregistered"]["pass"] + " [PREREGISTERED]")
    print(f"n_pairs={output['n_pairs']}  n_cells={output['n_cells']}")
    print(f"checkpoint={output['ckpt']}  weights={output['weights']}  xi_seed={output['xi_seed']}")
    print(f"chance_route_match={output['chance_route_match']:.4f}")
    print("four cells (action is the trained head's first predicted action; e changes, c stays fixed per row):")
    print("  pair cell          e-route  c-route  BFS-legal  predicted_dirs  legal_dirs")
    for row in output["four_cell"]:
        print(
            f"  {row['pair_id']:>4} {row['cell']:<14} {int(row['route_match_to_e']):>7}"
            f" {int(row['route_match_to_c']):>8} {int(row['bfs_legal']):>10}"
            f" {str(row['predicted_dirs']):>15} {row['legal_dirs']}"
        )
    s = output["summary"]
    ci = s["S_c"]["paired_bootstrap_95ci"]
    print(f"S_c={s['S_c']['mean']:.4f}  paired bootstrap 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
    print(
        f"diagonal E-route match={s['diagonal_E_route_match_rate']:.4f}; "
        f"all-zero-e match={s['zero_e_route_match_rate']:.4f}; "
        f"four-cell BFS-legal={s['four_cell_bfs_legal_rate']:.4f}"
    )
    print("read-head comparison (same paired examples):")
    for name, row in s["read_heads"].items():
        label = name.replace("F(cond,0)", "F(cond,0) [base zero-point inverse image]")
        print(f"  {label}: route-match={row['route_match_rate']:.4f}  BFS-legal={row['bfs_legal_rate']:.4f}")
    if output["status"] == "SMOKE":
        print("smoke only; these values do not trigger the preregistered full-table decision")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--ema", action="store_true", help="use the checkpoint's saved EMA head/conditioner")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--full", action="store_true", help="run all gated pairs")
    run_mode.add_argument("--smoke", action="store_true", help="run at most 8 gated pairs (default)")
    parser.add_argument("--condition-radius", type=float, default=DEFAULT_CONDITION_RADIUS)
    parser.add_argument("--xi-seed", type=int, default=20260927)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)
    dataset_dir = args.dataset_dir or _default_dataset_dir(args.env)
    env, train_ds, val_ds, pairs, metadata = collect_route_pairs(
        args.env, dataset_dir, args.condition_radius
    )
    print(
        f"A1 gate: n_pairs={len(pairs)}; PREREGISTERED minimum={PREREGISTERED_MIN_PAIRS}; "
        f"routes={json.dumps(metadata['distinct_route_groups_by_route_count'], sort_keys=True)}"
    )
    if len(pairs) < PREREGISTERED_MIN_PAIRS:
        print("STOP AND REPORT: n<30; four-cell and read-head forwards are not run.")
        print("No small-sample main result is reported; route-pair design needs a new decision.")
        return 0

    limit = None if args.full else SMOKE_MAX_PAIRS
    if limit is None:
        print("PREREGISTERED full run enabled; the primary result is still gated at n_pairs>=30.")
    else:
        print(f"SMOKE mode: using at most {limit} of {len(pairs)} qualified pairs.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A1Model(args.ckpt, device, use_ema=args.ema)
    output = run_experiment(env, train_ds, val_ds, pairs, model, limit, args.xi_seed)
    print_output(output)
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json={out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
