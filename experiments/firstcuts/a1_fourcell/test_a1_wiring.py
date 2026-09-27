#!/usr/bin/env python3
"""Two-way wiring self-check for the A1 trained encoder/action head."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_a1 import A1Model, DEFAULT_CKPT, SMOKE_MAX_PAIRS, run_experiment
from run_a1_synthetic import SOURCE, make_synthetic_pairs
from step0_count import (
    DEFAULT_CONDITION_RADIUS,
    DEFAULT_ENV,
    PREREGISTERED_MIN_PAIRS,
    _default_dataset_dir,
    collect_route_pairs,
)


class IgnoreE(nn.Module):
    """Mutation fixture: route code is wired to zero regardless of its input."""

    def __init__(self, trained_head):
        super().__init__()
        self.trained_head = trained_head

    def forward(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.trained_head(cond, torch.zeros_like(u))


def assert_two_way_wiring(output: dict, null_tolerance: float = 0.25) -> None:
    summary = output["summary"]
    chance = float(output["chance_route_match"])
    positive = float(summary["diagonal_E_route_match_rate"])
    null = float(summary["zero_e_route_match_rate"])
    if not positive > chance:
        raise AssertionError(
            f"positive wiring did not beat route-choice random: positive={positive:.4f}, chance={chance:.4f}"
        )
    if abs(null - chance) > null_tolerance:
        raise AssertionError(
            f"all-zero-e negative control did not collapse toward random: "
            f"null={null:.4f}, chance={chance:.4f}, tolerance={null_tolerance:.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--condition-radius", type=float, default=DEFAULT_CONDITION_RADIUS)
    parser.add_argument("--xi-seed", type=int, default=20260927)
    parser.add_argument("--null-tolerance", type=float, default=0.25)
    parser.add_argument("--informal-synthetic-two-pair", action="store_true",
                        help="exercise the real forward and IgnoreE mutation on two synthetic pairs")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir or _default_dataset_dir(args.env)
    if args.informal_synthetic_two_pair:
        from step0_count import load_official_data
        env, train_ds, _ = load_official_data(args.env, dataset_dir)
        val_ds, pairs = make_synthetic_pairs(env, json.loads(SOURCE.read_text()), 128, limit=2)
        print("INFORMAL SYNTHETIC TWO-PAIR SMOKE — not a preregistered PASS")
    else:
        env, train_ds, val_ds, pairs, metadata = collect_route_pairs(
            args.env, dataset_dir, args.condition_radius
        )
    if not args.informal_synthetic_two_pair and len(pairs) < PREREGISTERED_MIN_PAIRS:
        print(
            f"WIRING SELF-CHECK: STOP_SKIP — n_pairs={len(pairs)} < "
            f"preregistered {PREREGISTERED_MIN_PAIRS}; no small-sample forward or PASS is allowed."
        )
        print(f"route_diversity_distribution={metadata['distinct_route_groups_by_route_count']}")
        return 0
    smoke_pairs = pairs[:SMOKE_MAX_PAIRS]
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu" if args.informal_synthetic_two_pair else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = A1Model(args.ckpt, device, use_ema=args.ema)

    # First prove the assertions catch the real wiring error: make H ignore e.
    # The expected assertion failure is required before the unmutated check.
    actual_head = model.ahead
    model.ahead = IgnoreE(actual_head)
    mutant_output = run_experiment(
        env, train_ds, val_ds, smoke_pairs, model, len(smoke_pairs), args.xi_seed
    )
    try:
        assert_two_way_wiring(mutant_output, args.null_tolerance)
    except AssertionError as error:
        print(f"MUTATION PROBE: EXPECTED FAIL (e-ignoring H): {error}")
    else:
        print("MUTATION PROBE: FAIL — wiring assertions accepted a head that ignores e")
        return 1
    model.ahead = actual_head

    output = run_experiment(env, train_ds, val_ds, smoke_pairs, model, len(smoke_pairs), args.xi_seed)
    if args.informal_synthetic_two_pair:
        print(f"INFORMAL FORWARD: n_pairs={output['n_pairs']} n_cells={output['n_cells']} "
              f"positive={output['summary']['diagonal_E_route_match_rate']:.4f} "
              f"null={output['summary']['zero_e_route_match_rate']:.4f} "
              f"chance={output['chance_route_match']:.4f} "
              f"S_c={output['summary']['S_c']['mean']:.4f}")
        print("INFORMAL ONLY — no PASS asserted")
        return 0
    try:
        assert_two_way_wiring(output, args.null_tolerance)
    except AssertionError as error:
        print(f"WIRING SELF-CHECK: FAIL — {error}")
        print(
            f"positive={output['summary']['diagonal_E_route_match_rate']:.4f} "
            f"null={output['summary']['zero_e_route_match_rate']:.4f} "
            f"chance={output['chance_route_match']:.4f}"
        )
        return 1

    print("WIRING SELF-CHECK: PASS")
    print(
        f"n_pairs={output['n_pairs']} n_cells={output['n_cells']} "
        f"positive={output['summary']['diagonal_E_route_match_rate']:.4f} "
        f"null={output['summary']['zero_e_route_match_rate']:.4f} "
        f"chance={output['chance_route_match']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
