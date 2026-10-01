"""Smoke inventory and fail-closed entry point, not fabricated GPU coverage."""
import argparse
from pathlib import Path
from builder import build
from common import Blocked, verify_source, write_json
from harness_move3 import rollout_plan as move3_plan
from harness_move1 import rollout_plan as move1_plan


def inventory(builder):
    cases = [
        ('missing-row', 'test_17', 'CPU fixture passed; GPU artifact mutation pending'),
        ('duplicate-row', 'test_17', 'CPU fixture passed; GPU artifact mutation pending'),
        ('wrong-seed', 'test_17', 'CPU fixture passed; GPU artifact mutation pending'),
        ('wrong-reset-fingerprint', 'test_17', 'CPU fixture passed; GPU artifact mutation pending'),
        ('injection-overwritten-by-flow', 'test_19', 'CPU mutation rejected; GPU pending'),
        ('legal-waypoint-short-cap', 'test_09/test_23', 'real model/environment CPU passed; GPU pending'),
        ('reached-cap-stuck', 'test_09', 'CPU state core passed; explicit calibration required for GPU'),
        ('mid-chunk-switch-anomaly', 'test_09/test_18', 'CPU mutation rejected; GPU pending'),
        ('representation-pass-rollout-fail', 'test_36', 'full-artifact CPU E2 extraction and combine passed; GPU pending'),
        ('artificially-polluted-row', 'test_17', 'CPU fixture rejected; GPU pending'),
        ('d3-d5-tie', 'test_08', 'v5 CPU counterexample passed'),
        ('same-u-P-Q-reruns-each-geometry', 'test_16/test_23', 'real s33 CPU P/Q path passed; GPU pending'),
        ('move1-108-ceiling', 'test_24', '80 main + 8 N + 16 gate P/R + 4 conformance'),
        ('actual-steps-plus-one-trace', 'test_25', 'short trace accepted; mismatches/padding/splicing rejected'),
        ('d4-symmetric-exclusive-crossing', 'test_26', 'own >=1 and opposite=0; CPU positive/negative cases'),
        ('M9-sha-before-import', 'test_01/test_27', 'real M9 PIN and negative import guard'),
        ('frozen-oracle-A-bit-equivalence', 'test_23', 'real CPU 3 x 160 steps, actions and XY identical'),
        ('runtime-coordinator-mutation-regressions', 'test_37/test_38', 'stub fuzz, full coordination and source binding'),
        ('raw-shards-validation-error-resume', 'test_39/test_42', 'exclusive raw, compressed RNG, interrupted/resumed round trip'),
        ('hard-prerequisites-downgrade', 'test_31', 'named key qualification and conformance downgrade'),
        ('timing-ETA-vFinal', None, 'BLOCKED; no GPU timing; no invented ETA'),
    ]
    return dict(status='GPU_SMOKE_PENDING', production_evidence=False,
                move1_maximum=108,
                move1_scale=dict(main=80, gate_N=8, gate_PR=16, conformance=4),
                move1_full_plan=move1_plan(builder),
                calibration_required=['dataset_dir', 'steps_per_cell', 'stuck_window',
                                      'stuck_distance', 'noise_band.4', 'noise_band.5'],
                move3=move3_plan(builder, smoke=True), move1=move1_plan(builder, smoke=True),
                cases=[dict(name=n, selftest=t, status=s) for n, t, s in cases],
                determinism_repeats=[dict(task=t, arm=a, additional_draw=False,
                                         kind='determinism-verification') for t in (4, 5) for a in 'PQ'],
                run_note='gate P/R only after this question N succeeds; smoke has 2 gate questions '
                         'and cannot establish the >=5 behavioral eligibility gate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    args = p.parse_args()
    try:
        verify_source()
        # Separate processes: deterministic backends must be configured before
        # each process initializes CUDA. No automatic production promotion.
        import subprocess
        import sys
        from common import HERE
        for move in (3, 1):
            subprocess.run([sys.executable, '-B', str(HERE/f'harness_move{move}.py'), '--smoke',
                            '--outdir', str(args.outdir/f'move{move}'),
                            '--calibration', str(args.calibration)], check=True)
    except Blocked as e:
        p.exit(2, str(e)+'\n')


if __name__ == '__main__':
    main()
