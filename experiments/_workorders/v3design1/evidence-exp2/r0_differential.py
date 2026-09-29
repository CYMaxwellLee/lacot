"""Compare a supplied pre-edit scratch snapshot to the current full CPU loop.

Usage: .venv/bin/python experiments/_workorders/v3design1/evidence-exp2/r0_differential.py /tmp/scratch-before.py
No environment imports, GPU, datasets or formal training; tiny CPU fixtures only.
"""
import hashlib
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import torch
from test_refine_wiring_v3 import scratch_fixture


def main():
    torch.set_num_threads(1)
    before = Path(sys.argv[1]).read_text()
    print('Pre-edit scratch SHA256:', hashlib.sha256(before.encode()).hexdigest())
    for fsq, intent, bc, div in itertools.product([False, True], [False, True], [False, True], [0., .07]):
        results = []
        for source in (before, None):
            f = scratch_fixture(learned=False, fsq=fsq, intent=intent, bc_indep=bc, div=div, source=source)
            f.ns['_stage2_loop'](1)
            params = [(p.detach().clone(), None if p.grad is None else p.grad.clone())
                      for m in f.ns['f_mods'] for p in m.parameters()]
            params.extend((p.detach().clone(), p.grad.clone()) for p in f.ns['bc_head'].parameters())
            results.append((f.backwards, params, torch.get_rng_state()))
        assert len(results[0][0]) == len(results[1][0]) > 0, 'backward loss lists must be equally nonempty'
        assert len(results[0][1]) == len(results[1][1]) > 0, 'parameter lists must be equally nonempty'
        for a, b in zip(results[0][0], results[1][0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for (a, ga), (b, gb) in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if ga is None:
                assert gb is None
            else:
                torch.testing.assert_close(ga, gb, rtol=0, atol=0)
        assert torch.equal(results[0][2], results[1][2])
        print(f'R0 fsq_z={fsq} intent={intent} bc_indep={bc} div={div}: loss/grad/post-step-weights/RNG BITWISE PASS')
    print('16/16 pre-edit versus new production _stage2_loop CPU cases PASS')


if __name__ == '__main__':
    main()
