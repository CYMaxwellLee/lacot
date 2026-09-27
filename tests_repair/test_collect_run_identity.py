from _support import ROOT, workspace, dump, run


def main():
    cases = (
        ('collect_0902.py', 'dz2rep_d2', 2, ('B',)),
        ('collect_d.py', 'varsrc2_D1x', 2, ()),
        ('collect_ebfs_c.py', 'ebfs', 23, ()),
        ('collect_esel.py', 'esel16', 23, ()),
        ('collect_guard.py', 'v8diag', 23, ()),
        ('collect_snap.py', 'v8diag', 23, ()),
        ('collect_uora.py', 'uora', 23, ()),
    )
    for script, directory, seed, args in cases:
        tmp, root = workspace()
        try:
            if script == 'collect_esel.py': (root / 'experiments').symlink_to(ROOT / 'experiments')
            base = f'results/night_0902/{directory}'
            dump(root, f'{base}/rollout_alpha_s{seed}.json',
                 {'rates': {'subgoal': .5, 'bc': .5, 'null_u': .5}})
            dump(root, f'{base}/diag_beta_s{seed}.json',
                 [{'arm': '分段 conf2', 'task': 1, 'success': True}])
            result = run(f'experiments/collect_0902/{script}', tmp, *args)
            assert result.returncode != 0 and f'rollout_alpha_s{seed}.json' in result.stderr and f'diag_beta_s{seed}.json' in result.stderr, script + result.stdout + result.stderr
        finally:
            tmp.cleanup()


if __name__ == '__main__': main()
