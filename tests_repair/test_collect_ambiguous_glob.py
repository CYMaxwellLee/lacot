from _support import workspace, dump, run


def main():
    tmp, root = workspace()
    try:
        for name in ('alpha', 'beta'):
            dump(root, f'results/night_0902/dz2rep_d2/rollout_{name}_s2.json',
                 {'rates': {'subgoal': .5, 'bc': .5, 'null_u': .5}})
        result = run('experiments/collect_0902/collect_0902.py', tmp, 'B')
        assert result.returncode != 0 and 'rollout_alpha_s2.json' in result.stderr and 'rollout_beta_s2.json' in result.stderr, result.stdout + result.stderr
    finally:
        tmp.cleanup()


if __name__ == '__main__': main()
