from _support import workspace, dump, run


def main():
    cases = (
        ('experiments/collect_0902/collect_overnight.py', 'results/night_0902/dstart_hard', 23),
        ('experiments/collect_0902/collect_emb.py', 'results/night_0902/emb_s16000', 23),
        ('experiments/collect_0903/collect_dialect.py', 'results/night_0903/dialect', 40),
        ('experiments/collect_0903/collect_n345.py', 'results/night_0903/fsqz_cont', 40),
        ('experiments/collect_0903/collect_overnight_alts.py', 'results/night_0902/dstart_soft', 23),
    )
    for script, directory, seed in cases:
        tmp, root = workspace()
        try:
            if 'collect_n345' in script:
                for d in ('fsqz_dq_cd3', 'fsqz_dq_cd5', 'dialect_s26', 'dialect_s27'):
                    dump(root, f'results/night_0903/{d}/rollout_one_s40.json', {'rates': {'subgoal': .5, 'R0': .5, 'null_u': .5, 'shuf': .5, 'bc': .5}})
            for name in ('alpha', 'beta'):
                dump(root, f'{directory}/rollout_{name}_s{seed}.json',
                     {'rates': {'subgoal': .5, 'R0': .5, 'null_u': .5, 'shuf': .5, 'bc': .5}})
            result = run(script, tmp)
            assert result.returncode != 0 and f'rollout_alpha_s{seed}.json' in result.stderr and f'rollout_beta_s{seed}.json' in result.stderr, script + result.stdout + result.stderr
        finally:
            tmp.cleanup()


if __name__ == '__main__': main()
