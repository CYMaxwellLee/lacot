from _support import workspace, dump, run


def main():
    cases = (
        ('experiments/collect_0902/collect_guard.py', 'v8diag'),
        ('experiments/collect_0902/collect_snap.py', 'v8diag'),
        ('experiments/collect_0902/collect_dstart.py', 'dstart_hard'),
    )
    for script, directory in cases:
        tmp, root = workspace()
        try:
            base = f'results/night_0902/{directory}'
            dump(root, f'{base}/rollout_alpha_s23.json', {'rates': {'subgoal': .5, 'bc': .5}})
            dump(root, f'{base}/diag_alpha_s23.json', [{'arm': '分段 conf2', 'task': 1, 'success': True}])
            result = run(script, tmp)
            assert result.returncode == 0 and 'None' in result.stdout, script + result.stdout + result.stderr
        finally:
            tmp.cleanup()


if __name__ == '__main__': main()
