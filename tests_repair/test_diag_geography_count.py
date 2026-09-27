from _support import workspace, dump, run


def main():
    tmp, root = workspace()
    try:
        dump(root, 'results/night_0902/v8diag/diag_alpha_s23.json',
             [{'arm': arm, 'task': 1, 'success': True} for arm in ('分段 conf2', '誠實 BC')])
        result = run('experiments/diag_geography_0902.py', tmp, 'A')
        assert result.returncode != 0 and '50' in result.stderr and 'task' in result.stderr.lower(), result.stdout + result.stderr
    finally:
        tmp.cleanup()


if __name__ == '__main__': main()
