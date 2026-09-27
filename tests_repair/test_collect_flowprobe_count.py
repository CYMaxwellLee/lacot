from _support import workspace, dump, run


def main():
    tmp, root = workspace()
    try:
        dump(root, 'results/night_0902/v8diag/diag_alpha_s23.json',
             [{'arm': '分段 conf2', 'task': 1, 'success': True}])
        result = run('experiments/collect_0902/collect_flowprobe.py', tmp)
        assert result.returncode != 0 and '50' in result.stderr and 'task' in result.stderr.lower(), result.stdout + result.stderr
    finally:
        tmp.cleanup()


if __name__ == '__main__': main()
