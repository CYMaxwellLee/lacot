"""Shared checks for historical collection reports."""
import glob
import os
from collections import Counter


def unique_glob(pattern):
    paths = sorted(glob.glob(pattern))
    if len(paths) > 1:
        raise ValueError(f'Multiple files for {pattern}: {paths}')
    return paths[0] if paths else None


def same_run(rollout, diag):
    if rollout and diag:
        ro = os.path.basename(rollout).removeprefix('rollout_')
        dg = os.path.basename(diag).removeprefix('diag_')
        if ro != dg:
            raise ValueError(f'Run mismatch: {rollout} vs {diag}')


def check_seed(seen, seed, path):
    if seed in seen:
        raise ValueError(f'Duplicate seed {seed}: {seen[seed]} vs {path}')
    seen[seed] = path


def task_failures(eps):
    counts = Counter(e['task'] for e in eps)
    fails = Counter(e['task'] for e in eps if not e['success'])
    return [fails[t] if counts[t] else None for t in range(1, 6)]


def check_task_counts(eps, expected=50):
    counts = Counter(e['task'] for e in eps)
    if any(counts[t] != expected for t in range(1, 6)) or any(t not in range(1, 6) for t in counts):
        raise ValueError(f'Expected {expected} episodes per task (1-5); got {dict(sorted(counts.items()))}')
