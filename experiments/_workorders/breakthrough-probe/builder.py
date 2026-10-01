"""Deterministic shortest-path DAG builder. No environment/model import."""
import argparse
import ast
from collections import Counter, deque
import json
from pathlib import Path
import numpy as np
from common import GATE, ENV, Blocked, canonical, file_sha, sha, write_json

MAZE_SOURCE = Path('/home/cymaxwelllee/Projects/ogbench/ogbench/locomaze/maze.py')


def read_maze(path, expected):
    """Read literals only; select by gate0b's ndarray byte fingerprint."""
    tree = ast.parse(Path(path).read_text())
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == 'maze_map' for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        array = np.asarray(value, dtype='<i8')
        if sha(array.tobytes()) == expected:
            matches.append(array.tolist())
    if len(matches) != 1:
        raise Blocked('BLOCKED: large maze literal does not uniquely match gate0b hash')
    return matches[0]


def neighbors(c):
    i, j = c
    return sorted(((i-1, j), (i+1, j), (i, j-1), (i, j+1)))


def distances(free, start):
    found, queue = {start: 0}, deque([start])
    while queue:
        c = queue.popleft()
        for nxt in neighbors(c):
            if nxt in free and nxt not in found:
                found[nxt] = found[c] + 1
                queue.append(nxt)
    return found


def geometry(maze, s, g):
    free = {(i, j) for i, row in enumerate(maze) for j, v in enumerate(row) if v == 0}
    s, g = tuple(s), tuple(g)
    ds, dg = distances(free, s), distances(free, g)
    if g not in ds:
        raise ValueError('disconnected geometry')
    D = ds[g]
    dag = {c: [v for v in neighbors(c) if v in ds and v in dg
               and ds[v] == ds[c]+1 and ds[v]+dg[v] == D]
           for c in ds if c in dg and ds[c]+dg[c] == D}

    def walk(c):
        if c == g:
            return [[g]]
        return [[c] + tail for nxt in dag[c] for tail in walk(nxt)]

    paths = sorted(walk(s))
    if len(paths) not in (1, 2):
        raise ValueError('contract requires exactly one or two routes')
    result = dict(s=list(s), g=list(g), D=D, n_routes=len(paths),
                  routes={name: [list(c) for c in route]
                          for name, route in zip(('A', 'B'), paths)},
                  dag=[dict(cell=list(c), depth=ds[c], next=[list(v) for v in dag[c]])
                       for c in sorted(dag, key=lambda c: (ds[c], c))],
                  demo_label='unverified',
                  demo_evidence='Fine occupancy oracle and generation tie-break not verified.')
    if len(paths) == 1:
        depth = D // 2
        result.update(w_depth=depth, w={'A': list(paths[0][depth])},
                      exclusive={'A': [], 'B': []}, branch=None, merge=None)
    else:
        a, b = paths
        different = [i for i in range(D+1) if a[i] != b[i]]
        lo, hi = min(different), max(different)
        if different != list(range(lo, hi+1)):
            raise ValueError('multiple exclusive spans; no contract for this geometry')
        depth = (lo + hi) // 2
        result.update(w_depth=depth, w={'A': list(a[depth]), 'B': list(b[depth])},
                      exclusive={'A': [list(c) for c in a[lo:hi+1]],
                                 'B': [list(c) for c in b[lo:hi+1]]},
                      exclusive_depths=[lo, hi], branch=list(a[lo-1]), merge=list(a[hi+1]))
    return result


def build(gate_path=GATE, maze_path=MAZE_SOURCE):
    gate = json.loads(Path(gate_path).read_text())
    if gate['env'] != ENV or gate['fingerprint_mismatches']:
        raise ValueError('gate0b environment/fingerprint failure')
    rows = gate['rows']
    if len(rows) != 50 or len({(r['task'], r['episode']) for r in rows}) != 50:
        raise ValueError('gate0b missing/duplicate rows')
    for r in rows:
        if not r['fingerprint_match'] or r['env_seed'] != 1000*r['task']+r['episode']:
            raise ValueError('unverified gate0b row')
        if r['double_fail'] != (r['a_bits'] == r['b_bits'] == 0):
            raise ValueError('inconsistent failure bits')
    questions = {name: sorted([dict(task=r['task'], episode=r['episode']) for r in rows if pred(r)],
                             key=lambda r: (r['task'], r['episode']))
                 for name, pred in (
                     ('main', lambda r: r['double_fail'] and r['n_shortest'] == 2),
                     ('control', lambda r: r['double_fail'] and r['n_shortest'] == 1),
                     ('gate', lambda r: r['a_bits'] == r['b_bits'] == 64))}
    for name, expected in [('main', {4: 14, 5: 6}), ('control', {2: 10}), ('gate', {1: 7, 3: 1})]:
        if Counter(q['task'] for q in questions[name]) != expected:
            raise ValueError(f'question contract mismatch: {name}')
    maze = read_maze(maze_path, gate['maze_sha256'])
    geometries = {}
    for task in range(1, 6):
        task_rows = [r for r in rows if r['task'] == task]
        r = task_rows[0]
        geom = geometry(maze, r['s'], r['g'])
        for other in task_rows:
            if (other['s'], other['g'], other['shortest_len'], other['n_shortest']) != (
                    geom['s'], geom['g'], geom['D'], geom['n_routes']):
                raise ValueError('geometry disagrees with gate0b')
        geometries[str(task)] = geom
    for task, depth, points in [('4', 4, [[1, 6], [5, 10]]), ('5', 2, [[1, 3], [3, 1]])]:
        if geometries[task]['w_depth'] != depth or sorted(geometries[task]['w'].values()) != points:
            raise ValueError('derived waypoint disagrees with card')
    if geometries['2']['w_depth'] != 9:
        raise ValueError('task 2 midpoint disagrees with card')
    donor = {}
    for task, other, group in [(4, 5, 'main'), (5, 4, 'main'), (1, 3, 'gate'), (3, 1, 'gate')]:
        ep = min(q['episode'] for q in questions[group] if q['task'] == other)
        donor[str(task)] = dict(task=other, episode=ep, route='A')
    return dict(schema=1, env=ENV, gate_sha256=file_sha(gate_path),
                maze_sha256=gate['maze_sha256'], maze=maze,
                maze_source_sha256=file_sha(maze_path), questions=questions,
                geometries=geometries, R_donor=donor, historical_receipt=gate['receipt'],
                blockers=[])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate', type=Path, default=GATE)
    p.add_argument('--maze-source', type=Path, default=MAZE_SOURCE)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = build(args.gate, args.maze_source)
    print(write_json(args.out, result))


if __name__ == '__main__':
    main()
