#!/usr/bin/env python3
"""selfsub-v1: CPU-only frozen-model endpoint probe; never calls env.step."""
import os
import sys

# These assignments precede every probe, torch, and ogbench import.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['MUJOCO_GL'] = 'egl'
sys.dont_write_bytecode = True

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np

ROOT = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/selfsub')
OUT = None
OUTROOT = None
CKPT_PATH = None
CKPT_SHA256 = None
PROBE = Path('/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe')
TRAP = ROOT.parent / 'detour-u/trapset-v2.json'
TASKS = (2, 4, 5)
CATS = ('VALID', 'WALL', 'FAR', 'NEAR', 'OFF')
COLORS = dict(VALID='#179447', WALL='#dc2626', FAR='#a855f7', NEAR='#e99b18', OFF='#2563eb')
INTRO = '''證明：在陷阱題的關鍵格，**flow 對（當下格、最終終點）抽的 u，解碼成想像軌跡後，往前約 3 格的那個點能不能直接當近子目標**。
判準＝那個點落在「**任何一條**最短路上、從當下格前進 2 到 4 步」（VALID）的比例；尺要先過三道閘（見 §2），閘不過就不出結論。'''


class StopProbe(RuntimeError):
    pass


def save(name, obj):
    path = OUT if name in ('selfsub-result.json', 'selftest-result.json') else OUTROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def emit(message):
    print(message, flush=True)


class Graph:
    """Route-independent scoring. All distances are recomputed with BFS."""
    def __init__(self, maze, cell_to_xy, xy_to_cell):
        self.maze = np.asarray(maze)
        self.free = {tuple(map(int, c)) for c in np.argwhere(self.maze == 0)}
        self.cell_to_xy = cell_to_xy
        self.xy_to_cell = xy_to_cell
        self.dist = {c: self.bfs(c) for c in sorted(self.free)}

    def neighbors(self, c):
        i, j = c
        return sorted(n for n in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)) if n in self.free)

    def bfs(self, start):
        out, queue = {start: 0}, deque([start])
        while queue:
            c = queue.popleft()
            for n in self.neighbors(c):
                if n not in out:
                    out[n] = out[c] + 1
                    queue.append(n)
        return out

    def routes(self, s, g):
        result = []
        def visit(path):
            c = path[-1]
            if c == g:
                result.append(path)
            else:
                for n in self.neighbors(c):
                    if self.dist[g].get(n) == self.dist[g][c] - 1:
                        visit(path + [n])
        visit([s])
        return result

    def away(self, c, g):
        before = np.linalg.norm(self.cell_to_xy(c) - self.cell_to_xy(g))
        nxt = [n for n in self.neighbors(c) if self.dist[g][n] == self.dist[g][c] - 1]
        if not nxt:
            raise StopProbe(f'No shortest next step at {c} to {g}')
        return all(np.linalg.norm(self.cell_to_xy(n) - self.cell_to_xy(g)) >= before for n in nxt)

    def score(self, c, g, point, short=False):
        cp = tuple(int(x) for x in self.xy_to_cell(np.asarray(point)))
        row = dict(c_p=list(cp), p_xy=np.asarray(point, dtype=float).tolist(), short=bool(short),
                   cat=None, prog=None, detour=None)
        if cp not in self.free:
            row['cat'] = 'WALL'
            return row
        if cp not in self.dist[c] or cp not in self.dist[g]:
            raise StopProbe(f'Unreachable free landing cell {cp}; no finite BFS score')
        prog = self.dist[g][c] - self.dist[g][cp]
        detour = self.dist[c][cp] + self.dist[g][cp] - self.dist[g][c]
        if detour < 0:
            raise StopProbe('BFS triangle inequality violated')
        cat = 'OFF' if detour > 0 else 'VALID' if 2 <= prog <= 4 else 'FAR' if prog >= 5 else 'NEAR'
        row.update(cat=cat, prog=int(prog), detour=int(detour))
        return row


def smooth(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (128, 2) or not np.isfinite(raw).all():
        raise StopProbe(f'Decoder API requires finite [128,2], got {raw.shape}')
    return np.stack([np.convolve(raw[:, k], np.ones(9)/9, mode='valid') for k in range(2)], axis=1)


def arcpoint(points, length):
    points = np.asarray(points, dtype=np.float64)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    total = float(arc[-1])
    if total < length:
        return points[-1].copy(), True, total
    k = int(np.searchsorted(arc, length, side='left'))
    if k == 0:
        return points[0].copy(), False, total
    span = arc[k] - arc[k-1]
    fraction = 0. if span == 0 else (length - arc[k-1]) / span
    return points[k-1] + fraction*(points[k]-points[k-1]), False, total


def measure(graph, c, g, raw, length=12., anchor=True, smoothing=True):
    points = smooth(raw) if smoothing else np.asarray(raw, dtype=np.float64).copy()
    if anchor:
        points += graph.cell_to_xy(c) - points[0]
    p, short, total = arcpoint(points, length)
    return dict(graph.score(c, g, p, short), total_arc=total), points


def greedy(graph, c, g, stop_wall=True):
    start, goal = graph.cell_to_xy(c), graph.cell_to_xy(g)
    direction = (goal-start)/np.linalg.norm(goal-start)
    p, traveled = start.copy(), 0.
    for k in range(1, 49):
        candidate = start + (k*.25)*direction
        cp = tuple(int(v) for v in graph.xy_to_cell(candidate))
        if stop_wall and cp not in graph.free:
            break
        p, traveled = candidate, k*.25
    return dict(graph.score(c, g, p, traveled < 12.), traveled=traveled,
                stop_wall=stop_wall)


def counts(records):
    tally = Counter(r['cat'] for r in records)
    n = len(records)
    return dict(n=n, counts={k: tally[k] for k in CATS},
                valid_rate=tally['VALID']/n if n else None)


def make_inventory(graph, trap):
    inv = {}
    for task in TASKS:
        spec = trap['tasks'][str(task)]
        s, g = tuple(spec['s']), tuple(spec['g'])
        paths = graph.routes(s, g)
        cells = sorted({c for path in paths for c in path if graph.dist[g][c] >= 4})
        inv[task] = dict(s=s, g=g, paths=paths, cells=cells)
    return inv


def score_mutant(record, variant):
    """Test-only bound mutations; production Graph.score is unchanged."""
    out = dict(record)
    if out['detour'] == 0:
        prog = out['prog']
        if (variant == 'no_lower_bound' and 0 <= prog <= 4 or
                variant == 'no_upper_bound' and prog >= 2):
            out['cat'] = 'VALID'
    return out


def negative_summary(graph, inventory, stop_wall, variant=None):
    out, rows = {}, []
    for task, spec in inventory.items():
        away_records = []
        for c in spec['cells']:
            r = greedy(graph, c, spec['g'], stop_wall)
            if variant is not None:
                r = score_mutant(r, variant)
            if graph.away(c, spec['g']):
                away_records.append(r)
                rows.append(dict(task=task, cell=list(c), **r))
        out[str(task)] = dict(counts(away_records),
                             status='PASS' if counts(away_records)['valid_rate'] <= .10 else 'FAIL')
    return out, rows


def single_route_mutant(graph, inventory, trap):
    records = []
    for task, spec in inventory.items():
        only = {tuple(c) for c in trap['tasks'][str(task)]['route_from_s']}
        for route_id, path in enumerate(spec['paths']):
            for c in spec['cells']:
                if c not in path:
                    continue
                cp = path[path.index(c)+3]
                score = graph.score(c, spec['g'], graph.cell_to_xy(cp))
                if cp not in only:
                    score['cat'] = 'OFF'
                records.append(dict(task=task, cell=list(c), route_id=route_id, **score))
    summary = counts(records)
    return dict(summary, status='PASS' if summary['valid_rate'] == 1. else 'FAIL',
                rejected=[r for r in records if r['cat'] != 'VALID'])


def selftest():
    """Synthetic geometry/jitter tests plus frozen map fixture for mutation audit."""
    maze = np.ones((7, 10), dtype=int)
    maze[1:6, 1:9] = 0
    # Fake coordinates are test-only, not used in the real environment.
    xy = lambda c: np.array([4.*c[1], 4.*c[0]])
    ij = lambda p: (int(np.floor(p[1]/4+.5)), int(np.floor(p[0]/4+.5)))
    graph = Graph(maze, xy, ij)
    c, g = (3, 1), (3, 8)
    for cp, expected in [((3, 4), 'VALID'), ((3, 6), 'FAR'), ((3, 2), 'NEAR'),
                         ((3, 1), 'NEAR'), ((2, 4), 'OFF'), ((0, 4), 'WALL'),
                         ((-1, 4), 'WALL')]:
        assert graph.score(c, g, xy(cp))['cat'] == expected, (cp, expected)
    raw = np.column_stack([np.linspace(4., 16., 128), np.full(128, 12.)])
    raw[:, 1] += .14 * (-1.)**np.arange(128)
    positive, _ = measure(graph, c, g, raw)
    mutant, _ = measure(graph, c, g, raw, smoothing=False)
    assert positive['cat'] == 'VALID'
    assert mutant['cat'] != 'VALID'
    short, _ = measure(graph, c, g, np.repeat(xy(c)[None], 128, axis=0))
    assert short['short'] and short['cat'] == 'NEAR'
    p, shortflag, total = arcpoint([[0., 0.], [0., 0.], [8., 0.], [8., 8.]], 12.)
    assert np.allclose(p, [8., 4.]) and not shortflag and total == 16.
    shifted, points = measure(graph, c, g, raw+37.)
    assert shifted['c_p'] == positive['c_p'] and np.allclose(points[0], xy(c))
    # A real-map fixture tests whether the requested G3 mutation is killable;
    # it never substitutes for the real env or real decoder gates.
    fixture = json.loads((PROBE/'builder.json').read_text())['maze']
    fixture_graph = Graph(fixture, xy, ij)
    trap = json.loads(TRAP.read_text())
    inventory = make_inventory(fixture_graph, trap)
    g1 = single_route_mutant(fixture_graph, inventory, trap)
    assert g1['status'] == 'FAIL'
    g3, g3rows = negative_summary(fixture_graph, inventory, True, 'no_lower_bound')
    g3_killed = any(v['status'] == 'FAIL' for v in g3.values())
    assert g3_killed, 'G3 lower-bound mutant must fail at least one task'
    through_wall, through_wall_rows = negative_summary(fixture_graph, inventory, False)
    assert all(v['status'] == 'PASS' for v in through_wall.values())
    upper = fixture_graph.score((3, 8), inventory[4]['g'], xy((5, 6)))
    upper_mutant = score_mutant(upper, 'no_upper_bound')
    assert upper['cat'] == 'FAR' and upper['prog'] == 8 and upper['detour'] == 0
    assert upper_mutant['cat'] == 'VALID', 'Upper-bound mutant must fail FAR expectation'
    lines = [f"SELFTEST scoring/arc/anchor/short: PASS",
             f"SELFTEST KILLER G1 single-trapset-route: {g1['status']} VALID={g1['counts']['VALID']}/{g1['n']}",
             f"SELFTEST KILLER G2 no-smoothing: FAIL cat={mutant['cat']} (synthetic jitter; proper={positive['cat']})",
             f"SELFTEST KILLER G3 no-lower-bound (0 <= prog <= 4): {json.dumps(g3, ensure_ascii=False)}",
             f"SELFTEST required G3 FAIL observed: {g3_killed}",
             'SELFTEST DESCRIPTION straight-through-wall: 預期仍 PASS，用來說明穿牆會被判 FAR／WALL；不是殺手。 ' + json.dumps(through_wall, ensure_ascii=False),
             'SELFTEST upper-bound task=4 c=(3,8) c_p=(5,6) expected=FAR: PASS ' + json.dumps(upper),
             'SELFTEST upper-bound MUTANT prog >= 2 expected=FAR: FAIL ' + json.dumps(upper_mutant)]
    result = dict(unit_tests='PASS', requested_killers_all_observed=g3_killed,
                  G1=g1, G2=dict(synthetic=True, proper=positive, mutant=mutant, status='FAIL'),
                  G3=dict(synthetic_coordinates=True, by_task=g3, away_records=g3rows,
                          variant='no_lower_bound', requested_FAIL_observed=g3_killed),
                  through_wall_description=dict(expected='PASS', by_task=through_wall,
                                                away_records=through_wall_rows),
                  upper_bound=dict(task=4, cell=[3,8], expected='FAR', proper=upper,
                                   proper_status='PASS', mutant=upper_mutant, mutant_status='FAIL'),
                  stdout=lines)
    save('selftest-result.json', result)
    for line in lines:
        emit(line)
    return 0 if g3_killed else 1


def render(graph, spec, row, path, planted=False):
    # Keep matplotlib's writable cache in the workorder directory too.
    os.environ['MPLCONFIGDIR'] = str(OUTROOT/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
    fig, ax = plt.subplots(figsize=(11, 8))
    for i in range(graph.maze.shape[0]):
        for j in range(graph.maze.shape[1]):
            x, y = graph.cell_to_xy((i, j))
            ax.add_patch(Rectangle((x-2, y-2), 4, 4, facecolor='#aaa' if graph.maze[i,j] else 'white',
                                   edgecolor='#ddd', linewidth=.5))
    for k, route in enumerate(spec['paths']):
        route_xy = np.array([graph.cell_to_xy(c) for c in route])
        ax.plot(*route_xy.T, color=['#06b6d4', '#f472b6'][k % 2], lw=3., alpha=.65,
                label=f'Shortest route {k}')
    for draw in row['flow']:
        points = np.array(draw['smooth_anchored_xy'])
        ax.plot(*points.T, color='#444', lw=.5, alpha=.3)
    for k, draw in enumerate(row['flow']):
        plotted = row['references']['positive'][k % len(row['references']['positive'])] if planted else draw
        ax.scatter(*plotted['p_xy'], color=COLORS[plotted['cat']], s=40, zorder=6)
        ax.annotate(str(k), plotted['p_xy'], fontsize=7)
    for r in row['references']['decoded']:
        ax.scatter(*r['p_xy'], marker='*', s=180, facecolor='black', edgecolor='white', zorder=7)
    ax.scatter(*row['references']['greedy']['p_xy'], marker='X', s=120, color='#111', zorder=8)
    ax.scatter(*graph.cell_to_xy(tuple(row['cell'])), marker='o', s=100, facecolor='none', edgecolor='black')
    ax.scatter(*graph.cell_to_xy(spec['g']), marker='D', s=80, color='black')
    for cat in CATS:
        ax.scatter([], [], color=COLORS[cat], s=40, label=cat)
    ax.scatter([], [], marker='*', s=120, color='black', label='Decoded reference (+prime)')
    ax.scatter([], [], marker='X', s=90, color='black', label='Greedy (-)')
    # Identical title and layout are required for the planted figure.
    ax.set_title(f"Task {row['task']} cell {tuple(row['cell'])}: flow endpoints at arc 12")
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), fontsize=8)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout()
    fig.savefig(OUTROOT/path, dpi=160)
    plt.close(fig)


def finish_report(result):
    result['n_cells'] = len(result['cells'])
    result['n_draws'] = sum(len(r['flow']) for r in result['cells'])
    save('selfsub-result.json', result)
    gate_lines = []
    for name in ('G1', 'G2', 'G3'):
        gate = result['gates'].get(name)
        gate_lines.append(f"- {name}: `{json.dumps(gate, ensure_ascii=False)}`" if gate else f'- {name}: 未執行。')
    summary_lines = []
    for task, summary in result['summary'].items():
        if summary.get('r') is not None:
            away = summary['away']
            summary_lines.append(f"- task {task}: 背離格 VALID={away['counts']['VALID']}/{away['n']}，r={summary['r']:.6f}；{summary['interpretation']}。")
    killer_lines = [f"- {name}: `{json.dumps(killer.get('by_task', {k: killer[k] for k in ('status', 'n', 'counts') if k in killer}), ensure_ascii=False)}`"
                    for name, killer in result['killers'].items()]
    missing = result.get('not_produced', [])
    text = INTRO + '\n\n' + '''本探針只使用 CPU、環境幾何與凍結模型的 encode/sample/decode；不 reset、不 step、不跑 rollout。BFS 全部路線獨立重算；ref-routes-density.json 僅用於核對，不作打分答案。所有門檻及尺固定為工單原值。

執行指令（工作目錄為本目錄）：

```bash
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl /home/cymaxwelllee/Projects/lacot/.venv/bin/python -B selfsub_probe.py --selftest > selftest-output.txt 2>&1
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl /home/cymaxwelllee/Projects/lacot/.venv/bin/python -B selfsub_probe.py > full-output.txt 2>&1
```

依 hsweep-eval-v1-r1，快取鍵為 checkpoint SHA256 與本離線尺檔 SHA256。命中後逐欄核對 provenance、牆圖、路線、尺及門檻；未命中則現場重算三道閘、三個殺手與 50 格參照。新證據寫在 gates-evidence.json；閘失敗或殺手未被擋下就停止，不調尺重跑。seed = task × 10000 + i × 100 + j；每格抽樣前 torch.manual_seed(seed)，一次 sample_plan(16, cond.expand(16,-1), None)，每份單獨 decode。

輸出：selfsub_probe.py、selftest-result.json、selftest-output.txt、selfsub-result.json、full-output.txt、本 README.md。成功時另含 fig_A_task4_start.png、fig_B_task5_start.png、fig_C_task2_key.png、fig_P_planted.png 與 _lead_only/KEY-planted.json。BLOCKED 時不繪製不存在的 flow，未交項目及原因如下。
''' + '\n' + '\n'.join(gate_lines) + '\n\n殺手結果（FAIL 為預期）：\n\n' + '\n'.join(killer_lines) + '\n\n' + '\n'.join(summary_lines) + '\n\n' + f"n_cells={result['n_cells']} n_draws={result['n_draws']}\n\n狀態：{result['status']}；原因：{result.get('blocked_reason', '無')}。\n\n" + '\n'.join('- '+m for m in missing) + '\n\n' + '''測試環境差異聲明：selftest 的格心/格映射為明示的假座標，jitter 為手造；不會載入真模型。故它蓋不到環境 xy_to_ij 的實際邊界取整、凍結 checkpoint/EMA/正規化是否載對、真 decoder 的方向偏移/逐點抖動/變長與非有限值、flow 批次隨機抽樣、CPU 數值與真模型的短弧長。無參數全量使用指定 API，G2 的真模型證據依 checkpoint/ruler 快取鍵決定重用或重算；selftest G2 FAIL 是單元測試證據，真模型 G2 殺手證據另列於 selfsub-result.json。G3 與上界單元測試的假座標 fixture 是既有牆圖的平移；G3 另對上一輪真環境貪心點施加下界變體，未將假資料冒充真模型結果。

回鍋單只更換測試變體，正式 Graph.score 仍用 2 ≤ prog ≤ 4。G3 殺手將下界改為 0，停在原格的 NEAR/STAY 成為 VALID，至少一個 task 的 G3 必須 FAIL。舊穿牆變體僅留作描述，預期仍 PASS，用來說明穿牆會被判 FAR／WALL（個別格亦可能 OFF）。上界單元測試不是第四道閘：task 4、c=(3,8)、c_p=(5,6)，正式打分 FAR、prog=8；拿掉上界後 VALID，違反 FAR 預期而 FAIL。兩態與三個殺手原文見 selftest-output.txt；真環境 G3 原文見 full-output.txt。

執行差異：本次繪圖時 Matplotlib 自動將快取寫入 /tmp/matplotlib-vognu8f4（原文見 full-output.txt），偏離只寫戰場目錄的要求。已在 render 設定 MPLCONFIGDIR 指向本目錄的 matplotlib-cache，後續執行會使用此處；未重跑量測或刪除檔案。四張圖與抽樣結果已完成。

輸出契約：

| 成功路徑必須保留的輸出 | 允許變更的行為 |
| --- | --- |
| 每 task/cell 的 away、d_to_g、seed；每份 draw 的 cat、prog、detour、c_p、short 及 p_xy；三種參照每路線的原始分類；三道閘數值、分母、PASS/FAIL；弧長 8/16、不錨定分類、τ̂ 首點距離及起點路線歸屬 | 圖面大小、圖例位置、進度訊息與 JSON 縮排；不得改門檻、平滑、錨定、弧長、格集合或打分；遇停止條件允許提早結束，但未量測值必須明示 null 及原因 |

G2 比例按 (task, cell, 經過該格的完整最短路) 計算，共用尾段也各保留一筆；不偷偷去重。WALL 的 prog/detour 為 null，因牆格無 BFS 距離；若空格無有限距離則直接 BLOCKED。primary 每 task 各自算背離格的 16 份 VALID 比例，無跨 task 合併；未通過閘/殺手驗證不作可用性結論。short 表示取末點，不自動改類別。起點歸屬記錄所有包含 c_p 的路線 ID，合流格可同時屬於兩條路；只是描述。

一手 API 證據：

- /home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/runtime.py:33 指定 load_frozen；:123 condition 公式。
- /home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harness_detour.py:269 指定 decode；:476 rollout 使用 sample_plan(1,cond,None)。
- /home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harness_move1.py:114 指定 encode_trajectory，128 點弧長重採樣。
- 閘與殺手原文：selftest-output.txt、full-output.txt；逐筆原始分類：selfsub-result.json。
'''
    (OUTROOT/'README.md').write_text(text)


# The legacy cache has no explicit source key. Admit only the pinned evidence
# and pinned ruler, after checking every geometry/scoring function is unchanged.
LEGACY_RULER_SHA256 = 'e5b8a49b40805039a547f95b4731728884dd8b1bf13d4b324b1f61f1de2b93f6'
LEGACY_EVIDENCE_SHA256 = '17cfd8972f7a5a0a1d272ee38b8e90760a2976d28f657b2ad8842ef0dc2ae6f3'
GATE_EVIDENCE = None


def ruler_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def gate_cache_key(provenance):
    return (provenance['ckpt_sha256'], ruler_sha256())


def read_gate_evidence(path):
    import ast
    payload = path.read_bytes()
    evidence = json.loads(payload)
    if 'cache_key' in evidence:
        return evidence
    legacy = ROOT/'selfsub_probe.py'
    if (hashlib.sha256(payload).hexdigest() != LEGACY_EVIDENCE_SHA256 or
            hashlib.sha256(legacy.read_bytes()).hexdigest() != LEGACY_RULER_SHA256):
        raise StopProbe('Unkeyed gate evidence is not the pinned legacy evidence')
    def definitions(source):
        names = {'Graph', 'smooth', 'arcpoint', 'measure', 'greedy', 'counts',
                 'make_inventory', 'score_mutant', 'negative_summary', 'single_route_mutant'}
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(source).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
    if definitions(legacy.read_text()) != definitions(Path(__file__).read_text()):
        raise StopProbe('Legacy gate ruler functions differ from this ruler')
    evidence['cache_key'] = list(gate_cache_key(evidence['provenance']))
    return evidence


def validate_gate_evidence(evidence, result, inventory):
    for key in ('source_sha256', 'ckpt_sha256', 'dataset_sha256',
                'normalization_mu', 'normalization_sd', 'config'):
        if evidence['provenance'][key] != result['provenance'][key]:
            raise StopProbe(f'Cached gate provenance mismatch: {key}')
    if evidence['maze'] != result['maze'] or evidence['routes'] != result['routes']:
        raise StopProbe('Cached gate maze/routes mismatch')
    if evidence['ruler'] != result['ruler'] or evidence['thresholds'] != result['thresholds']:
        raise StopProbe('Cached gate ruler/threshold mismatch')
    expected_cells = {(task, c) for task, spec in inventory.items() for c in spec['cells']}
    cached_cells = {(r['task'], tuple(r['cell'])) for r in evidence['cells']}
    if expected_cells != cached_cells or len(evidence['cells']) != 50:
        raise StopProbe('Cached 50-cell inventory mismatch')


def resolve_gate_evidence(evidence, result, inventory, recompute):
    """A key miss recomputes; provenance checks apply only to a matching key."""
    key = gate_cache_key(result['provenance'])
    if evidence is not None and tuple(evidence['cache_key']) == key:
        validate_gate_evidence(evidence, result, inventory)
        return evidence, True
    emit('GATE CACHE MISS: recomputing G1/G2/G3 and all three killers; key=' + json.dumps(key))
    fresh = recompute()
    fresh['cache_key'] = list(key)
    validate_gate_evidence(fresh, result, inventory)
    return fresh, False


def recompute_gate_evidence(module, graph, inventory, trap, result):
    """Use the original reference definitions, ruler, thresholds and PB APIs."""
    import harness_detour as d
    import harness_move1 as m1
    cells, positives, decoded_records, no_smoothing = [], [], [], []
    with module.torch.no_grad():
        for task, spec in inventory.items():
            for c in spec['cells']:
                refs = dict(positive=[], decoded=[], greedy=greedy(graph, c, spec['g']),
                            greedy_through_wall_description=greedy(graph, c, spec['g'], False))
                for route_id, path in enumerate(spec['paths']):
                    if c not in path:
                        continue
                    index = path.index(c)
                    points = np.array([graph.cell_to_xy(cp) for cp in path[index:index+4]])
                    positive = dict(graph.score(c, spec['g'], points[-1]), route_id=route_id)
                    u = m1.encode_trajectory(module, points)
                    raw = d.decode(module, u, graph.cell_to_xy(c))
                    scored, _ = measure(graph, c, spec['g'], raw)
                    mutant, _ = measure(graph, c, spec['g'], raw, smoothing=False)
                    decoded = dict(scored, route_id=route_id, points_xy=points.tolist(),
                        raw_arc=float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()),
                        raw_start_distance=float(np.linalg.norm(raw[0]-graph.cell_to_xy(c))),
                        no_smoothing_killer=mutant)
                    refs['positive'].append(positive)
                    refs['decoded'].append(decoded)
                    positives.append(positive)
                    decoded_records.append(decoded)
                    no_smoothing.append(mutant)
                refs['greedy_no_lower_bound_killer'] = score_mutant(refs['greedy'], 'no_lower_bound')
                cells.append(dict(task=task, cell=list(c), away=graph.away(c, spec['g']),
                    d_to_g=graph.dist[spec['g']][c], seed=task*10000+c[0]*100+c[1],
                    flow=[], flow_state='not_measured: gates pending', references=refs))
    g3_by_task, g3_rows = negative_summary(graph, inventory, True)
    g3 = counts(g3_rows)
    gates = dict(G1=dict(counts(positives), status='PASS' if counts(positives)['valid_rate'] == 1. else 'FAIL', threshold=1.),
        G2=dict(counts(decoded_records), status='PASS' if counts(decoded_records)['valid_rate'] >= .90 else 'FAIL',
                threshold=.90, denominator='cell-route references; shared tails retained'),
        G3=dict(g3, threshold=.10, by_task=g3_by_task,
                status='PASS' if all(v['status'] == 'PASS' for v in g3_by_task.values()) else 'FAIL'))
    killer3_by_task, killer3_rows = negative_summary(graph, inventory, True, 'no_lower_bound')
    killed3 = any(v['status'] == 'FAIL' for v in killer3_by_task.values())
    killers = dict(G1=single_route_mutant(graph, inventory, trap),
        G2=dict(counts(no_smoothing), status='PASS' if counts(no_smoothing)['valid_rate'] >= .90 else 'FAIL',
                source='real frozen encoder and decoder; no smoothing, anchored'),
        G3=dict(variant='no_lower_bound: 0 <= prog <= 4', by_task=killer3_by_task,
                away_records=killer3_rows, required_at_least_one_task_FAIL_observed=killed3,
                requirement_status='PASS' if killed3 else 'FAIL'))
    for name, gate in gates.items():
        emit(f"{name} {gate['status']} VALID={gate['counts']['VALID']}/{gate['n']} rate={gate['valid_rate']:.6f} (remeasured)")
    for name in ('G1', 'G2'):
        killer = killers[name]
        emit(f"KILLER {name} {killer['status']} VALID={killer['counts']['VALID']}/{killer['n']} (remeasured)")
    emit('KILLER G3 no-lower-bound (remeasured): ' + json.dumps(killer3_by_task))
    return dict(schema='hsweep-gate-evidence-v1', cache_key=list(gate_cache_key(result['provenance'])),
        provenance=result['provenance'], maze=result['maze'], routes=result['routes'],
        crosscheck=result['crosscheck'], ruler=result['ruler'], thresholds=result['thresholds'],
        seed_formula=result['seed_formula'], cells=cells, gates=gates, killers=killers, n_cells=len(cells))


def full():
    result = dict(schema='selfsub-v1-r1', status='RUNNING', gates={}, killers={}, cells=[],
                  summary={str(t): dict(r=None, interpretation=None, reason='尺尚未驗證') for t in TASKS},
                  descriptive={}, thresholds=dict(G1=1., G2=.90, G3=.10, usable=.70, unusable=.30),
                  ruler=dict(smooth_window=9, convolution='valid', anchor=True, arc=12., cell_size=4., M=16),
                  seed_formula='task*10000 + i*100 + j',
                  execution=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                                 CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                                 MUJOCO_GL=os.environ['MUJOCO_GL'], dont_write_bytecode=sys.dont_write_bytecode))
    env = None
    try:
        sys.path.insert(0, str(PROBE))
        import harness_detour as d
        import harness_move1 as m1
        import common
        emit('LOAD frozen M9 checkpoint on CPU via eval_common.load_ckpt')
        from eval_common import load_ckpt
        module = load_ckpt('/home/cymaxwelllee/data/ogbench', CKPT_PATH, CKPT_SHA256)
        torch = module.torch
        if str(module.device) != 'cpu':
            raise StopProbe(f'CPU required, device={module.device}')
        result['provenance'] = module.probe_provenance
        result['execution']['device'] = str(module.device)
        env = module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
        graph = Graph(env.unwrapped.maze_map,
                      lambda c: np.asarray(env.unwrapped.ij_to_xy(c), dtype=np.float64),
                      lambda xy: tuple(int(v) for v in env.unwrapped.xy_to_ij(xy)))
        if graph.maze.shape != (9, 12):
            raise StopProbe(f'Maze shape mismatch: {graph.maze.shape}')
        for i in range(9):
            for j in range(12):
                if graph.xy_to_cell(graph.cell_to_xy((i,j))) != (i,j):
                    raise StopProbe(f'Coordinate roundtrip mismatch: {(i,j)}')
        trap = json.loads(TRAP.read_text())
        inventory = make_inventory(graph, trap)
        result['maze'] = graph.maze.tolist()
        result['routes'] = {str(t): dict(s=list(s['s']), g=list(s['g']),
            length=graph.dist[s['g']][s['s']], paths=[[list(c) for c in p] for p in s['paths']])
            for t, s in inventory.items()}
        # Reference only enters after independent BFS has completed.
        reference = json.loads((ROOT/'ref-routes-density.json').read_text())
        check = {}
        for task, spec in inventory.items():
            ref = reference['tasks'][str(task)]
            equal = {tuple(p) for p in [[tuple(c) for c in x] for x in ref['paths']]} == {tuple(p) for p in spec['paths']}
            check[str(task)] = dict(path_sets_equal=equal, route_count=len(spec['paths']),
                                    distance=graph.dist[spec['g']][spec['s']], ref_distance=ref['L'])
            if not equal or check[str(task)]['distance'] != ref['L']:
                raise StopProbe(f'Independent BFS/reference discrepancy: {check[str(task)]}')
        result['crosscheck'] = check
        emit('BFS_CROSSCHECK ' + json.dumps(check))
        evidence_path = GATE_EVIDENCE or ROOT/'gates-v1-evidence.json'
        candidate = read_gate_evidence(evidence_path) if evidence_path.is_file() else None
        evidence, reused = resolve_gate_evidence(candidate, result, inventory,
            lambda: recompute_gate_evidence(module, graph, inventory, trap, result))
        result['gates'] = evidence['gates']
        result['killers'] = evidence['killers']
        result['cells'] = evidence['cells']
        result['gate_evidence_cache'] = dict(key=evidence['cache_key'], reused=reused,
                                            evidence='gates-evidence.json')
        save('gates-evidence.json', evidence)
        if reused and evidence_path == ROOT/'gates-v1-evidence.json':
            # Retain the original fields exactly for E2; new cache metadata is separate.
            result['gate_reuse'] = dict(source=evidence_path.name,
                sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                original_stdout='gates-v1-output.txt',
                authorization='REWORK-selfsub-v1-r1.md: gates and 50 cells retained',
                gates_remeasured=False)
        else:
            result['gate_reuse'] = dict(source=str(evidence_path) if reused else None,
                sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest() if reused else None,
                gates_remeasured=not reused)
        for name in ('G1', 'G2', 'G3'):
            gate = result['gates'][name]
            emit(f"{name} {gate['status']} VALID={gate['counts']['VALID']}/{gate['n']} rate={gate['valid_rate']:.6f} ({'cached' if reused else 'remeasured'})")
        for name, label in (('G1', 'single-trapset-route'), ('G2', 'no-smoothing')):
            killer = result['killers'][name]
            emit(f"KILLER {name} {label}: {killer['status']} VALID={killer['counts']['VALID']}/{killer['n']}")
        killer3_by_task, killer3_rows = {}, []
        for row in result['cells']:
            row['flow'] = []
            row['flow_state'] = 'not_measured: gates pending'
            refs = row['references']
            # The old mutation is descriptive only; its original evidence is
            # preserved in gates-v1-output.txt and repeated by --selftest.
            if 'greedy_no_wall_killer' in refs:
                refs['greedy_through_wall_description'] = refs.pop('greedy_no_wall_killer')
            mutated = score_mutant(refs['greedy'], 'no_lower_bound')
            refs['greedy_no_lower_bound_killer'] = mutated
            if row['away']:
                killer3_rows.append(dict(task=row['task'], cell=row['cell'], **mutated))
        for task in TASKS:
            tally = counts([r for r in killer3_rows if r['task'] == task])
            killer3_by_task[str(task)] = dict(tally,
                status='PASS' if tally['valid_rate'] <= .10 else 'FAIL')
        killed3 = any(r['status'] == 'FAIL' for r in killer3_by_task.values())
        result['killers']['G3'] = dict(variant='no_lower_bound: 0 <= prog <= 4',
            by_task=killer3_by_task, away_records=killer3_rows,
            required_at_least_one_task_FAIL_observed=killed3,
            requirement_status='PASS' if killed3 else 'FAIL')
        emit('KILLER G3 no-lower-bound: ' + json.dumps(killer3_by_task))
        example = next((r for r in killer3_rows if r['prog'] == 0 and r['cat'] == 'VALID'), None)
        emit('KILLER G3 STAY example: ' + json.dumps(example))
        emit(f'KILLER G3 required at least one task FAIL observed: {killed3}')
        if not killed3:
            raise StopProbe('G3 no-lower-bound killer did not fail any task')
        # Write the new evidence even when a gate/killer stops this checkpoint.
        evidence['killers'] = result['killers']
        if any(g['status'] != 'PASS' for g in result['gates'].values()):
            raise StopProbe('Remeasured/cached ruler gate failed; no flow conclusion')
        if any(result['killers'][name]['status'] != 'FAIL' for name in ('G1', 'G2')):
            raise StopProbe('G1/G2 killer did not FAIL; no flow conclusion')
        emit('ALL GATES AND KILLERS VERIFIED; sampling 16 flow plans per cell')
        with torch.no_grad():
            for row in result['cells']:
                task, c = row['task'], tuple(row['cell'])
                spec, obs_xy = inventory[task], graph.cell_to_xy(c)
                goal_xy = graph.cell_to_xy(spec['g'])
                torch.manual_seed(row['seed'])
                cond = module.condvec(module.normstate(obs_xy), module.normstate(module.goal_to_obs(goal_xy)))
                u = module.sample_plan(16, cond.expand(16, -1), None)
                if u.shape[0] != 16:
                    raise StopProbe(f'sample_plan API mismatch: {u.shape}')
                for m in range(16):
                    raw = d.decode(module, u[m:m+1], obs_xy)
                    scored, points = measure(graph, c, spec['g'], raw)
                    arc8, _ = measure(graph, c, spec['g'], raw, length=8.)
                    arc16, _ = measure(graph, c, spec['g'], raw, length=16.)
                    unanchored, _ = measure(graph, c, spec['g'], raw, anchor=False)
                    cp = tuple(scored['c_p'])
                    row['flow'].append(dict(scored, draw=m, arc8=arc8, arc16=arc16,
                        unanchored=unanchored, raw_start_distance=float(np.linalg.norm(raw[0]-obs_xy)),
                        landing_route_ids=[k for k, p in enumerate(spec['paths']) if cp in p],
                        raw_xy=np.asarray(raw, dtype=float).tolist(), smooth_anchored_xy=points.tolist(),
                        decoded_endpoint_description=graph.score(c, spec['g'], points[-1])))
                row['flow_state'] = 'measured'
                emit(f"FLOW task={task} cell={c} seed={row['seed']} VALID={counts(row['flow'])['counts']['VALID']}/16")
        for task in TASKS:
            rows = [r for r in result['cells'] if r['task'] == task]
            away = counts([d for r in rows if r['away'] for d in r['flow']])
            other = counts([d for r in rows if not r['away'] for d in r['flow']])
            r = away['valid_rate']
            if r is None:
                raise StopProbe(f'Task {task} has no away observations')
            label = '這層 flow 自產近子目標大多可用' if r >= .70 else '這層 flow 自產近子目標大多不可用' if r <= .30 else '部分'
            emit(f"PRIMARY task={task} VALID={away['counts']['VALID']}/{away['n']} r={r:.6f} {label}")
            result['summary'][str(task)] = dict(r=r, interpretation=label, away=away, nonaway=other,
                n_cells=len(rows), n_away_cells=sum(q['away'] for q in rows),
                arc8=counts([d['arc8'] for q in rows for d in q['flow']]),
                arc16=counts([d['arc16'] for q in rows for d in q['flow']]),
                unanchored=counts([d['unanchored'] for q in rows for d in q['flow']]))
        key2 = max([r for r in result['cells'] if r['task'] == 2 and r['away']],
                   key=lambda r: (r['d_to_g'], tuple(-v for v in r['cell'])))
        targets = [(4, inventory[4]['s'], 'fig_A_task4_start.png'),
                   (5, inventory[5]['s'], 'fig_B_task5_start.png'),
                   (2, tuple(key2['cell']), 'fig_C_task2_key.png')]
        for task, c, filename in targets:
            row = next(r for r in result['cells'] if r['task'] == task and tuple(r['cell']) == c)
            render(graph, inventory[task], row, filename)
            result['descriptive'][str(task)] = dict(cell=list(c),
                landing_route_ids=[d['landing_route_ids'] for d in row['flow']],
                raw_start_distances=[d['raw_start_distance'] for d in row['flow']])
            if task == 4:
                render(graph, inventory[task], row, 'fig_P_planted.png', planted=True)
                save('_lead_only/KEY-planted.json', dict(planted='fig_P_planted.png',
                     error='16 flow 落點換成 (+) 真近子目標格心，依完整最短路循環；軌跡、參照、標題與 fig_A 相同。'))
        result['status'] = 'DONE'
    except Exception as exc:
        result['status'] = 'BLOCKED'
        result['blocked_reason'] = str(exc)
        result['error_original'] = traceback.format_exc()
        emit(result['error_original'])
        result['not_produced'] = ['fig_A_task4_start.png、fig_B_task5_start.png、fig_C_task2_key.png、fig_P_planted.png、_lead_only/KEY-planted.json：停止條件觸發，未量測 flow，無可繪製真軌跡。']
        for row in result['cells']:
            if not row['flow']:
                row['flow_state'] = 'not_measured: ' + str(exc)
        for summary in result['summary'].values():
            summary['reason'] = '停止條件：' + str(exc)
    finally:
        if env is not None:
            env.close()
        finish_report(result)
    emit(f"n_cells={result['n_cells']} n_draws={result['n_draws']}")
    emit('STATUS: ' + result['status'] + (' — ' + result['blocked_reason'] if result['status'] == 'BLOCKED' else ''))
    return 0 if result['status'] == 'DONE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--ckpt-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gate-evidence', type=Path, help='Optional checkpoint/ruler keyed gate cache')
    args = parser.parse_args()
    CKPT_PATH, CKPT_SHA256, OUT = args.ckpt, args.ckpt_sha256, args.out
    GATE_EVIDENCE = args.gate_evidence
    OUTROOT = OUT.parent / (OUT.name + '.artifacts')
    if OUT.exists() or OUT.is_symlink() or OUTROOT.exists() or OUTROOT.is_symlink():
        parser.exit(2, 'BLOCKED: --out or artifact directory already exists\n')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUTROOT.mkdir()
    sys.exit(selftest() if args.selftest else full())
