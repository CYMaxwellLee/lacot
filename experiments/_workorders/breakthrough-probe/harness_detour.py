"""Detour-u v8. Independent protocol; existing PB files remain byte-identical."""
import argparse
import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import common
from common import Blocked, canonical, digest_array, file_sha, sha, write_json, verify_source
import harness_move1 as m1
import harness_move3 as m3
import runtime

HERE = Path(__file__).resolve().parent
PROTOCOL = 'detour-u-v8'
CARD = HERE / 'CONTRACT-FROZEN-detour.md'
CARD_PIN = '206306391e136e309b52e4184a3a9fa94dce8128df2a7d832289a8b6ca184051'
TRAPSET = HERE / 'detour-trapset-v2.json'
TRAPSET_PIN = '3b2662ba0f4e791228b7cf661932119f78653d755f2815350eaaf52ece0396fc'
M9_PIN = '276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc'
COLLECTOR_PIN = '7d8e2bf3b8760388ffa0cdad5460b4ea298c738325d5bf25bc084a0a248c26a5'
CODE_FILES_DETOUR = tuple(sorted(set(common.CODE_FILES) | {
    'harness_detour.py', 'harvest_detour.py', 'detour_plan.py', 'detour_selftest.py',
    'detour-smoke.sbatch', 'detour-formal.sbatch', '../ucontrast1/collector.py'}))
OUTPUT_ROOT = Path('/archive/cymaxwelllee/breakthrough1/detour-u')
H = 1000
ARMS = ('O3','F3','OF','Q-C','Q-O3','Q-F3')
FLOW_ARMS = ('F3','Q-C','Q-F3')
SHORT_ARMS = ('O3','F3','Q-O3','Q-F3')
RULER = HERE/'detour-horizon-ruler.json'
CPU_E1 = HERE/'detour-e1-v6-h3.json'
RULER_PIN = 'c3af2cc005f6a1a355fbb49af5af927d171d3d765aa32be4947ff89fd7536805'
CPU_E1_PIN = 'a248ab6be1e968d572fda812bae4348852959bb407ba02134e8cb829454f0de8'



def checked_json(path, expected=None):
    path = Path(path)
    actual = file_sha(path)
    sidecar_path = Path(str(path) + '.sha256')
    sidecar = sidecar_path.read_text().split()[0] if sidecar_path.exists() else expected
    if sidecar is None:
        raise Blocked(f'BLOCKED: missing input hash: {path}')
    if actual != sidecar or (expected is not None and actual != expected):
        raise Blocked(f'BLOCKED: input SHA256 mismatch: {path}')
    return json.loads(path.read_text())


def verify_pins():
    for path, pin in ((CARD, CARD_PIN), (TRAPSET, TRAPSET_PIN),
                      (common.SOURCE, M9_PIN), (common.COLLECTOR, COLLECTOR_PIN),
                      (RULER,RULER_PIN), (CPU_E1,CPU_E1_PIN)):
        if file_sha(path) != pin:
            raise Blocked(f'BLOCKED: frozen input mismatch: {path}')
    common.verify_source()


def code_hashes():
    return {name: file_sha(HERE / name) for name in CODE_FILES_DETOUR}


def read_flags():
    flags = json.loads((HERE / 'detour-flags.json').read_text())
    if set(flags) != {'PRODUCTION_READY', 'SMOKE_ENABLED'} or any(type(v) is not bool for v in flags.values()):
        raise Blocked('BLOCKED: detour-flags.json requires exactly two boolean release flags')
    return flags


def release_check(smoke):
    flags = read_flags()
    if not smoke and not flags['PRODUCTION_READY']:
        raise Blocked('BLOCKED: PRODUCTION_READY=False; lead release requires GPU smoke')
    if smoke and not flags['SMOKE_ENABLED']:
        raise Blocked('BLOCKED: SMOKE_ENABLED=False')


def seeds_detour(task, ep, draw, injected=False):
    # Match common.seeds' default for the factory; O3/OF explicitly pass injected=True.
    if type(draw) is not int or draw not in range(80, 84):
        raise ValueError('draw outside 80..83')
    stream = 7 * task + ep + 1_000_003 * draw
    return dict(env_seed=1000*task+ep, numpy_seed=1000*task+ep,
                action_space_seed=1000*task+ep, base_seed=7*task+ep,
                stream_seed=stream, flow_seed=None if injected else stream,
                noise_seed=stream, python_seed=stream, torch_seed=stream)


def _make_collector_with(seed_fn, task, episode, draw, obs, goal):  # harness_move3.py:26
    """Use the read-only Collector's A fresh-flow branch and paired noise RNG."""  # harness_move3.py:27
    verify_source()  # harness_move3.py:28
    import sys  # harness_move3.py:29
    from common import REPO  # harness_move3.py:30
    sys.path.insert(0, str(REPO))  # harness_move3.py:31
    from experiments._workorders.ucontrast1.collector import Collector  # harness_move3.py:32
    collector = Collector('A', 0., 16)  # harness_move3.py:33
    collector.begin(task, episode, draw, obs, goal)  # harness_move3.py:34
    expected = seed_fn(task, episode, draw)  # harness_move3.py:35 (only body substitution)
    if collector.flow_seed != expected['flow_seed']:  # harness_move3.py:36
        raise RuntimeError('fresh-flow branch mismatch')  # harness_move3.py:37
    return collector  # harness_move3.py:38


def make_collector_detour(task, episode, draw, obs, goal):
    # Lead ruling: same factory body, detour seeds; O3/OF receipt flow_seed stays None.
    return _make_collector_with(seeds_detour, task, episode, draw, obs, goal)


def load_trapset():
    return checked_json(TRAPSET, TRAPSET_PIN)


def cell_key(c):
    return ','.join(str(int(v)) for v in c)


def free_cells(trapset, task):
    return sorted(tuple(map(int, k.split(','))) for k in trapset['tasks'][str(task)]['oracle_table'])


def effective_cell(env, xy, free):
    raw = tuple(int(v) for v in env.unwrapped.xy_to_ij(xy))
    eff = raw if raw in free else min(free, key=lambda c: (
        float(np.sum((np.asarray(env.unwrapped.ij_to_xy(c))-xy)**2)), c))
    return raw, eff


def oracle_points(trapset, task, cell, cell_to_xy):
    route = trapset['tasks'][str(task)]['oracle_table'][cell_key(cell)]['route']
    return np.asarray([cell_to_xy(c) for c in route], np.float64)


def resample(points):
    """Get reference τ through the required Move 1 encoder-input path (no second interpolator)."""
    import torch
    capture = []
    def encode(x):
        capture.append(x.detach().cpu().numpy()[0].copy())
        return torch.zeros((1, 1, 1))
    adapter = SimpleNamespace(torch=torch, T_CAP=128, MU_XY=0., SD_XY=1.,
                              device='cpu', K=1, D_MODEL=1, encode_u=encode)
    m1.encode_trajectory(adapter, points)
    return capture[0].astype(np.float64)


def segment_box_distance(a, b, center, half=2.):
    """Exact closed segment / axis-aligned square distance, including zero length."""
    a, b, center = map(lambda x: np.asarray(x, np.float64), (a, b, center))
    lo, hi, delta = center-half, center+half, b-a
    enter, leave = 0., 1.
    intersects = True
    for axis in range(2):
        if delta[axis] == 0:
            if a[axis] < lo[axis] or a[axis] > hi[axis]:
                intersects = False
                break
        else:
            x, y = sorted(((lo[axis]-a[axis])/delta[axis], (hi[axis]-a[axis])/delta[axis]))
            enter, leave = max(enter, x), min(leave, y)
            if enter > leave:
                intersects = False
                break
    if intersects:
        return 0.
    def point_box(p):
        return float(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0.)))
    result = min(point_box(a), point_box(b))
    squared = float(delta @ delta)
    for corner in (lo, hi, np.array([lo[0], hi[1]]), np.array([hi[0], lo[1]])):
        t = 0. if squared == 0 else float(np.clip((corner-a) @ delta / squared, 0., 1.))
        result = min(result, float(np.linalg.norm(corner - (a+t*delta))))
    return result


def clearance(points, walls):
    # Vectorized exact same endpoint/corner/slab construction as segment_box_distance.
    p = np.asarray(points, np.float64)
    a, b = p[:-1, None, :], p[1:, None, :]
    centers = np.asarray(walls, np.float64)[None, :, :]
    lo, hi, delta = centers-2., centers+2., b-a
    denominator = np.where(delta == 0, 1., delta)
    t1, t2 = (lo-a)/denominator, (hi-a)/denominator
    enter = np.where(delta == 0, -np.inf, np.minimum(t1, t2)).max(axis=2)
    leave = np.where(delta == 0, np.inf, np.maximum(t1, t2)).min(axis=2)
    parallel_outside = np.any((delta == 0) & ((a < lo) | (a > hi)), axis=2)
    hit = (~parallel_outside) & (np.maximum(enter, 0.) <= np.minimum(leave, 1.))
    da = np.linalg.norm(np.maximum(np.maximum(lo-a, a-hi), 0.), axis=2)
    db = np.linalg.norm(np.maximum(np.maximum(lo-b, b-hi), 0.), axis=2)
    distance = np.minimum(da, db)
    squared = np.sum(delta*delta, axis=2)
    for sign in ((-1,-1), (-1,1), (1,-1), (1,1)):
        corner = centers + 2*np.asarray(sign)
        t = np.clip(np.sum((corner-a)*delta, axis=2) / np.where(squared == 0, 1., squared), 0., 1.)
        distance = np.minimum(distance, np.linalg.norm(corner-(a+t[:,:,None]*delta), axis=2))
    return float(np.min(np.where(hit, 0., distance)))


def discrete_frechet(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    previous = np.full(len(b), np.inf)
    for i in range(len(a)):
        current = np.empty(len(b))
        for j in range(len(b)):
            prior = 0. if i == j == 0 else min(previous[j], current[j-1] if j else np.inf,
                                               previous[j-1] if j else np.inf)
            current[j] = max(distances[i, j], prior)
        previous = current
    return float(previous[-1])


def e1_check(decoded, tau, walls, goal=None):
    """Legacy v5 geometry: OF qualification is descriptive; O3 uses direction_check."""
    decoded, tau = np.asarray(decoded, np.float64), np.asarray(tau, np.float64)
    if decoded.shape != (128, 2) or tau.shape != (128, 2) or not np.isfinite([decoded, tau]).all():
        raise ValueError('E1 requires finite [128,2] trajectories')
    gap, frechet = clearance(decoded, walls), discrete_frechet(decoded, tau)
    end = None if goal is None else float(np.linalg.norm(decoded[-1]-goal))
    a, b, c = gap >= .7, frechet <= 2., end is None or end <= 1.
    return dict(clearance=gap, frechet=frechet, goal_distance=end, a=a, b=b, c=c,
                qualification='PASS' if a and b and c else 'FAIL')


def direction_threshold():
    ruler = checked_json(RULER,RULER_PIN)
    return float(np.quantile([r['cos_4'] for r in ruler['real'] if r['cos_4'] is not None],.1))


def short_points(traps, task, cell, cell_to_xy, xy, goal):
    route = traps['tasks'][str(task)]['oracle_table'][cell_key(cell)]['route']
    k = min(3,len(route)-1)
    w = route[k]
    target = np.asarray(goal).copy() if w == traps['tasks'][str(task)]['g'] else np.asarray(cell_to_xy(w))
    points = np.asarray([xy,target]) if k == 0 else np.asarray([cell_to_xy(c) for c in route[:k]]+[target])
    return w,target,points


def direction_metrics(D, P, G, threshold):
    vectors = [np.asarray(v,np.float64) for v in (D,P,G)]
    lengths = [np.linalg.norm(v) for v in vectors]
    if min(lengths) < 1e-6:
        return dict(cos_plan=None,cos_greedy=None,cos_plan_greedy=None,qualification='FAIL')
    D,P,G = vectors; nd,np_,ng = lengths
    cp,cg,pg = float(D@P/(nd*np_)),float(D@G/(nd*ng)),float(P@G/(np_*ng))
    return dict(cos_plan=cp,cos_greedy=cg,cos_plan_greedy=pg,
                qualification='PASS' if cp >= threshold and (cp>cg or pg>np.cos(np.pi/4)) else 'FAIL')


def direction_check(decoded, points, goal, threshold=None):
    decoded = np.asarray(decoded,np.float64); points = np.asarray(points,np.float64)
    tau = np.repeat(points[:1],128,axis=0) if np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<1e-6 else resample(points)
    if decoded.shape != (128,2) or not np.isfinite(decoded).all():
        raise ValueError('E1 requires finite [128,2] decoded trajectory')
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(tau,axis=0),axis=1))]
    k = min(127,int(np.searchsorted(arc,4.)))
    D,P,G = decoded[k]-decoded[0],points[1]-points[0],np.asarray(goal)-points[0]
    return dict(direction_metrics(D,P,G,direction_threshold() if threshold is None else threshold),
                direction_D=D.tolist(),direction_P=P.tolist(),direction_G=G.tolist(),arc4_index=k)


def short_record(module, u, points, goal, walls, threshold, decode_xy=None):
    decoded = decode(module,u,points[0] if decode_xy is None else decode_xy)
    tau = np.repeat(points[:1],128,axis=0) if np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<1e-6 else resample(points)
    return {**e1_check(decoded,tau,walls,points[-1]),
            **direction_check(decoded,points,goal,threshold)}


def decode(module, u, xy):
    if module.OBS_DIM != 2:
        raise Blocked('BLOCKED: pointmaze OBS_DIM=2 required by v5 decoder contract')
    with module.torch.no_grad():
        points = module._dec(u, module.normstate(xy))[0]
        return (points * module.SD_XY_T + module.MU_XY_T).cpu().numpy()


def wall_centers(builder, cell_to_xy):
    return [cell_to_xy((i,j)) for i,row in enumerate(builder['maze']) for j,v in enumerate(row) if v == 1]


def geometry_context(env, trapset=None):
    trapset = load_trapset() if trapset is None else trapset
    builder = checked_json(HERE/'builder.json', trapset['inputs']['builder.json'])
    if not np.array_equal(env.unwrapped.maze_map, builder['maze']):
        raise Blocked('BLOCKED: environment maze differs from frozen builder')
    walls = wall_centers(builder, env.unwrapped.ij_to_xy)
    return trapset, builder, walls


def checker_controls(trapset, builder, cell_to_xy, walls):
    threshold = direction_threshold()
    positives, negatives, limits, goal_positives = [],[],[],[]
    for task in range(1,6):
        goal = np.asarray(cell_to_xy(trapset['tasks'][str(task)]['g']))
        points=np.asarray([goal-[0.,.4],goal+[0.,.4]])
        goal_positives.append(direction_check(resample(points),points,points[-1],threshold))
        for c in free_cells(trapset,task):
            if list(c) == trapset['tasks'][str(task)]['g']: continue
            w,target,points = short_points(trapset,task,c,cell_to_xy,cell_to_xy(c),goal)
            tau = resample(points)
            good = direction_check(tau,points,goal,threshold)
            positives.append(good['qualification']=='PASS')
            P,G = points[1]-points[0],goal-points[0]
            if 0 in trapset['tasks'][str(task)]['oracle_table'][cell_key(c)]['turn_away']:
                straight = direction_check(resample([points[0],goal]),points,goal,threshold)
                reverse = direction_check(2*tau[0]-tau,points,goal,threshold)
                negatives.append(dict(task=task,cell=list(c),final_goal=straight,reverse=reverse))
            direct = resample([points[0],target])
            direction = direction_check(direct,points,goal,threshold)
            if direction['qualification']=='PASS' and clearance(direct,walls)<.7:
                limits.append(dict(task=task,cell=list(c),qualification='PASS',clearance=clearance(direct,walls),
                                   limitation='直衝近目標切牆角；首方向閘不證明整段可執行'))
    boundaries = {str(delta):direction_metrics([threshold+delta,np.sqrt(1-(threshold+delta)**2)],
                    [1,0],[1,0],threshold) for delta in (-1e-7,0.,1e-7)}
    zeros = [direction_metrics(*vs,threshold) for vs in (([0,0],[1,0],[1,0]),([1,0],[0,0],[1,0]),([1,0],[1,0],[0,0]))]
    if not all(positives) or any(r['qualification']!='PASS' for r in goal_positives) or not negatives or any(r[k]['qualification']!='FAIL' for r in negatives for k in ('final_goal','reverse')):
        raise Blocked('BLOCKED: E1 direction controls failed')
    if boundaries[str(-1e-7)]['qualification']!='FAIL' or any(boundaries[str(v)]['qualification']!='PASS' for v in (0.,1e-7)) or any(z['qualification']!='FAIL' for z in zeros):
        raise Blocked('BLOCKED: E1 boundary/zero controls failed')
    return dict(positive_count=len(positives),positive_pass=sum(positives),negatives=negatives,
                boundaries=boundaries,zeros=zeros,goal_positives=goal_positives,known_limitations=limits)


def build_e1_table(module, *, env=None, out=None):
    """Actual table is generated on jasmine before rollouts, under load_frozen settings."""
    verify_pins()
    if not getattr(module,'detour_mock',False):
        import socket
        require_real_module(module)
        if socket.gethostname().split('.')[0] != 'jasmine':
            raise Blocked('BLOCKED: true E1 table requires jasmine GPU')
        if out is not None and not Path(out).resolve().is_relative_to(OUTPUT_ROOT.resolve()):
            raise Blocked('BLOCKED: true E1 output must be jasmine local /archive')
    owned = env is None
    if owned:
        env = module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
    try:
        traps, builder, walls = geometry_context(env)
        controls = checker_controls(traps,builder,env.unwrapped.ij_to_xy,walls)
        table = dict(protocol=PROTOCOL, card_sha256=CARD_PIN, trapset_sha256=TRAPSET_PIN,
                     source_sha256=M9_PIN, runtime=module.probe_provenance,code_sha256=code_hashes(),
                     synthetic=bool(getattr(module,'detour_mock',False)), cells={}, of_cells={}, controls=controls,
                     ruler_sha256=RULER_PIN,cpu_e1_sha256=CPU_E1_PIN,threshold=direction_threshold(),h=3)
        for task in range(1,6):
            cells = table['cells'][str(task)] = {}
            of_cells = table['of_cells'][str(task)] = {}
            g = tuple(traps['tasks'][str(task)]['g']); goal = np.asarray(env.unwrapped.ij_to_xy(g))
            for c in free_cells(traps,task):
                if c == g: continue
                raw = oracle_points(traps,task,c,env.unwrapped.ij_to_xy)
                u = m1.encode_trajectory(module,raw)
                of_cells[cell_key(c)] = dict(e1_check(decode(module,u,raw[0]),resample(raw),walls),
                    tau_sha256=digest_array(raw), u_sha256=digest_array(m1.array_of(u)),
                    delivered_sha256=digest_array(m1.array_of(module._q(u))))
                w,target,raw = short_points(traps,task,c,env.unwrapped.ij_to_xy,env.unwrapped.ij_to_xy(c),goal)
                if tuple(w)==g: continue  # No static hashes for the noisy-goal tail.
                u = m1.encode_trajectory(module,raw)
                cells[cell_key(c)] = dict(short_record(module,u,raw,goal,walls,table['threshold']),
                    w=w,target_xy=target.tolist(),h=3,tau_endpoint=raw[-1].tolist(),tau_dtype=str(raw.dtype),
                    tau_sha256=digest_array(raw),u_sha256=digest_array(m1.array_of(u)),
                    delivered_sha256=digest_array(m1.array_of(module._q(u))))
        cpu = checked_json(CPU_E1,CPU_E1_PIN)['table']
        table['cpu_comparison'] = dict(compared=sum(len(set(cells)&set(cpu[t])) for t,cells in table['cells'].items()),
            disagreements=[dict(task=int(t),cell=k,gpu_qualification=c['qualification'],cpu_pass=cpu[t][k]['pass_'])
                for t,cells in table['cells'].items() for k,c in cells.items()
                if k in cpu[t] and (c['qualification']=='PASS')!=cpu[t][k]['pass_']])
        table['critical_cells'] = {t:[dict(cell=cell_key(c),**table['cells'][t].get(cell_key(c),dict(qualification='runtime-only')))
            for c in record['route_from_s'] if 0 in record['oracle_table'][cell_key(c)]['turn_away']]
            for t,record in traps['tasks'].items()}
        raw = np.array([env.unwrapped.ij_to_xy(c) for c in (
            builder['geometries']['4']['s'],builder['geometries']['4']['g'])])
        u = m1.encode_trajectory(module,raw)
        table['end_to_end_wall_record_only'] = e1_check(decode(module,u,raw[0]),resample(raw),walls)
        if out is not None:
            write_json(out,table)
        return table
    finally:
        if owned:
            env.close()


class Provider:
    def __init__(self, module, env, table, arm='O3'):
        self.module,self.env,self.table,self.arm = module,env,table,arm
        self.traps,self.builder,self.walls = geometry_context(env)
        self.cache = {}; self.previous = None
        self.threshold = direction_threshold()

    def get(self, task, xy, goal):
        raw,eff = effective_cell(self.env,xy,free_cells(self.traps,task))
        g = tuple(self.traps['tasks'][str(task)]['g'])
        short = self.arm in SHORT_ARMS
        w,target,points = short_points(self.traps,task,eff,self.env.unwrapped.ij_to_xy,xy,goal)
        if not short:
            w,target = list(g),goal
            points = np.asarray([xy,goal]) if eff==g else oracle_points(self.traps,task,eff,self.env.unwrapped.ij_to_xy)
        fields = dict(c_raw=list(raw),c_eff=list(eff),source_cell=list(eff),w=list(w),target_xy=target.tolist(),h=3 if short else None)
        if eff==g and np.linalg.norm(np.asarray(xy)-goal)<1e-6:
            if self.previous is None: raise Blocked('BLOCKED: zero-length goal has no previous chunk u')
            u,record = self.previous; record=copy.deepcopy(record)
            record.update(fields,mode='零長度沿用',qualification_source='runtime')
        else:
            static = tuple(w)!=g if short else eff!=g
            key = task, eff, 3 if short else None  # c_raw is expressly NOT the cache key.
            # Tail caching is local to this draw. Goal-cell points also depend on exact xy.
            if static and key in self.cache:
                u,record = self.cache[key]; record=copy.deepcopy(record)
            else:
                u = m1.encode_trajectory(self.module,points)
                record = short_record(self.module,u,points,np.asarray(self.env.unwrapped.ij_to_xy(g)) if static else goal,self.walls,self.threshold) if short else e1_check(decode(self.module,u,points[0]),resample(points),self.walls,goal if eff==g else None)
                record.update(tau_sha256=digest_array(points),tau_dtype=str(points.dtype),tau_endpoint=points[-1].tolist(),
                    u_sha256=digest_array(m1.array_of(u)),expected_delivered_sha256=digest_array(m1.array_of(self.module._q(u))),
                    mode='static' if static else 'goal' if eff==g else 'tail',qualification_source='靜態表' if static else 'runtime')
                if static:
                    frozen=self.table['cells' if short else 'of_cells'][str(task)][cell_key(eff)]
                    if any(record[k]!=frozen[k] for k in ('u_sha256','tau_sha256','qualification')) or record['expected_delivered_sha256']!=frozen['delivered_sha256']:
                        raise ValueError('E0: provider differs from frozen E1 table')
                    self.cache[key]=u,copy.deepcopy(record)
            if static:
                frozen=self.table['cells' if short else 'of_cells'][str(task)][cell_key(eff)]
                if digest_array(m1.array_of(u))!=frozen['u_sha256'] or record['tau_sha256']!=frozen['tau_sha256']:
                    raise ValueError('E0: cache source differs from frozen E1 table')
            record.update(fields)
        self.previous=u,copy.deepcopy(record)
        return u,record


def run_draw_detour(module, env, plan, arm):
    if plan.get('protocol')!=PROTOCOL: raise ValueError('unknown detour protocol')
    if arm not in ARMS or arm!=plan['arm']: raise ValueError('unknown/mismatched detour arm')
    # Instrument the real flow.sample boundary, not the number of requested samples.
    calls=[]; context={}
    original=module.flow.sample
    def sample(*args,**kwargs):
        calls.append(copy.deepcopy(context))
        return original(*args,**kwargs)
    module.flow.sample=sample
    try:
        return _run_draw_detour(module,env,plan,arm,calls,context)
    finally:
        module.flow.sample=original


def recorded_condition(module, obs, target):
    """Return the actual condition argument with the tensor built from that argument."""
    actual_target=np.asarray(target).copy()
    return runtime.condition(module,obs,actual_target),actual_target.tolist()


def _run_draw_detour(module, env, plan, arm, calls, flow_context):
    verify_pins()
    task,episode,draw = (plan[k] for k in ('task','episode','draw'))
    seeds = seeds_detour(task,episode,draw,arm not in FLOW_ARMS)
    # runtime.py:162-164 reset; :113-119 policy cache and RNG reset.
    obs,info = common.reset_env(env,task,episode)
    initial,goal = np.asarray(obs).copy(),np.asarray(info['goal']).copy()
    random.seed(seeds['python_seed']); module.torch.manual_seed(seeds['torch_seed'])
    module._GRAD_CACHE['u']=None
    module._reseed_shuf(1000*task+episode); module._RDIR[0]=1.
    collector=make_collector_detour(task,episode,draw,obs,goal)
    provider=Provider(module,env,module.detour_e1_table,arm)
    before=m1.rng_snapshot(module.torch,env,collector.noise)
    xy,success_trace,chunks,heads,actions_sent = [initial[:2].tolist()],[False],[],[],[]
    terminated=truncated=False; reason=None
    while len(actions_sent)<H and reason is None:
        step=len(actions_sent)
        with module.torch.no_grad():
            raw,eff=effective_cell(env,np.asarray(obs[:2]),free_cells(provider.traps,task))
            w,target,points=short_points(provider.traps,task,eff,env.unwrapped.ij_to_xy,obs[:2],goal)
            if arm not in SHORT_ARMS: w,target=provider.traps['tasks'][str(task)]['g'],goal
            cond,cond_target_xy=recorded_condition(module,obs,target)
            flow_context.update(step=step,target_xy=cond_target_xy)
            if arm not in FLOW_ARMS:
                u,rec=provider.get(task,np.asarray(obs[:2]),goal)
            else:
                u=collector.get_u(lambda:module.sample_plan(1,cond,None))
                rec=short_record(module,u,points,goal,provider.walls,provider.threshold,decode_xy=np.asarray(obs[:2])) if arm in SHORT_ARMS else dict(qualification='N/A')
                rec.update(c_raw=list(raw),c_eff=list(eff),mode='flow',tau_sha256=None,
                    w=list(w),target_xy=target.tolist(),h=3 if arm in SHORT_ARMS else None,
                    u_sha256=digest_array(m1.array_of(u)),qualification_source='runtime-flow' if arm in SHORT_ARMS else 'flow-reference',
                    expected_delivered_sha256=digest_array(m1.array_of(module._q(u))))
            injection=m1.FixedInjection(module.ahead,module._q(u))
            actions=injection(cond)[0].cpu().numpy()
            rec=dict(rec,step=step,xy=np.asarray(obs[:2]).tolist(),cond_target_xy=cond_target_xy,delivered_sha256=injection.head_hashes[0])
            if rec['delivered_sha256']!=rec['expected_delivered_sha256']: raise ValueError('E0: actual head delivery differs from expected')
        # runtime.py:179-191; no waypoint switch or stuck early termination.
        for action in actions:
            sent=m3.noisy_action(collector,action)
            heads.append(np.asarray(action,np.float32).tolist()); actions_sent.append(sent.tolist())
            obs,_,terminated,truncated,info=env.step(sent)
            success=bool(info.get('success',False)); xy.append(list(obs[:2])); success_trace.append(success)
            reason=('success' if success else 'terminated' if terminated else 'truncated' if truncated else 'horizon' if len(actions_sent)==H else None)
            if reason is not None: break
        rec['end_step'],rec['end_xy']=len(actions_sent),np.asarray(obs[:2]).tolist()
        chunks.append(rec)
    if len(calls)!=(len(chunks) if arm in FLOW_ARMS else 0): raise ValueError('E0: actual flow.sample count')
    # float64 trace preserves both float32 and float64 environment positions exactly.
    trace=np.asarray(xy,np.float64)
    return dict(protocol=PROTOCOL,task=task,episode=episode,draw=draw,arm=arm,group=plan['group'],
        seeds=seeds,source_sha256=M9_PIN,synthetic=bool(getattr(module,'detour_mock',False)),
        initial=initial.tolist(),goal=goal.tolist(),observation_dtype=str(initial.dtype),goal_dtype=str(goal.dtype),
        initial_sha256=digest_array(initial),goal_sha256=digest_array(goal),xy=trace.tolist(),
        trace_sha256=digest_array(trace),observed_steps=len(actions_sent),success=any(success_trace),
        goal_success=success_trace,termination_reason=reason,terminated=bool(terminated),truncated=bool(truncated),
        chunk_steps=module.CHUNK,chunks=chunks,flow_calls=calls,head_actions=heads,actions=actions_sent,
        failed_qualification_chunks=sum(c['qualification']=='FAIL' for c in chunks),
        rng_before=before,rng_after=m1.rng_snapshot(module.torch,env,collector.noise),
        env_properties=dict(success_timing=env.unwrapped._success_timing,add_noise_to_goal=env.unwrapped._add_noise_to_goal,goal_tol=env.unwrapped._goal_tol))


def layer_result(n, successes):
    from math import ceil, floor
    if not 0 <= successes <= n:
        raise ValueError('invalid layer count')
    return ('觀測不足' if n < 3 else '救回' if successes >= ceil(2*n/3)
            else '未救回' if successes <= floor(n/6) else '部分')


def question_success(bits):
    if len(bits) != 4:
        raise ValueError('O3/F3 question requires four draws')
    return sum(bits) >= 2


def failure_label(counts):
    total = sum(counts.values())
    return ('N/A' if total == 0 else '停在陷阱（首方向合格）' if counts['i']/total >= .8 else
            '近陷阱但首方向不合格' if (counts['i']+counts['ii'])/total >= .8 else '其他／混合型態')


def failure_types(rows, traps, xy_to_cell, norm='chebyshev'):
    if norm not in ('chebyshev','manhattan'):
        raise ValueError('unknown trap norm')
    counts, details = dict(i=0,ii=0,iii=0), []
    for row in rows:
        if row['success']:
            continue
        c = np.asarray(xy_to_cell(row['xy'][-1]))
        distances = [np.abs(c-t) for t in np.asarray(traps)]
        near = any((d.max() if norm == 'chebyshev' else d.sum()) <= 1 for d in distances)
        qualified = all(x['qualification']=='PASS' for x in row['chunks'])
        category = ('i' if qualified else 'ii') if near else 'iii'
        counts[category] += 1
        details.append(dict(task=row['task'],episode=row['episode'],draw=row['draw'],category=category,
                            failed_cells=[c['c_eff'] for c in row['chunks'] if c['qualification']=='FAIL']))
    return dict(counts=counts,denominator=sum(counts.values()),label=failure_label(counts),details=details)


def compliance(delta, plan_direction, greedy_direction):
    vectors = [np.asarray(v,np.float64) for v in (delta,plan_direction,greedy_direction)]
    lengths = [np.linalg.norm(v) for v in vectors]
    if min(lengths) < 1e-6:
        return dict(plan_cos=None,greedy_cos=None,preferred=None,status='N/A')
    p, g = (float(vectors[0]@vectors[k]/(lengths[0]*lengths[k])) for k in (1,2))
    return dict(plan_cos=p,greedy_cos=g,preferred=p>g,status='observed')


def q_result(o, f, c):
    if c<8:
        return dict(O3=o,F3=f,C=c,label='Q 觀測不足',annotations=['機制檢查無參考線'])
    statements={a:'本預算下沒有比 Q-C 少超過 2 次' if v>=c-2 else '變差' for a,v in (('O3',o),('F3',f))}
    return dict(O3=o,F3=f,C=c,label='Q 已判',statements=statements,
                annotations=(["格心短計畫在易題也變差"] if o<c-2 else [])+(["flow＋近目標在易題也變差"] if f<c-2 else []))


RULES = {'R1': '（本預算、題級門檻下）近子目標策略可行：每段給往前 3 步的近目標，oracle 短計畫與 flow 自己都過得去 ⇒ 下一步可以試「誰來產生近目標」。⚠️ 不證明原本失敗是長程規劃或表示的病', 'R2': '（本預算、題級門檻下）oracle 短計畫可行、flow 給近目標仍不行；剩三種解釋未分：flow 抽的路線內容錯／flow 的 u 動作頭讀不了／近目標 (s,w) 落在 flow 條件分布外（M9 hindsight goal 取樣）', 'R3': '（本預算、題級門檻下）flow＋近目標可行、格心短計畫不行；候選解釋並列、⛔ 不排序：格心 u 的格式（M9:1707-1734 動作頭只用真資料 E(τ) 學過）／格心路徑內容（解碼會切牆角）／可執行性／條件分布。⛔ 不是矛盾、⛔ 不當接線錯', 'R4': '（本預算、題級門檻下）兩種近目標計畫都不行；只報失敗三分＋順從度描述，⛔ 不給歸因', 'R5': '逐題明細呈主人'}


def conclusion(o, f, q):
    pairs={('救回','救回'):'R1',('救回','未救回'):'R2',('未救回','救回'):'R3',('未救回','未救回'):'R4'}
    rule=pairs.get((o,f),'R5')
    return dict(rule=rule,statement=RULES[rule],annotations=q.get('annotations',[]))


def collector_compatibility_check():
    """Exercise the approved factory and flow_seed cross-check for all new draws."""
    for draw in range(80,84):
        make_collector_detour(4,4,draw,np.zeros(2,np.float32),np.ones(2,np.float32))


def receipt(module, env, plan, table_path):
    verify_pins()
    base = env.unwrapped
    return dict(protocol=PROTOCOL,card_sha256=CARD_PIN,source_sha256=M9_PIN,collector_sha256=COLLECTOR_PIN,
        trapset_sha256=TRAPSET_PIN,code_sha256=code_hashes(),plan_sha256=sha(canonical(plan)),
        e1_sha256=file_sha(table_path),runtime=module.probe_provenance,H=H,sigma=common.SIGMA,env=common.ENV,
        offline_only=True,training_use_prohibited=True,
        env_properties=dict(success_timing=base._success_timing,add_noise_to_goal=base._add_noise_to_goal,
                            goal_tol=base._goal_tol),
        coordinate_system=dict(unit=base._maze_unit,offset_x=base._offset_x,offset_y=base._offset_y,
            centers={cell_key((i,j)):list(base.ij_to_xy((i,j))) for i,row in enumerate(base.maze_map) for j in range(len(row))}))


def require_real_module(module):
    if getattr(module,'detour_mock',False):
        raise Blocked('BLOCKED: mock module may not write production artifact')
    if not str(module.device).startswith('cuda') or not module.probe_provenance['deterministic_settings'].get('deterministic'):
        raise Blocked('BLOCKED: true detour E1 and rollouts require deterministic GPU settings')


def execute(outdir, dataset_dir, *, smoke=False, e1_table=None, resume_from=None, e1_only=False):
    """Single production path, exclusive files and PB/artifacts.py shard resume.

    Approved C factory compatibility is checked before model loading. e1_only
    remains runnable independently; smoke/formal includes Q-C.
    """
    from artifacts import resume_rows, save_row
    from detour_plan import validate_plan
    import harvest_detour as harvest
    release_check(smoke)
    verify_pins()
    if not e1_only:
        collector_compatibility_check()
    plan = validate_plan(checked_json(HERE/'detour-plan.json'))
    outdir = Path(outdir)
    if outdir.resolve().parent != OUTPUT_ROOT.resolve() or outdir.name not in ('smoke','formal','formal-r2'):
        raise Blocked('BLOCKED: output must be jasmine local detour-u/{smoke,formal,formal-r2}')
    if outdir.exists():
        raise Blocked('BLOCKED: choose a fresh output directory')
    import socket
    if socket.gethostname().split('.')[0] != 'jasmine':
        raise Blocked('BLOCKED: production outputs and E1 generation require jasmine')
    if not smoke and e1_table is None:
        raise Blocked('BLOCKED: formal requires frozen smoke E1 table')
    start = time.perf_counter()
    module = runtime.load_frozen(dataset_dir)  # configures determinism before model initialization
    env = module.ogbench.make_env_and_datasets(common.ENV,env_only=True)
    refs, rows, reused, times = [], [], {}, []
    manifest = None
    def artifact():
        return dict(protocol=PROTOCOL,plan=plan,receipt=manifest,smoke=smoke,storage='shards-v1',rows=refs,
                    conformance_rows=[],scientific_release=False)
    try:
        require_real_module(module)
        outdir.mkdir(parents=True,exist_ok=False)
        phase = time.perf_counter()
        if resume_from is not None:
            old_header, old_rows = resume_rows(resume_from)
            previous_table = Path(resume_from)/'e1-table-v8.json'
            if e1_table is not None and file_sha(e1_table)!=file_sha(previous_table):
                raise Blocked('BLOCKED: resume E1 differs')
            e1_table = previous_table
        if e1_table is None:
            table = build_e1_table(module,env=env,out=outdir/'e1-table-v8.json')
        else:
            table = checked_json(e1_table)
            harvest.validate_table(table,module.probe_provenance)
            write_json(outdir/'e1-table-v8.json',table)
        e1_seconds = time.perf_counter()-phase
        frozen_path = outdir/'e1-table-v8.json'
        harvest.validate_table(table,module.probe_provenance)
        if e1_only:
            write_json(outdir/'e1-generation.json',dict(e1_sha256=file_sha(frozen_path),runtime=module.probe_provenance,
                e1_seconds=e1_seconds,total_seconds=time.perf_counter()-start,rollouts=0))
            return table
        module.detour_e1_table = table
        manifest = receipt(module,env,plan,frozen_path)
        harvest.validate_receipt(manifest,plan,frozen_path)
        plans = plan['smoke' if smoke else 'formal']
        if resume_from is not None:
            harvest.validate_receipt(old_header['receipt'],plan,previous_table)
            harvest.require(old_header['receipt']==manifest and old_header['smoke']==smoke,'resume protocol/provenance')
            allowed = {harvest.key(p):p for p in plans}
            for row in old_rows:
                k = harvest.key(row)
                harvest.require(k in allowed and k not in reused,'extra/duplicate resume row')
                harvest.validate_row(row,allowed[k],table,env.unwrapped.xy_to_ij,env.unwrapped.ij_to_xy)
                reused[k] = row
        write_json(outdir/'checkpoint.json',artifact())
        for index,p in enumerate(plans):
            tick = time.perf_counter(); k = harvest.key(p)
            resumed = k in reused
            row = reused.pop(k) if resumed else run_draw_detour(module,env,p,p['arm'])
            refs.append(save_row(outdir,row,index)); rows.append(row)
            # Persist first, then validate: failed evidence survives for audit.
            harvest.validate_row(row,p,table,env.unwrapped.xy_to_ij,env.unwrapped.ij_to_xy)
            times.append(dict(task=p['task'],arm=p['arm'],episode=p['episode'],draw=p['draw'],
                              seconds=time.perf_counter()-tick,resumed=resumed))
            print(f'detour rollouts={len(rows)}/{len(plans)}',flush=True)
        write_json(outdir/'raw.json',artifact())
        harvest.validate_rows(rows,plans,table,env.unwrapped.xy_to_ij,env.unwrapped.ij_to_xy)
        result = harvest.readout(rows,plan,table,env.unwrapped.xy_to_ij,env.unwrapped.ij_to_xy,smoke)
        # Whole-draw times include provider cache creation, decoding and disk writes.
        weighted = 0.
        for task,arm in sorted({(p['task'],p['arm']) for p in plan['formal']}):
            samples = [t['seconds'] for t in times if t['task']==task and t['arm']==arm and not t['resumed']]
            if not samples:
                weighted = None
                break
            weighted += np.mean(samples)*sum(p['task']==task and p['arm']==arm for p in plan['formal'])
        overhead = time.perf_counter()-start-sum(t['seconds'] for t in times)
        write_json(outdir/'timing.json',dict(rollouts=times,e1_seconds=e1_seconds,load_setup_and_e1_seconds=overhead,
            total_seconds=time.perf_counter()-start,weighted_formal_seconds=None if weighted is None else float(weighted+overhead),
            proposed_time_limit_seconds=None if weighted is None else float(2*(weighted+overhead))))
        write_json(outdir/'result.json',dict(artifact(),readout=result))
        return result
    except Exception as error:
        if outdir.exists():
            if manifest is not None and not (outdir/'raw.json').exists():
                write_json(outdir/'raw.json',artifact())
            write_json(outdir/'validation_error.json',dict(error_type=type(error).__name__,message=str(error),
                                                        completed_rollouts=len(refs)))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--e1-table',type=Path)
    parser.add_argument('--e1-only',action='store_true')
    parser.add_argument('--resume-from',type=Path)
    args = parser.parse_args()
    try:
        execute(args.outdir,args.dataset_dir,smoke=args.smoke,e1_table=args.e1_table,
                resume_from=args.resume_from,e1_only=args.e1_only)
    except Blocked as error:
        parser.exit(2,str(error)+'\n')
