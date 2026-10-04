"""CPU checkpoint metrics; one load_ckpt invocation per process.

Geometry/prep/land_k are literal methods from the lead's five prototypes.
This file also supplies the shared, checkpoint-free geometry to the analyzer.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/wallpen-eval-matplotlib')
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = Path('/home/cymaxwelllee/Projects/lacot-hsweep-eval-frozen')
BACKGROUND = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u')
DATASET = Path('/home/cymaxwelllee/data/ogbench')
DATASET_PIN = '9add335e598e48ebc483447d61415a9755ffb738ddfc406cb9708b2a238992e8'
ENV = 'pointmaze-large-stitch-v0'
TAU = 0.0664
SEED = 20261004
for p in (FROZEN, FROZEN/'experiments/_workorders/breakthrough-probe',
          FROZEN/'experiments/_workorders/hsweep'):
    sys.path.insert(0, str(p))


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def prep(tau, sx):
    sm = np.stack([np.convolve(tau[:, i], np.ones(9) / 9, 'valid') for i in (0, 1)], 1)
    return sm - sm[:1] + sx


def land_k(sm):
    c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))]
    return len(sm) - 1 if c[-1] < 12 else int(np.searchsorted(c, 12))


class Geometry:
    """clean_baselines.py dep + same_ruler_teacher_vs_flow.py realdep."""
    def __init__(self, obs, mu, sd, env):
        import torch
        from scipy.ndimage import distance_transform_edt
        from lacot.refine_grad import GeoEnergy
        self.torch = torch
        self.mu, self.sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
        self.env = env
        self.mm = np.asarray(env.maze_map)
        self.geo = GeoEnergy(np.asarray(obs, np.float64), self.mu, self.sd, res=8, device='cpu')
        self.res = 20
        self.x0, self.y0 = self.c2xy((0, 0)) - 2
        self.wm = np.zeros((self.mm.shape[0]*4*20, self.mm.shape[1]*4*20), bool)
        for i, j in np.argwhere(self.mm == 1):
            x, y = self.c2xy((i, j))
            r0, c0 = int(round((y-2-self.y0)*20)), int(round((x-2-self.x0)*20))
            self.wm[r0:r0+80, c0:c0+80] = True
        self.inwall = distance_transform_edt(self.wm) / 20

    def c2xy(self, c):
        return np.asarray(self.env.ij_to_xy(tuple(c)), np.float64)

    def xy2c(self, xy):
        return tuple(int(v) for v in self.env.xy_to_ij(np.asarray(xy)))

    def depth(self, xy):
        with self.torch.no_grad():
            return self.geo.wall_depth(self.torch.tensor(
                ((np.asarray(xy)-self.mu)/self.sd)[None], dtype=self.torch.float32))[0].numpy()

    def realdep(self, xy):
        r = np.clip(((xy[:, 1]-self.y0)*20).astype(int), 0, self.wm.shape[0]-1)
        c = np.clip(((xy[:, 0]-self.x0)*20).astype(int), 0, self.wm.shape[1]-1)
        return float(np.where(self.wm[r, c], self.inwall[r, c], 0.0).max())

    def stats(self, sm):
        seg = np.asarray(sm, np.float64)[:land_k(sm)+1]
        return float(self.depth(seg).max()), self.realdep(seg)


def geometry_from_dataset(dataset=DATASET):
    """No checkpoint loaded: normalization exactly as frozen M9 lines 44–52."""
    import torch
    import ogbench
    torch.set_num_threads(1)
    path = Path(dataset)/(ENV+'.npz')
    if sha_file(path) != DATASET_PIN:
        raise ValueError('DATASET_PIN mismatch')
    with np.load(path, allow_pickle=False) as d:
        obs = np.asarray(d['observations'], np.float32)
    mu, sd = obs.mean(0), obs.std(0)+1e-6
    wrapped = ogbench.make_env_and_datasets(ENV, env_only=True)
    return Geometry(obs[:, :2], mu[:2], sd[:2], wrapped.unwrapped), wrapped


def positive_controls(module, geo, stageo, routes):
    import harness_detour as d
    import harness_move1 as m1
    entries = []
    for cell in stageo['cells']:
        t, c = str(cell['task']), tuple(cell['cell'])
        for route_id, path in enumerate(routes['tasks'][t]['paths']):
            path = [tuple(x) for x in path]
            if c not in path:
                continue
            k = path.index(c)
            seg = path[k:k+4]
            if len(seg) < 2:
                continue
            pts = np.asarray([geo.c2xy(x) for x in seg])
            sm = prep(d.decode(module, m1.encode_trajectory(module, pts), pts[0]), pts[0])
            md, rd = geo.stats(sm)
            entries.append(dict(task=t, cell=list(c), route_id=route_id, away=cell['away'],
                                points_xy=pts.tolist(), max_occupancy_depth=md,
                                max_true_wall_depth=rd, clean=md <= TAU))
    mx = np.asarray([e['max_occupancy_depth'] for e in entries])
    if not len(mx):
        raise ValueError('No positive control routes')
    named = [e for e in entries if (e['task'], tuple(e['cell'])) in
             {('2', (5, 6)), ('2', (5, 1)), ('4', (3, 9))}]
    return dict(n=len(entries), tau=TAU, clean_count=sum(e['clean'] for e in entries),
                clean_rate=float(np.mean(mx <= TAU)), p50=float(np.median(mx)),
                p95=float(np.percentile(mx, 95)), max=float(mx.max()),
                named=named, entries=entries)


def straight_reconstruction(module, geo):
    """First 2048 straight windows; turns from every sampled batch (descriptive)."""
    torch = module.torch
    rng = np.random.default_rng(SEED)
    straight, turn = [], []
    coord_hash = hashlib.sha256()
    batches = 0
    while len(straight) < 2048:
        batches += 1
        if batches > 100000:
            raise RuntimeError('2048 straight trajectories not reached within 100000 batches')
        traj, mask, s, _, _ = module.make_batch(rng, teacher_mix=0.0)
        norm = traj.cpu().numpy()
        raw = norm.astype(np.float64)*geo.sd+geo.mu
        si, ti = [], []
        for i, xy in enumerate(raw):
            cells = [geo.xy2c(p) for p in xy]
            cells = [c for k, c in enumerate(cells) if k == 0 or c != cells[k-1]]
            aligned = all(c[0] == cells[0][0] for c in cells) or all(c[1] == cells[0][1] for c in cells)
            if len(set(cells)) >= 3 and aligned and len(straight)+len(si) < 2048:
                si.append(i)
                # Ordered world-coordinate float64 little-endian bytes, no headers.
                coord_hash.update(np.ascontiguousarray(xy, dtype='<f8').tobytes())
            elif not aligned:
                ti.append(i)
        indices = si+ti
        if indices:
            with torch.no_grad():
                ix = torch.tensor(indices, dtype=torch.long)
                err = (module._dec(module.etarget(traj[ix], mask[ix]), s[ix])-traj[ix]).pow(2)
                mse = err.mean(dim=(1, 2)).cpu().numpy()
            straight.extend(float(x) for x in mse[:len(si)])
            turn.extend(float(x) for x in mse[len(si):])
    if not turn:
        raise ValueError('No turning trajectories sampled; descriptive MSE unavailable')
    return dict(n=len(straight), mse=float(np.mean(straight)),
                coordinate_sha256=coord_hash.hexdigest(), coordinate_bytes='world XY <f8 C order [2048,128,2]',
                per_trajectory_mse=straight, batches=batches, batch_size=int(module.B), seed=SEED,
                turn_n=len(turn), turn_mse=float(np.mean(turn)),
                turn_definition='grid sequence changes both row and column; all sampled batches',
                definition='make_batch world XY, consecutive grid duplicates removed; >=3 distinct cells, one row or column')


def compute(args):
    from eval_common import load_ckpt
    module = load_ckpt(str(args.dataset), str(args.ckpt), args.ckpt_sha256)
    if str(module.device) != 'cpu':
        raise ValueError('CPU required')
    wrapped = module.ogbench.make_env_and_datasets(ENV, env_only=True)
    try:
        geo = Geometry(module.OBS_XY, module.MU_XY, module.SD_XY, wrapped.unwrapped)
        stageo = json.loads(args.cells.read_text())
        routes = json.loads(args.routes.read_text())
        out = dict(schema='wallpen-model-v1', device='cpu', provenance=module.probe_provenance,
                   inventory_sha256=sha_file(args.cells), routes_sha256=sha_file(args.routes),
                   positive_control=positive_controls(module, geo, stageo, routes),
                   straight=straight_reconstruction(module, geo))
        write_json(args.out, out)
        print(json.dumps(dict(out=str(args.out), positive_n=out['positive_control']['n'],
                              positive_p95=out['positive_control']['p95'],
                              straight={k: v for k, v in out['straight'].items() if k != 'per_trajectory_mse'}),
                         ensure_ascii=False), flush=True)
    finally:
        wrapped.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--ckpt-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--cells', type=Path, default=BACKGROUND/'hsweep/stageO/H3.json')
    p.add_argument('--routes', type=Path, default=BACKGROUND/'detour-u/routes-density.json')
    args = p.parse_args()
    try:
        compute(args)
    except Exception as e:
        p.exit(2, f'BLOCKED: {type(e).__name__}: {e}\n')


if __name__ == '__main__':
    main()
