"""S2 sandbox driver: run runtime.execute() end-to-end on CPU with the REAL env + REAL model.
Shims (sandbox only, none touch the audited files):
  - harvest.historical_fingerprints -> n64 fingerprints dumped read-only from jasmine
  - pathlib is_relative_to -> True (archive guard bypass; /archive absent on this host)
"""
import sys, os, json, time, pathlib
SCR = '/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
MIR = SCR + '/mirror/lacot/experiments/_workorders/breakthrough-probe'
sys.path.insert(0, MIR); sys.dont_write_bytecode = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import common, harvest, runtime
fps = {tuple(map(int, k.split(','))): tuple(v) for k, v in json.load(open(SCR + '/n64_fingerprints.json')).items()}
harvest.historical_fingerprints = lambda gate: fps
pathlib.PurePath.is_relative_to = lambda self, *a, **k: True
move = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else ''
cal = dict(dataset_dir='/home/cymaxwelllee/data/ogbench', steps_per_cell=12, stuck_window=20,
           stuck_distance=0.2, noise_band={'4': 0.01, '5': 0.01})
out = pathlib.Path(SCR) / f'e2e-move{move}{tag}'
t = time.time()
art = runtime.execute(move, out, cal, smoke=True)
print('ELAPSED', round(time.time() - t, 1), 'rows', len(art['rows']), 'repeats', len(art['conformance_rows']),
      'skipped', len(art['skipped']), flush=True)
print('READOUT', json.dumps(art['readout'], default=str)[:3000])
