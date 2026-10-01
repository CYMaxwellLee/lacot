"""Run Move 1 smoke e2e under an audit hook to list every path opened for writing / created."""
import sys, os, json, pathlib
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
writes=set()
def hook(event,args):
    try:
        if event=='open':
            path,mode,flags=args
            if isinstance(path,(str,bytes,os.PathLike)):
                p=os.fsdecode(path)
                m=mode or ''
                if any(c in str(m) for c in 'wax+') or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT)):
                    writes.add(('open',p))
        elif event in ('os.mkdir','os.rename','os.remove','os.rmdir','os.symlink','os.link','shutil.copyfile','shutil.move'):
            writes.add((event,tuple(os.fsdecode(a) if isinstance(a,(str,bytes,os.PathLike)) else str(a) for a in args[:2])))
    except Exception: pass
sys.addaudithook(hook)
MIR=SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'
sys.path.insert(0,MIR); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl', PYTHONDONTWRITEBYTECODE='1')
import tempfile
tmp=tempfile.mkdtemp(prefix='audit-tmp-',dir=SCR); os.environ['TMPDIR']=tmp; tempfile.tempdir=tmp
os.environ['MPLCONFIGDIR']=tmp+'/mpl'
import common, harvest, runtime
fps={tuple(map(int,k.split(','))):tuple(v) for k,v in json.load(open(SCR+'/n64_fingerprints.json')).items()}
harvest.historical_fingerprints=lambda gate: fps
pathlib.PurePath.is_relative_to=lambda self,*a,**k: True
cal=dict(dataset_dir='/home/cymaxwelllee/data/ogbench', steps_per_cell=12, stuck_window=20, stuck_distance=0.2, noise_band={'4':0.01,'5':0.01})
out=pathlib.Path(SCR)/'e2e-move1-audit'
runtime.execute(1,out,cal,smoke=True)
for w in sorted(writes, key=str):
    p=w[1] if w[0]=='open' else str(w[1])
    if '/proc/' in p or p.startswith('/dev/'): continue
    print('WRITE',w[0],p)
