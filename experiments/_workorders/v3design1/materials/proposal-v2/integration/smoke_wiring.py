"""u 編碼想像軌跡。Four physically detached exams and shared composition."""
from pathlib import Path
import tempfile,subprocess,shutil,hashlib,sys,os
ROOT=Path(__file__).resolve().parents[1]
PY='/home/cymaxwelllee/Projects/lacot/.venv/bin/python3'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ('objective','actor','oracle','verification'):
    module=ROOT/f'mod-{name}'
    for f in (ROOT/'contracts').glob('*'):
        if f.is_file(): assert f.read_bytes()==(module/'contracts'/f.name).read_bytes()
    with tempfile.TemporaryDirectory(prefix='refine-v2-') as tmp:
        dst=Path(tmp)/module.name;shutil.copytree(module,dst)
        proc=subprocess.run([PY,'-B',str(dst/'tests.py'),'--self-check'],cwd=tmp,capture_output=True,text=True)
        print(proc.stdout,end='');assert proc.returncode==0,proc.stderr
        if name=='verification':
            proc=subprocess.run([PY,'-B',str(dst/'integration/mutations.py')],cwd=tmp,capture_output=True,text=True)
            print(proc.stdout,end='');assert proc.returncode==0,proc.stderr
        # A stub must fail, not silently fall back to fakes.
        stub=subprocess.run([PY,'-B',str(dst/'tests.py'),'--candidate',str(dst/'solution.py')],cwd=tmp,capture_output=True,text=True)
        assert stub.returncode!=0 and 'NotImplementedError' in stub.stderr
    print(f'PASS independent mod-{name}: identical contract, /tmp relocation, stub fails closed')
sys.path.insert(0,str(ROOT/'integration'))
from toy import *
c,cl,a=batch();sample=cl.clone();sample[...,1]+=.8
r=TinyRefine();teacher=copy.deepcopy(r).requires_grad_(False);h=TinyHead()
value,logs,states=ref_loss(r,teacher,h,c,cl,a,sample,3,torch.ones_like(sample)*.08)
assert torch.isfinite(value) and len(states)==4 and logs['l_plan_quality']>0
q=Quality(sample,c);g=gauge(h,states,cl,a,q)
assert g['exposure/valid'] and 'exposure/r3_gap' in g
print('PASS wiring M4 oracle -> M1 objective -> M2 composition -> M3 observed gauge')
print('PASS smoke_wiring: four zero-wait exams; reference wiring only, not production repair')
