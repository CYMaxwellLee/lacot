"""u 編碼想像軌跡。Standalone exam; self-check certifies fixtures only."""
import sys,os,argparse,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'contracts'))
from conformance import objective_suite,composition_suite,oracle_suite
from acceptance import suite
p=argparse.ArgumentParser();p.add_argument('--candidate');p.add_argument('--self-check',action='store_true');a=p.parse_args()
if a.candidate:
    spec=importlib.util.spec_from_file_location('candidate',a.candidate);candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    print('WORKER_CANDIDATE (no fallback)')
else:
    import acceptance as candidate
    print('EXAM_SELF_CHECK (fixtures only, no worker completion)')
suite(candidate.assess_run)

if a.candidate:
    import tempfile
    with tempfile.TemporaryDirectory(prefix='m3-audit-') as tmp:
        report=candidate.scientific_audit('/home/cymaxwelllee/Projects/lacot',tmp)
        assert report['current_mainline_exit']==1
        assert report['mutation_kills']==6
        assert report['identity_rejected'] and report['actual_flow_samples']
        assert set(report['trained_seeds'])=={0,1,2}
        assert all(Path(tmp,p).is_file() for p in report['artifacts'])
        assert len(report['artifacts'])>=6
