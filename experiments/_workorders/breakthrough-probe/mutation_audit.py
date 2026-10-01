"""Re-run archived S2 survivors in isolated temporary mirrors; never mutate live code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from common import HERE, REPO, write_json

TARGETS = {'S5', 'H3', 'H13', 'H15', 'H18', 'H19', 'H20',
           'X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X14'}


def run(entry):
    identity, filename, old, new, description = entry
    with tempfile.TemporaryDirectory(prefix='probe-mutation-') as tmp:
        repo = Path(tmp)/'lacot'
        dest = repo/'experiments/_workorders/breakthrough-probe'
        shutil.copytree(HERE, dest, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('lacot', 'results', '.venv'):
            (repo/name).symlink_to(REPO/name, target_is_directory=True)
        (repo/'experiments/_workorders/ucontrast1').symlink_to(REPO/'experiments/_workorders/ucontrast1', target_is_directory=True)
        path = dest/filename
        source = path.read_text()
        if source.count(old) != 1 and identity != 'S3':
            return dict(id=identity, status='STALE_PATTERN', matches=source.count(old), description=description)
        path.write_text(source.replace(old,new,1))
        run = subprocess.run([str(REPO/'.venv/bin/python'), '-B', str(dest/'selftest.py')],
                             capture_output=True,text=True,timeout=180,
                             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
        output = run.stdout+run.stderr
        failures = sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+)',output,re.M)))
        status = ('SURVIVED' if run.returncode == 0 else 'KILLED' if failures else 'CRASH')
        return dict(id=identity,status=status,description=description,tests=failures,
                    returncode=run.returncode, diagnostic=output[-1000:] if status=='CRASH' else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-survivors',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    catalog=json.loads((HERE/'s2-surviving-mutations.json').read_text())
    selected=[e for e in catalog if args.all_survivors or e[0] in TARGETS or e[0]=='S3']
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run,selected):
            print(result['id'],result['status'],','.join(result.get('tests',[])),flush=True)
            results.append(result)
    write_json(args.out,dict(targets=sorted(TARGETS),results=results))
    ok=all(r['status']==('SURVIVED' if r['id']=='S3' else 'KILLED') for r in results if r['id'] in TARGETS or r['id']=='S3')
    return 0 if ok else 1


if __name__=='__main__':
    sys.exit(main())
