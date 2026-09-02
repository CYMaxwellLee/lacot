"""收 E 選計畫（esel16）：同顆 ckpt，conf2（M=4 共識）vs conf2+esel16（抽 16 用 E 留 4 再共識）；逐題失敗與死點地理。"""
import json,glob,os,re,sys,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
sys.argv=['x']
exec(open('experiments/collect_0902/collect_0902.py').read().split("mode=sys.argv[1]")[0])
print("== E 選計畫：conf2 vs conf2+esel16（同顆 ckpt、EMA、官方 250 集；雜訊線 ±0.03）==")
base={}; new={}
for S in (23,25,26,27):
    a=glob.glob(f'results/night_0902/v8diag/rollout_*_s{S}.json'); b=glob.glob(f'results/night_0902/esel16/rollout_*_s{S}.json')
    ra=json.load(open(a[0]))['rates']['subgoal'] if a else None; rb=json.load(open(b[0]))['rates']['subgoal'] if b else None
    base[S]=ra; new[S]=rb
    print(f"s{S}: conf2 {ra if ra is None else f'{ra:.3f}'}  →  esel16 {rb if rb is None else f'{rb:.3f}'}   Δ {'' if (ra is None or rb is None) else f'{rb-ra:+.3f}'}")
vals=[(base[S],new[S]) for S in base if base[S] is not None and new[S] is not None]
if vals: print(f"平均 conf2 {np.mean([v[0] for v in vals]):.3f} → esel16 {np.mean([v[1] for v in vals]):.3f}  (n={len(vals)})")
files=glob.glob('results/night_0902/esel16/diag_*.json')
if files: report_A(files,lambda p:'s'+re.search(r'_s(\d+)\.json$',p).group(1)+'@esel')
