"""收路標吸附批：conf2（v8diag）vs snap+ma2 vs snap+ma1，同顆 ckpt；逐題失敗與死點口袋比。"""
import json,glob,os,re,collections
import numpy as np
from _collect_common import unique_glob, same_run, task_failures
os.chdir(os.path.expanduser('~/Projects/lacot'))
def rate(d,S):
    p=unique_glob(f'results/night_0902/{d}/rollout_*_s{S}.json'); return json.load(open(p))['rates']['subgoal'] if p else None
def pt(d,S):
    p=unique_glob(f'results/night_0902/{d}/diag_*_s{S}.json')
    if not p: return None
    same_run(unique_glob(f'results/night_0902/{d}/rollout_*_s{S}.json'),p)
    eps=[e for e in json.load(open(p)) if e['arm'].startswith('分段 conf2')]
    return task_failures(eps)
print("== 路標吸附：同顆 ckpt、官方 250 集（雜訊線 ±0.03）==")
f=lambda x: '  —  ' if x is None else f'{x:.3f}'
rows=[]
for S in (23,25,26,27):
    a,b,c=rate('v8diag',S),rate('snap_ma2',S),rate('snap_ma1',S)
    rows.append((a,b,c))
    print(f"s{S}: conf2 {f(a)}   snap {f(b)}   snap+ma1 {f(c)}   | 逐題失敗 conf2 {pt('v8diag',S)}  snap {pt('snap_ma2',S)}  snap+ma1 {pt('snap_ma1',S)}")
for k,name in ((1,'snap'),(2,'snap+ma1')):
    v=[(r[0],r[k]) for r in rows if r[0] is not None and r[k] is not None]
    if v: print(f"{name}: n={len(v)}  conf2 平均 {np.mean([x[0] for x in v]):.3f} → {np.mean([x[1] for x in v]):.3f}  （Δ {np.mean([x[1]-x[0] for x in v]):+.3f}）")
