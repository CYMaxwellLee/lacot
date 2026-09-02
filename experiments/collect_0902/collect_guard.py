"""收守門批：conf2（v8diag）/ snap（snap_ma2）/ HG（guard_HG）/ DA（guard_DA）/ HGS（guard_HGS），同顆 ckpt。"""
import json,glob,os,re,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
ARMS=[('conf2','v8diag'),('snap','snap_ma2'),('HG','guard_HG'),('DA','guard_DA'),('HGS','guard_HGS')]
def rate(d,S):
    p=glob.glob(f'results/night_0902/{d}/rollout_*_s{S}.json'); return json.load(open(p[0]))['rates']['subgoal'] if p else None
def pt(d,S):
    p=glob.glob(f'results/night_0902/{d}/diag_*_s{S}.json')
    if not p: return None
    eps=[e for e in json.load(open(p[0])) if e['arm'].startswith('分段 conf2')]
    c=collections.defaultdict(int)
    for e in eps: c[e['task']]+=(0 if e['success'] else 1)
    return [c[t] for t in range(1,6)]
print("== 守門批（同顆 ckpt、官方 250 集；雜訊線 ±0.03）==")
f=lambda x:'  —  ' if x is None else f'{x:.3f}'
tab={}
for S in (23,25,26,27):
    row=[rate(d,S) for _,d in ARMS]; tab[S]=row
    print(f"s{S}: "+"  ".join(f"{n} {f(v)}" for (n,_),v in zip(ARMS,row)))
    print("      逐題失敗 "+"  ".join(f"{n} {pt(d,S)}" for n,d in ARMS if pt(d,S) is not None))
for k,(n,_) in enumerate(ARMS):
    v=[tab[S][k] for S in tab if tab[S][k] is not None]
    if v: print(f"{n}: n={len(v)} 平均 {np.mean(v):.3f}"+(f"（對 conf2 同顆 Δ {np.mean([tab[S][k]-tab[S][0] for S in tab if tab[S][k] is not None and tab[S][0] is not None]):+.3f}）" if k>0 else ""))
# headguard 觸發次數（log）
for A in ('HG','HGS'):
    for S in (23,25,26,27):
        fs=glob.glob(f'slurm/logs/{A}-s{S}-*.out')
        if fs:
            t=open(fs[-1]).read(); m=re.findall(r'n_headguard[^\d]*(\d+)',t)
            if m: print(f"{A}-s{S} headguard 觸發 {m[-1]}")
