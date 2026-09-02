"""收過夜 C/E/F：dshard_{HG,DA}（hard ckpt × 守門/純錨定；含 n_headguard 開火數）、heldout2_{base,hard}（s28~35）、medium_hard。"""
import json,glob,os,re,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
def rates(d):
    out={}
    for p in glob.glob(f'results/night_0902/{d}/rollout_*.json'):
        S=int(re.search(r'_s(\d+)\.json$',p).group(1)); out[S]=json.load(open(p))['rates']['subgoal']
    return out
def summ(name,r,ref=None):
    if not r: print(f"== {name}: 尚無"); return
    ks=sorted(r); v=[r[k] for k in ks]
    line=f"== {name}: "+" ".join(f"s{k} {r[k]:.3f}" for k in ks)+f" ｜ n={len(v)} 平均 {np.mean(v):.3f} sd {np.std(v,ddof=1) if len(v)>1 else 0:.3f} 爛顆 {sum(x<.3 for x in v)}"
    if ref: 
        common=[k for k in ks if k in ref]; 
        if common: line+=f" ｜ 對照同 seed 平均 {np.mean([ref[k] for k in common]):.3f}（Δ {np.mean([r[k]-ref[k] for k in common]):+.3f}）"
    print(line)
hard=rates('dstart_hard'); base=rates('heldout2_base')
summ('C hard ckpt × 守門 HG',rates('dshard_HG'),hard)
summ('C hard ckpt × 純錨定 DA',rates('dshard_DA'),hard)
for A in ('CHG','CDA'):
    for f in sorted(glob.glob(f'slurm/logs/{A}-s*-*.out')):
        m=re.findall(r'n_headguard[^\d]*(\d+)', open(f,errors='ignore').read())
        if m: print(f"   {os.path.basename(f)[:10]} headguard 開火 {m[-1]}")
summ('E 新 held-out s28~35 原配方',base)
summ('E 新 held-out s28~35 hard(v4)',rates('heldout2_hard'),base)
summ('F medium-stitch hard(v4)',rates('medium_hard'))
print("對照：V8 large 八顆 .665 sd .149；medium 現有 K8 2~3 顆 .784/.920/.788")
