"""收 9/2 過夜替代臂：B guard_{HG,DA,HGS} 八顆、C' dssoft_HG、E' heldout2_{base,soft}、F' medium_base。讀法同 collect_overnight.py（rates.subgoal）。"""
import json,glob,os,re
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
V8={20:.696,21:.808,22:.728,23:.492,24:.724,25:.512,26:.492,27:.872}
def rates(d):
    out={}
    for p in glob.glob(f'results/night_0902/{d}/rollout_*.json'):
        S=int(re.search(r'_s(\d+)\.json$',p).group(1)); out[S]=json.load(open(p))['rates']['subgoal']
    return out
def summ(name,r,ref=None,refname='對照'):
    if not r: print(f"== {name}: 尚無"); return
    ks=sorted(r); v=[r[k] for k in ks]
    line=f"== {name}: "+" ".join(f"s{k} {r[k]:.3f}" for k in ks)+f" ｜ n={len(v)} 平均 {np.mean(v):.3f} sd {np.std(v,ddof=1) if len(v)>1 else 0:.3f} 爛顆 {sum(x<.3 for x in v)}"
    if ref:
        common=[k for k in ks if k in ref]
        if common: line+=f" ｜ {refname}同 seed 平均 {np.mean([ref[k] for k in common]):.3f}（Δ {np.mean([r[k]-ref[k] for k in common]):+.3f}）"
    print(line)
soft=rates('dstart_soft'); base2=rates('heldout2_base')
print("--- B 守門/錨定 八顆（HG/HGS 是安全網數字、非 latent 計畫數字）---")
summ('HG 守門（V8 ckpt）',rates('guard_HG'),V8,'V8')
summ('DA 純錨定（V8 ckpt）',rates('guard_DA'),V8,'V8')
summ('HGS 守門+吸附',rates('guard_HGS'),V8,'V8')
print("--- C' soft ckpt × HG（守門疊在綁定上還開不開火）---")
summ('CSHG soft×守門',rates('dssoft_HG'),soft,'soft')
for f in sorted(glob.glob('slurm/logs/CSHG-s*-*.out')):
    m=re.findall(r'n_headguard[^\d]*(\d+)', open(f,errors='ignore').read())
    if m: print(f"   {os.path.basename(f)[:8]} headguard 開火 {m[-1]}")
print("--- E' 新 held-out s28~35（第二組獨立 seed）---")
summ('E2 原配方',base2)
summ('E2 soft',rates('heldout2_soft'),base2,'base')
print("--- F' medium-stitch 原配方八顆 ---")
summ('F medium_base',rates('medium_base'))
print("對照：V8 large 八顆 .665 sd .149；soft 八顆 .752 sd .168；medium 現有 2~3 顆 .784/.920/.788")
