"""收 embedding 批：A 長 stage1（emb_s16000/）、B VQ64（emb_vq64/）對 V8（final8/ 與 v8diag/）。含 rt_gate 往返尺。"""
import json,glob,os,re
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
V8={20:.696,21:.808,22:.728,23:.492,24:.724,25:.512,26:.492,27:.872}
def load_dir(d):
    out={}
    for p in glob.glob(f'results/night_0902/{d}/rollout_*.json'):
        S=int(re.search(r'_s(\d+)\.json$',p).group(1)); j=json.load(open(p)); out[S]=j
    return out
def summarize(name,dd):
    print(f"== {name}（n={len(dd)}）==")
    rows=[]
    for S in sorted(dd):
        j=dd[S]; r=j['rates']; g=j.get('rt_gate')
        gm=np.mean([x['mse'] for x in g if x]) if g else float('nan'); gw=np.mean([x['wall'] for x in g if x]) if g else float('nan'); gd=np.mean([x['gdist'] for x in g if x]) if g else float('nan')
        rows.append((S,r['subgoal'],r['bc'],gm,gw,gd))
        print(f"s{S}: 主打 {r['subgoal']:.3f} (V8 {V8[S]:.3f}, Δ{r['subgoal']-V8[S]:+.3f})  bc {r['bc']:.3f}  往返 mse {gm:.4f} 穿牆 {gw:.3f} 末點距 {gd:.2f}")
    if len(rows)>1:
        v=[x[1] for x in rows]; print(f"   平均 {np.mean(v):.3f}  sd {np.std(v,ddof=1):.3f}  範圍 {min(v):.3f}~{max(v):.3f}  爛顆(<0.3) {sum(x<0.3 for x in v)}   ｜ V8 同 seed 平均 {np.mean([V8[x[0]] for x in rows]):.3f} sd {np.std([V8[x[0]] for x in rows],ddof=1):.3f}")
        gm=[x[3] for x in rows]; sg=[x[1] for x in rows]
        if all(np.isfinite(gm)): print(f"   往返 mse 與主打的相關 r={np.corrcoef(gm,sg)[0,1]:.2f}（負＝往返差的顆成績低＝尺有效）")
for name,d in (('A 長 stage1 6000','emb_s16000'),('B VQ64','emb_vq64')):
    dd=load_dir(d); summarize(name,dd) if dd else print(f"== {name}: 尚無 ==")
