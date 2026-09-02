"""收開頭綁定批：dstart_hard（v4）/ dstart_soft 對 V8 同八顆；含 rt_gate 往返尺、flow_probe 進度、逐題失敗。"""
import json,glob,os,re,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
V8={20:.696,21:.808,22:.728,23:.492,24:.724,25:.512,26:.492,27:.872}
def pt(p):
    dg=p.replace('rollout_','diag_')
    if not os.path.exists(dg): return None
    eps=[e for e in json.load(open(dg)) if e['arm'].startswith('分段 conf2')]
    c=collections.defaultdict(int)
    for e in eps: c[e['task']]+=(0 if e['success'] else 1)
    return [c[t] for t in range(1,6)]
for name,d in (('hard(v4)','dstart_hard'),('soft','dstart_soft')):
    rows=[]
    print(f"== 開頭綁定 {name}（對 V8 同 seed；雜訊線 ±0.03）==")
    for p in sorted(glob.glob(f'results/night_0902/{d}/rollout_*.json'),key=lambda x:int(re.search(r'_s(\d+)\.json$',x).group(1))):
        S=int(re.search(r'_s(\d+)\.json$',p).group(1)); j=json.load(open(p)); r=j['rates']; g=j.get('rt_gate'); fp=j.get('flow_probe')
        gm=np.mean([x['mse'] for x in g if x]) if g else float('nan'); pm=np.mean([x['prog_med'] for x in fp if x]) if fp else float('nan')
        rows.append((S,r['subgoal'])); print(f"s{S}: 主打 {r['subgoal']:.3f}（V8 {V8[S]:.3f} Δ{r['subgoal']-V8[S]:+.3f}） bc {r['bc']:.3f} 往返mse {gm:.4f} 進度中位 {pm:.2f} 逐題失敗 {pt(p)}")
    if rows:
        v=[x[1] for x in rows]; ref=[V8[x[0]] for x in rows]
        print(f"   n={len(v)} 平均 {np.mean(v):.3f} sd {np.std(v,ddof=1) if len(v)>1 else 0:.3f} 範圍 {min(v):.3f}~{max(v):.3f} 爛顆(<.3) {sum(x<.3 for x in v)} ｜ V8 同 seed 平均 {np.mean(ref):.3f} sd {np.std(ref,ddof=1) if len(ref)>1 else 0:.3f}")
    else: print("   尚無")
