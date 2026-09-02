"""往返尺校準：V8 八顆的 rt_gate（gate_v8/）對它們的官方成功率（final8/）。尺有效＝往返 mse／穿牆 與成功率呈負相關。"""
import json,glob,os,re
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
V8={20:.696,21:.808,22:.728,23:.492,24:.724,25:.512,26:.492,27:.872}
rows=[]
for p in sorted(glob.glob('results/night_0902/gate_v8/rollout_*.json'),key=lambda x:int(re.search(r'_s(\d+)\.json$',x).group(1))):
    S=int(re.search(r'_s(\d+)\.json$',p).group(1)); g=json.load(open(p)).get('rt_gate')
    if not g: print(f"s{S}: 無 rt_gate"); continue
    m=[x['mse'] for x in g if x]; w=[x['wall'] for x in g if x]; d=[x['gdist'] for x in g if x]
    rows.append((S,V8[S],np.mean(m),np.max(m),np.mean(w),np.mean(d)))
    print(f"s{S}: 官方 {V8[S]:.3f} | 往返 mse 平均 {np.mean(m):.4f} 最差題 {np.max(m):.4f} | 穿牆 {np.mean(w):.3f} | 末點距 {np.mean(d):.2f} | 逐題 mse {[round(x,3) for x in m]}")
if len(rows)>=4:
    y=[r[1] for r in rows]
    for k,name in ((2,'往返 mse 平均'),(3,'往返 mse 最差題'),(4,'穿牆'),(5,'末點距')):
        x=[r[k] for r in rows]; print(f"相關 {name} vs 官方成功率: r={np.corrcoef(x,y)[0,1]:+.2f}")
