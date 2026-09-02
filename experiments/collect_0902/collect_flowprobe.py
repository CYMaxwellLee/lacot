"""flow 探針校準：V8 八顆的 flow_probe（每題對路率／路徑距／穿牆／分散）對官方成功率與逐題失敗（v8diag 有 s23/25/26/27）。"""
import json,glob,os,re,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
V8={20:.696,21:.808,22:.728,23:.492,24:.724,25:.512,26:.492,27:.872}
pertask={}
for S in (23,25,26,27):
    dg=glob.glob(f'results/night_0902/v8diag/diag_*_s{S}.json')
    if dg:
        eps=[e for e in json.load(open(dg[0])) if e['arm'].startswith('分段 conf2')]
        c=collections.defaultdict(int)
        for e in eps: c[e['task']]+=(1 if e['success'] else 0)
        pertask[S]=[c[t]/50 for t in range(1,6)]
rows=[]
for p in sorted(glob.glob('results/night_0902/flowprobe_v8/rollout_*.json'),key=lambda x:int(re.search(r'_s(\d+)\.json$',x).group(1))):
    S=int(re.search(r'_s(\d+)\.json$',p).group(1)); fp=json.load(open(p)).get('flow_probe')
    if not fp: print(f"s{S}: 無 flow_probe"); continue
    onr=[r['onroute'] if r else float('nan') for r in fp]; rd=[r['route_d_med'] if r else float('nan') for r in fp]; wl=[r['wall_med'] if r else float('nan') for r in fp]; sp=[r['head_end_spread'] if r else float('nan') for r in fp]
    rows.append((S,V8[S],np.nanmean(onr),np.nanmean(rd),np.nanmean(wl),np.nanmean(sp)))
    line=f"s{S}: 官方 {V8[S]:.3f} | 對路率 逐題 {[round(x,2) for x in onr]} 平均 {np.nanmean(onr):.2f} | 路徑距中位 {[round(x,2) for x in rd]} | 穿牆 {[round(x,3) for x in wl]} | 分散 {[round(x,1) for x in sp]}"
    if S in pertask: line+=f" | 逐題官方成功率 {[round(x,2) for x in pertask[S]]}"
    print(line)
if len(rows)>=4:
    y=[r[1] for r in rows]
    for k,name in ((2,'對路率平均'),(3,'路徑距中位平均'),(4,'穿牆平均'),(5,'開頭段終點分散')):
        x=[r[k] for r in rows]; print(f"相關 {name} vs 官方成功率: r={np.corrcoef(x,y)[0,1]:+.2f}")
# 逐題層級：對路率 vs 逐題成功率（四顆 × 五題 = 20 點）
xs=[];ys=[]
for p in glob.glob('results/night_0902/flowprobe_v8/rollout_*.json'):
    S=int(re.search(r'_s(\d+)\.json$',p).group(1))
    if S in pertask:
        fp=json.load(open(p)).get('flow_probe') or []
        for t,r in enumerate(fp):
            if r: xs.append(r['onroute']); ys.append(pertask[S][t])
if len(xs)>=8: print(f"逐題層級（{len(xs)} 點）對路率 vs 逐題成功率: r={np.corrcoef(xs,ys)[0,1]:+.2f}")
