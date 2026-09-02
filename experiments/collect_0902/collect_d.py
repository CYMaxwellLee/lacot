"""收 D 批（定版配方 warmup500+EMA、無 boot）：D1 固定 init=s23 換資料順序；D2 固定資料順序=23 換 init。對照：V8 s23 原顆 0.492/0.488。"""
import json,glob,os,re,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
print("== D 批：定版配方（wu500+EMA、無 boot）方差拆分；雜訊線 ±0.03 ==")
rows=collections.defaultdict(list)
for d in sorted(glob.glob('results/night_0902/varsrc2_*')):
    name=d.split('varsrc2_')[1]; ro=glob.glob(d+'/rollout_*.json'); dg=glob.glob(d+'/diag_*.json')
    if not ro: print(f"{name}: (尚無)"); continue
    r=json.load(open(ro[0]))['rates']; rows[name[:2]].append(r['subgoal'])
    line=f"{name:<8} 主打 {r['subgoal']:.3f}  bc {r['bc']:.3f}  null_u {r['null_u']:.3f}"
    if dg:
        eps=[e for e in json.load(open(dg[0])) if e['arm'].startswith('分段 conf2')]
        c=collections.defaultdict(int)
        for e in eps: c[e['task']]+=(0 if e['success'] else 1)
        line+=f"   逐題失敗 {[c[t] for t in sorted(c)]}"
    print(line)
for g,v in rows.items():
    if len(v)>1: print(f"{g}: n={len(v)} mean={np.mean(v):.3f} sd={np.std(v,ddof=1):.3f} range={min(v):.3f}~{max(v):.3f}")
print("對照 V8 s23（init 23、資料順序 23）：0.492 / 重評 0.488。判讀：D1 散得開＝資料順序主導；D2 散得開＝init 主導；兩者都散＝任何隨機都行（同 C 批形狀）")
