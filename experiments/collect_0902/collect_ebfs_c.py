"""收 ebfs 分辨器（v8diag 的 conf2 vs ebfs 的餵正確路標）與 C 批（C1 只動 boot 抽樣 / C2 只動主資料順序）。"""
import json,glob,os,re,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
def rate(p,key='subgoal'):
    d=json.load(open(p)); r=d['rates']; return r.get(key, r)
def per_task(p,arm_prefix):
    d=json.load(open(p)); eps=[e for e in d if e['arm'].startswith(arm_prefix)]
    if not eps: return None
    c=collections.defaultdict(int)
    for e in eps: c[e['task']]+= (0 if e['success'] else 1)
    return {t:c[t] for t in sorted(c)}
print("== ebfs 分辨器：同一顆 ckpt，計畫路標（conf2）vs 正確路標（ebfs）==")
for S in (23,25,26,27):
    a=glob.glob(f'results/night_0902/v8diag/rollout_*_s{S}.json'); b=glob.glob(f'results/night_0902/ebfs/rollout_*_s{S}.json')
    ca=rate(a[0]) if a else None; cb=rate(b[0]) if b else None
    dg=glob.glob(f'results/night_0902/ebfs/diag_*_s{S}.json')
    arms=sorted({e['arm'] for e in json.load(open(dg[0]))}) if dg else []
    ebarm=[x for x in arms if 'ebfs' in x.lower() or 'E 圖' in x or '供點' in x]
    pt=per_task(dg[0],ebarm[0]) if dg and ebarm else None
    print(f"s{S}: conf2 {ca if ca is None else f'{ca:.3f}'}   ebfs {cb if cb is None else (f'{cb:.3f}' if isinstance(cb,float) else cb)}   ebfs 逐題失敗 {pt}   (arms: {[x[:14] for x in arms]})")
print("\n== C 批：init s2、boot_s2_dz2 不動 ==")
rows=collections.defaultdict(list)
for d in sorted(glob.glob('results/night_0902/varsrc_*')):
    name=d.split('varsrc_')[1]; ro=glob.glob(d+'/rollout_*_s2.json')
    if not ro: print(f"{name}: (尚無)"); continue
    r=json.load(open(ro[0]))['rates']; grp=name[:2]; rows[grp].append(r['subgoal'])
    print(f"{name:<8} 主打 {r['subgoal']:.3f}  bc {r['bc']:.3f}")
for g,v in rows.items():
    if len(v)>1: print(f"{g}: n={len(v)} mean={np.mean(v):.3f} sd={np.std(v,ddof=1):.3f} range={min(v):.3f}~{max(v):.3f}")
print("判讀：C1（只動 boot 抽樣）散得開＝boot 抽樣是病根；C2（只動主資料順序）散得開＝主資料順序是病根；雜訊線 ±0.03")
