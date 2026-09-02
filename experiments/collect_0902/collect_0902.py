"""收 9/2 兩批診斷：A) V8 失敗地理（v8diag/）B) dz2 重現（dz2rep_d*/）。
用法：python collect_0902.py [A|B|both] [--test]   (--test 拿昨晚 dz2/dz3 的 diag 當兩顆試跑)
座標：diag 的 start/goal/final 是【原始】座標（x∈[-1,37], y∈[-1,25]）；資料密度也在原始座標算。"""
import json,glob,os,sys,re,collections,itertools
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
ARM='分段 conf2'; BCARM='誠實 BC'
BIN=2.0; XR=(-3.0,39.0); YR=(-3.0,27.0)           # 原始座標、2 單位細格
def load(p): return json.load(open(p))
def per_task(eps):
    c=collections.defaultdict(lambda:[0,0])
    for e in eps: c[e['task']][1]+=1; c[e['task']][0]+=(0 if e['success'] else 1)
    tot=sum(b for a,b in c.values()); fail=sum(a for a,b in c.values())
    return 1-fail/tot, {t:a for t,(a,b) in sorted(c.items())}
_DEN=None
def density():
    global _DEN
    if _DEN is None:
        o=np.load(os.path.expanduser('~/data/ogbench/pointmaze-large-stitch-v0.npz'))['observations'][:,:2]
        H,xe,ye=np.histogram2d(o[:,0],o[:,1],bins=[int((XR[1]-XR[0])/BIN),int((YR[1]-YR[0])/BIN)],range=[XR,YR])
        _DEN=(H,xe,ye)
    return _DEN
def den_pct(xy):
    """該點所在細格的資料密度，換成「在所有有資料格中的分位數」（低＝荒漠）"""
    H,xe,ye=density(); vis=H[H>0]
    i=np.clip(np.searchsorted(xe,xy[:,0],side='right')-1,0,H.shape[0]-1); j=np.clip(np.searchsorted(ye,xy[:,1],side='right')-1,0,H.shape[1]-1)
    v=H[i,j]; return np.array([(vis<=x).mean() for x in v])
def report_A(files,keyfn):
    print("== A. V8 失敗地理（官方 eval：起點只有五團，看的是死在哪＋死點的資料密度）==")
    print("   校準（資料集本身、2 單位細格）：靠牆格密度分位 中位 0.38 / p25 0.21 / p10 0.13；內部格中位 0.81 ⇒ 死點分位 <0.1 才算「荒漠」訊號、不是靠牆效應")
    death={}; tv={}
    for p in sorted(files):
        k=keyfn(p); d=load(p)
        eps=[e for e in d if e['arm'].startswith(ARM)]; bc=[e for e in d if e['arm'].startswith(BCARM)]
        sr,ft=per_task(eps); bsr,bft=per_task(bc); tv[k]=np.array([ft[t] for t in sorted(ft)])
        fails=[e for e in eps if not e['success']]
        line=f"{k:<10} conf2 {sr:.3f} fail/50 {ft} | BC {bsr:.3f}"
        if fails:
            fin=np.array([e['final'] for e in fails]); st=np.array([e['start'] for e in fails])
            pf=den_pct(fin); ps=den_pct(st)
            line+=f" | 死點資料密度分位 中位 {np.median(pf):.2f}（起點 {np.median(ps):.2f}）"
            pocket=(pf<=0.05).mean(); dense=(pf>=0.30).mean()
            line+=f" | 零資料口袋死 {pocket:.0%}、高密度區死 {dense:.0%}"
            bt=collections.defaultdict(lambda:[0,0])
            for e,q in zip(fails,pf): bt[e['task']][0]+= (q<=0.05); bt[e['task']][1]+=1
            line+="  逐題口袋比 "+" ".join(f"t{t}:{a}/{b}" for t,(a,b) in sorted(bt.items()))
            death[k]={t:np.array([e['final'] for e in fails if e['task']==t]) for t in sorted({e['task'] for e in fails})}
        print(line)
    ks=sorted(death)
    if len(ks)>1:
        print("-- 逐題失敗數向量的兩兩相關（同題全爛→高）--")
        for a,b in itertools.combinations(sorted(tv),2):
            x,y=tv[a],tv[b]; r=np.corrcoef(x,y)[0,1] if x.std()>0 and y.std()>0 else float('nan'); print(f"  {a} vs {b}: r={r:.2f}")
        print("-- 同一題的死點中心：各顆的中位位置與彼此距離（小＝同一條走廊死）--")
        for t in range(1,6):
            cs={k:np.median(death[k][t],axis=0) for k in ks if t in death[k] and len(death[k][t])>=5}
            if len(cs)<2: print(f"  t{t}: 只有 {list(cs)} 有≥5 次失敗"); continue
            pairs=[f"{a}-{b} {np.linalg.norm(cs[a]-cs[b]):.1f}" for a,b in itertools.combinations(sorted(cs),2)]
            print(f"  t{t}: 中心 "+"  ".join(f"{k}({c[0]:.0f},{c[1]:.0f})" for k,c in cs.items())+"  | 距離 "+"  ".join(pairs))
def report_B(dirs):
    print("== B. dz2 重現（boot 不動、換資料流 seed）==")
    rows=[]
    for dd in sorted(dirs,key=lambda x:int(re.search(r'_d(\d+)',x).group(1))):
        dseed=int(re.search(r'_d(\d+)',dd).group(1))
        ro=glob.glob(dd+'/rollout_*_s2.json'); dg=glob.glob(dd+'/diag_*_s2.json')
        if not ro: print(f"d{dseed}: (尚無 rollout)"); continue
        r=load(ro[0])['rates']; line=f"d{dseed:<3} 主打 {r['subgoal']:.3f}  bc {r['bc']:.3f}  null_u {r['null_u']:.3f}"
        if dg:
            eps=[e for e in load(dg[0]) if e['arm'].startswith(ARM)]; sr,ft=per_task(eps); line+=f"   逐題失敗 {ft}"
        rows.append(r['subgoal']); print(line)
    if rows: print(f"n={len(rows)} mean={np.mean(rows):.3f} sd={np.std(rows,ddof=1) if len(rows)>1 else 0:.3f} min={min(rows):.3f} max={max(rows):.3f}   原 dz2=0.880（d2 是同 rng 對照）")
mode=sys.argv[1] if len(sys.argv)>1 else 'both'; test='--test' in sys.argv
if mode in ('A','both'):
    if test:
        files=[f for f in glob.glob('results/night_0901/**/diag_*_s2.json',recursive=True) if '/p2/' in f or '/dz3/' in f]
        report_A(files,lambda p:'s2@'+p.split('/')[-2])
    else:
        files=glob.glob('results/night_0902/v8diag/diag_*.json')
        report_A(files,lambda p:'s'+re.search(r'_s(\d+)\.json$',p).group(1)) if files else print("A: 尚無 diag 檔")
if mode in ('B','both'):
    dirs=glob.glob('results/night_0902/dz2rep_d*'); report_B(dirs) if dirs else print("B: 尚無目錄")
