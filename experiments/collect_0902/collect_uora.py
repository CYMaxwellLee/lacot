"""收 oracle-u 探針：同顆 ckpt，conf2（flow 生 u）vs conf2+oracle u（BFS 正確路徑→encoder→u）。
判讀：oracle 高（≈ebfs 的 1.0）＝表示 enc/dec 沒歪、錯在 flow 生成；oracle 也低＝表示本身歪。"""
import json,glob,os,re,sys,collections
import numpy as np
os.chdir(os.path.expanduser('~/Projects/lacot'))
print("== oracle-u 探針：conf2(flow u) vs conf2(oracle u) vs ebfs(正確路標) ==")
for S in (23,25,26,27):
    def r(d):
        p=glob.glob(f'results/night_0902/{d}/rollout_*_s{S}.json'); return json.load(open(p[0]))['rates']['subgoal'] if p else None
    a,b,c=r('v8diag'),r('uora'),r('ebfs')
    dg=glob.glob(f'results/night_0902/uora/diag_*_s{S}.json'); pt=None
    if dg:
        eps=[e for e in json.load(open(dg[0])) if e['arm'].startswith('分段 conf2')]
        cnt=collections.defaultdict(int)
        for e in eps: cnt[e['task']]+=(0 if e['success'] else 1)
        pt=[cnt[t] for t in sorted(cnt)]
    f=lambda x: '—' if x is None else f'{x:.3f}'
    print(f"s{S}: flow-u {f(a)}   oracle-u {f(b)}   ebfs {f(c)}   oracle 逐題失敗 {pt}")
