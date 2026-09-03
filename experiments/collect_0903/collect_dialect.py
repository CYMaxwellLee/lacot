"""收 9/3 方言分辨批：night_0903/dialect（凍 s20 soft stage1、stage2 換 s40~47）。
判讀（決策 10）：sd 對 soft 八顆 .168 —— 仍大＝抽籤住 stage 2 訓練；變小＝住 u 空間形狀（⇒ FSQ）。"""
import json,glob,re
import numpy as np
import os
os.chdir(os.path.expanduser('~/Projects/lacot'))
rs={}
for p in glob.glob('results/night_0903/dialect/rollout_*.json'):
    S=int(re.search(r'_s(\d+)\.json$',p).group(1))
    rs[S]=json.load(open(p))['rates']
ks=sorted(rs)
v=[rs[k]['subgoal'] for k in ks]
print("== 方言分辨（凍 s20 soft stage1、stage2 換 seed）==")
for k in ks: print(f"  s{k}: {rs[k]['subgoal']:.3f}")
print(f"n={len(v)} 平均 {np.mean(v):.3f} sd {np.std(v,ddof=1) if len(v)>1 else 0:.3f} 範圍 {min(v):.3f}~{max(v):.3f} 爛顆(<.3) {sum(x<.3 for x in v)}")
print("對照：soft 八顆(各自 stage1) .752 sd .168 ｜ D批只換 init sd .151 ｜ 只換資料順序 sd .056 ｜ 重評雜訊 ±.03")
