"""收 9/3 夜批 N3/N4/N5（沿 collect_dialect.py 讀法；subgoal＋R0＋bc 三欄並列）。
讀法（handoff 9/3 ②）：
  N3 fsqz_cont  對 z甲 .712 與基線 .792：≈.79 ⇒ −.08 全是 round 的錯；≈.71 ⇒ 全是壓縮費；中間＝分帳
  N4 cd{3,5}    主指標 R0＋u淨貢獻（對 z乙 cd0.1 R0 .505）；⛔ subgoal 預期仍低、別當退步報
  N5 s26/s27    對凍 s20 .792 sd .040：兩組 sd 都小 ⇒「挑得好的 stage 1 都穩」；平均差＝stage 1 品質
"""
import json, glob, re, os
import numpy as np

os.chdir(os.path.expanduser('~/Projects/lacot'))

ARMS = {
    'N3 fsqz_cont (只壓縮不round)': 'results/night_0903/fsqz_cont',
    'N4 z乙×cd0.3':                 'results/night_0903/fsqz_dq_cd3',
    'N4 z乙×cd0.5':                 'results/night_0903/fsqz_dq_cd5',
    'N5 凍s26':                     'results/night_0903/dialect_s26',
    'N5 凍s27':                     'results/night_0903/dialect_s27',
}

def stat(v):
    v = np.array(v, float)
    sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.3f} sd {sd:.3f} 範圍 {v.min():.3f}~{v.max():.3f}"

for name, d in ARMS.items():
    rs = {}
    for p in glob.glob(f'{d}/rollout_*.json'):
        S = int(re.search(r'_s(\d+)\.json$', p).group(1))
        rs[S] = json.load(open(p))['rates']
    ks = sorted(rs)
    print(f"== {name}  n={len(ks)} ==")
    for k in ks:
        r = rs[k]
        net = r['R0'] - r['null_u']
        print(f"  s{k}: subgoal {r['subgoal']:.3f}  R0 {r['R0']:.3f} (null_u {r['null_u']:.3f}, shuf {r['shuf']:.3f}, 淨 {net:+.3f})  bc {r['bc']:.3f}")
    for col in ('subgoal', 'R0', 'bc'):
        print(f"  {col:>7}: {stat([rs[k][col] for k in ks])}")
    print(f"  u淨貢獻(R0-null_u): {stat([rs[k]['R0']-rs[k]['null_u'] for k in ks])}")
    print()

print("對照（9/3 已量）：基線凍s20 subgoal .792 sd .040 / R0 .373")
print("　　　z甲(round) subgoal .712 / R0 .455 ｜ z乙(cd0.1) subgoal .604 / R0 .505 升7平1降0")
print("　　　soft 八顆(各自 stage1) .752 sd .168 ｜ 重評雜訊 ±.03")
