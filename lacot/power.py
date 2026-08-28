"""這把 dev 尺的解析度 —— 要偵測 X 個百分點的差，需要什麼條件。

🚨 2026-08-28 寫。起因：8/27 我拿一個「300 個獨立樣本」假設下算出來的 bootstrap CI
   去推效應上界，而我十分鐘前才說過那個假設不成立。⇒ 主人當場質疑「太快下結論」。

⭐ 核心事實：配對比較的資訊【只】在兩邊結果不同的題裡。
   一致的題（兩邊都對、或兩邊都錯）對「誰比較好」一個位元都不帶。
   ⇒ 有效樣本數 = discordant pairs 的個數，⛔ 不是題數。
   ⇒ 實測 LaCoT vs BC 在 900 題裡只有 20 個 discordant。

⚠️ 硬下限：discordant < 6 時，就算【全部同向】也達不到 p<0.05
   （2 × 2^-5 = 0.0625）⇒ 那種資料在數學上不可能得出顯著結論，⛔ 不是門檻鬆緊問題。

用法：
    python3 -m lacot.power              # 印檢定力表
    from lacot.power import power_from, effect_upper_bound
"""
from math import comb

import numpy as np

ALPHA = 0.05


def mcnemar_p(nb, nc):
    """discordant pairs 的 exact binomial 雙尾 p。⛔ 不用 scipy。"""
    n = nb + nc
    if n == 0:
        return 1.0
    k = min(nb, nc)
    return float(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n))


def min_discordant(alpha=ALPHA):
    """全部同向時，最少要幾個 discordant 才【可能】顯著。"""
    d = 1
    while mcnemar_p(d, 0) >= alpha and d < 100:
        d += 1
    return d


def power_from(d_mean, q, trials=20000, seed=0, alpha=ALPHA):
    """檢定力 = P(抓得到)。

    ⭐ 用 (d, q) 參數化而不是 (N, 效應) —— 因為真實情境裡兩者是【一起】變的：
       改對了 ⇒ 不一致的題變多（d↑）而且方向變偏（q↑）。
       ⛔ 固定效應只調噪音的那種算法會給出「差異越大越難偵測」的誤導結論。

    d_mean : 平均 discordant 數（Poisson）
    q      : 這些題裡倒向我們那邊的比例（0.5 = 純雜訊）
    """
    rng = np.random.default_rng(seed)
    d = rng.poisson(d_mean, trials)
    nb = rng.binomial(d, q)
    return float(np.mean([mcnemar_p(int(b), int(x - b)) < alpha for b, x in zip(nb, d)]))


def effect_upper_bound(n_discordant, n_total, z=1.96):
    """⭐ discordant rate 本身就是效應的【絕對】上界 —— 兩個 arm 只在這些題上結果不同。

    ⚠️ 這只講【被量的那幾顆模型】，⛔ 不是方法的上界。
    回傳 Wilson 區間（比例接近 0 時 normal 近似會失效）。
    """
    p, n = n_discordant / n_total, n_total
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def _main():
    print(f"⚠️ 硬下限：discordant 至少要 {min_discordant()} 個（且全同向）才可能顯著")
    for d in range(1, 8):
        print(f"     d={d}  全同向 p={mcnemar_p(d, 0):.4f}"
              f"  {'✓' if mcnemar_p(d, 0) < ALPHA else '⛔ 不可能顯著'}")

    qs = (0.6, 0.7, 0.8, 0.9, 1.0)
    print("\n⭐ 檢定力：不一致的題有幾個（d）× 其中倒向我們的比例（q）")
    print(f"{'d':>4} " + "".join(f"{f'q={q}':>9}" for q in qs))
    for d in (5, 10, 20, 40, 80):
        print(f"{d:>4} " + "".join(f"{power_from(d, q):>8.0%} " for q in qs))

    print("\n實測與情境（900 題 = 三個 seed 合併成一批）")
    for d, q, lab in [(20, 0.60, "現在 LaCoT vs BC"), (40, 0.75, "改對一半"), (80, 0.85, "改對了")]:
        print(f"   {lab:>14}  d={d:>3} q={q:.2f}"
              f"  ⇒ 效應 {d * (2 * q - 1) / 900:+.1%}   檢定力 {power_from(d, q):>4.0%}")

    lo, hi = effect_upper_bound(20, 900)
    print(f"\n⭐ 效應絕對上界（實測 20/900 discordant）：{20/900:.2%}"
          f"  95% CI [{lo:.2%}, {hi:.2%}]")
    print("⇒ 真的改對了的話這把尺【現在就夠用】，⛔ 不用先去加題目。")
    print("⇒ 免費的早期訊號：改完先【數 discordant】，還是個位數 ⇒ 就是沒改到東西。")


if __name__ == "__main__":
    _main()
