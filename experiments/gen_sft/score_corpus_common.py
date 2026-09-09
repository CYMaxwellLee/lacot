"""experiments/gen_sft/score_corpus_common.py —— gen-sft-v2 離線四關打分器（共用函式）。

工單〈四關離線打分挑教材〉的核心：對每條教材（=corpus_v1.pt 裡的一個 (s0,wp_xy,
code_seq) 樣本，對應一整集 episode）算出「這是不是好教材」的 scalar 品質分。

⛔ 全部從錄好的軌跡（raw npz 的 qpos/actions）離線算，不碰模擬器（不 import mujoco、
   不呼叫 env.step）。⛔ 新檔，不改動任何既有檔案。

四關對應（design 第 5 點）：
  ① 到達：hindsight 恆真可略 —— 不進最終 scalar 分數，但下面把「某條腿完全找不到」
     這個退化情形折進磨蹭分（給一個固定滿分懲罰），不需要另開一個獨立門檻，見
     leg_reach_steps() docstring。
  ② 不磨蹭：對每條腿，重新從（可能被破壞過的）xy 軌跡序列偵測「第幾步進到 wp_xy[m]
     的 rho 容忍圈內」（跟 run_teacher_relay.run_one()/eval_gen_sft.run_one_generated()
     的 active_leg 循序判定同語意，但這裡讀的是錄好的陣列，不是即時模擬），除以
     ruler 對該腿直線距離查出的「典型（p50）步數」，比值 >1 代表比典型慢（磨蹭）。
  ③ 不亂動：該集動作序列相鄰差 L2 範數的平均（跟 wv_common.build_ruler 的
     action_smoothness 算法同一套公式，只是這裡算的是「單一 episode 自己的均值」
     不是「全資料集攤平的一個大池」——刻意這樣做，見 score_corpus() 的分位設計說明）。
  ④ 像樣：該集 z 高度序列裡「z < fall_line」比例（fall_line 用既有 ruler_pack.json
     的 p1 翻倒線，不重新定義）。

design 明講「合成一個 scalar score，方法你定並記錄」——這裡記錄選擇：
  - 三個原始量（磨蹭步數、亂動均值、翻倒比例）分別對「訓練集全體」算經驗分位
    （0~1，值越大代表越差），再取三者**最大值**（哪一關最差就以它為準）當
    「badness」，1-badness 當最終 score_quality（0~1，越大越好）。用分位而不是
    z-score，是因為翻倒比例對大部分真實 episode 是 0（零膨脹分佈），z-score
    假設近似常態不成立；分位排名對這種分佈穩健，且天然落在 [0,1] 方便之後轉權重。
  - 三關取 max 不取平均：平均會被『只有一關真的很差、其他兩關普通或良好』的
    情形稀釋掉（這正是驗收①第二次失敗的原因，見 build_score_table() docstring
    的完整推導記錄）；對『好教材』這個目的來說，任何一關踩到很差本身就足以
    否決這條教材，不需要另外兩關來平均稀釋，取 max 更貼近這個直覺，也是這裡
    的設計選擇，記錄於此，不是唯一可能的合成方式。

⚠️ 磨蹭分的公式踩過一次雷、修過一次，記錄在這裡（NOTE 會附完整數字）：
  第一版設計＝「該腿實際步數 / ruler 對該腿直線距離查出的典型步數」（比值），
  跑第一次驗收①（adversarial 必須顯著低於真實 p10）時直接 FAIL——診斷發現
  ~21% 的腿，其『起點到路標』或『路標到路標』的**直線距離**本身就小於 rho
  （常見於這份 antmaze-stitch 資料：弧長 DELTA_SUB=7.5 是沿實際走過的路徑積出來
  的，路徑會繞、會晃，7.5 的弧長不保證換到 >1.875 的直線淨位移），這時
  `ruler_steps_for_distance` 對「距離→典型步數」的查表在 below_grid 分支是線性
  外插到 0（`ys[0]*d/grid[0]`），分母被推到接近 0，比值炸成幾十萬～百萬量級的
  離群值，把整個分位排名淹沒（真實分佈自己就長滿離群值，adversarial 樣本的分位
  排名反而被壓成不顯著）。改法＝不做比值：因為每條腿的**弧長本身幾乎是常數**
  （都卡在 DELTA_SUB=7.5 附近，這是 waypoint 定義方式保證的），『同距離分佈』這件
  事其實已經自動成立，不需要再去查 ruler 表算一個典型值當分母——直接把『重新
  偵測到的逐腿實際步數』本身（見 leg_reach_steps）當作磨蹭的原始量，對全 corpus
  做分位排名即可。ruler 查表版的 leg_typical_steps()/leg_distances() 兩個函式留著
  （沒刪，只是不再進最終分數），因為推導本身沒有錯，錯的是拿它們做除法分母這個
  合成方式；診斷數字見 score_corpus.py 執行 log。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import gen_sft_common as gc  # noqa: E402


def episode_span(raw, s0_idx, n_chunks):
    """回傳這個 corpus 樣本對應的原始錄製陣列切片（不含模擬）：
      xy_after: (horizon,2) —— 第 i 步『執行完之後』的 xy（i=0..horizon-1），
                跟 run_one_generated() 的 global_step 慣例對齊（global_step=0 時
                比對的是走完第一步之後的位置，不含 s0 當下那個起始點本身）。
      actions:  (horizon,8) —— 這集這段用掉的原始動作。
      z_after:  (horizon,) —— 跟 xy_after 同步的 qpos[2]（軀幹高度）。
    horizon = n_chunks*4（跟 eval_gen_sft.run_one_generated 的模擬步數上限同定義）。

    ⚠️ 驗收②抓到的潛在雷（在這份資料集上目前不會真的炸，但留著危險）：這份
    antmaze-medium-stitch-v0 train split 的每一集固定 T=201 步、n_chunks=50，
    horizon=200=T-1，切片 `[s0_idx+1 : s0_idx+horizon+1]` 剛好停在 e0_idx（=這集
    最後一個 index，含）之內，不會讀到下一集。但這是這份資料集『T 剛好不整除 4』
    的巧合，不是這段程式碼本身保證的——如果哪天套到 T 整除 4 的資料集（horizon
    會等於 T），這個切片會多讀一格、讀到下一集的 reset pose。加一個 assert 把這個
    前提明釘出來，而不是靠巧合安全。
    """
    horizon = n_chunks * 4
    e0_idx = int(np.argmax(raw["terminals"][s0_idx:])) + s0_idx
    assert s0_idx + horizon <= e0_idx, (
        f"⛔ horizon={horizon} 會把切片推出這個 episode 的邊界（s0_idx={s0_idx} "
        f"e0_idx={e0_idx}，這集長度={e0_idx - s0_idx + 1}）——會讀到下一集，停手")
    xy_after = raw["qpos"][s0_idx + 1: s0_idx + horizon + 1, :2].copy()
    z_after = raw["qpos"][s0_idx + 1: s0_idx + horizon + 1, 2].copy()
    actions = raw["actions"][s0_idx: s0_idx + horizon].copy()
    assert len(xy_after) == horizon and len(z_after) == horizon and len(actions) == horizon, (
        f"⛔ 切片長度不對：horizon={horizon} xy={len(xy_after)} z={len(z_after)} "
        f"act={len(actions)}（s0_idx={s0_idx} 是否太靠近資料尾端？）")
    return xy_after, actions, z_after, horizon


def leg_distances(s0_xy, wp_xy):
    """每條腿的直線距離：腿1 = s0->wp[0]，腿m = wp[m-2]->wp[m-1]（m>=2）。"""
    M = len(wp_xy)
    d = np.zeros(M, dtype=float)
    prev = np.asarray(s0_xy, dtype=float)
    for m in range(M):
        d[m] = float(np.linalg.norm(np.asarray(wp_xy[m], dtype=float) - prev))
        prev = np.asarray(wp_xy[m], dtype=float)
    return d


def leg_typical_steps(ruler, s0_xy, wp_xy, rho):
    """每條腿『典型（p50）步數』，直接沿用 run_teacher_relay.build_tasks() 算 n_table
    的同一個公式：ruler_steps_for_distance(ruler, max(d_leg-rho,1e-6), "50")——腿的
    有效距離扣掉 rho 容忍半徑（因為只需要『進到 rho 圈內』不必真的走到終點正上方）。
    不透過重新呼叫 rt.build_tasks() 取得（那樣要整批重算、且會把這個函式的正確性
    綁死在另一支程式的內部實作），而是直接呼叫它底層在用的同一個
    wv.ruler_steps_for_distance()，兩者是同一條公式、同一個輸入，數值上保證等價。
    """
    d = leg_distances(s0_xy, wp_xy)
    out = np.zeros(len(d), dtype=float)
    for m in range(len(d)):
        nt, _tag = gc.wv.ruler_steps_for_distance(ruler, max(d[m] - rho, 1e-6), "50")
        out[m] = nt
    return out


def leg_reach_steps(xy_traj, wp_xy, rho, horizon):
    """沿 wp_xy 順序、逐步在 xy_traj 裡偵測『第幾步進到 rho 容忍圈內』。

    ⚠️ 驗收②抓到的語意落差、已修正：第一版是『逐腿』各自對自己的搜尋窗做
    `np.flatnonzero`（向量化、快），但這樣沒辦法在『同一步同時滿足腿 m 和腿
    m+1』時像 run_teacher_relay.run_one() 的 while 迴圈一樣同一步發兩張條—— 舊版
    永遠把腿 m+1 的搜尋起點強制設成『腿 m 命中步 +1』，即使那一步本身也在腿 m+1
    的 rho 圈內也不算。wp_xy 兩點間弧長雖然設計成 ≈DELTA_SUB=7.5，但直線距離可能
    小到 <2*rho=3.75（診斷過，見 NOTE），這種情形下舊版會錯過『同一步吃兩腿』的
    真實可能，讓那一腿的 actual_steps 被高估、甚至在路徑之後沒有繞回來時被錯判成
    『找不到』。改法：逐『步』（不是逐『腿』）掃過 xy_traj，每一步用 while（不是
    if）把當下滿足的所有『已啟用』腿一次判完再往下一步走——這是把
    run_teacher_relay.run_one() 的 active_leg while 迴圈逐行搬過來、只是讀的是
    給定陣列（可能是真實軌跡、也可能是刻意打壞過的假陣列）不是即時模擬。這也是
    本函式故意設計成『對給定陣列重新偵測』而不是查表既有 metadata 的原因：只有
    重新偵測，這個分數才會對『陣列被打壞』敏感（見 NOTE 打分器驗證章節的推導：
    如果直接沿用 build_tasks() 算好的 n_data，那是綁死在原始乾淨軌跡上的靜態
    數字，餵一條打亂的假陣列進來分數不會變，打分器就不會亮）。

    某腿在整個剩餘步數預算裡都沒被判到（可能是它自己沒找到，也可能是更前面的
    腿卡住、它永遠沒『啟用』）：固定給滿懲罰 = horizon（用完整個 episode 的步數
    預算都不算，是能給的最大懲罰）——這個滿懲罰同時把『到達』這個本來該獨立成①
    的門檻，折進磨蹭分數裡（design 允許①可略，這裡的做法是『略掉獨立門檻，但
    没有放棄它的訊號』，見檔頭說明）。

    回傳 (actual_steps (M,) float, reached (M,) bool)。
    """
    M = len(wp_xy)
    wp = np.asarray(wp_xy, dtype=float)
    active_leg = 1                       # 1-indexed，對齊 run_one() 的慣例
    active_step_of_leg = {1: 0}
    leg_reach_step = {}
    for t in range(horizon):
        if active_leg > M:
            break
        cur = xy_traj[t]
        while active_leg <= M and np.linalg.norm(cur - wp[active_leg - 1]) <= rho:
            leg_reach_step[active_leg] = t
            active_leg += 1
            if active_leg <= M:
                active_step_of_leg[active_leg] = t + 1
    actual_steps = np.full(M, float(horizon), dtype=float)
    reached = np.zeros(M, dtype=bool)
    for m in range(1, M + 1):
        if m in leg_reach_step:
            rs, act = leg_reach_step[m], active_step_of_leg[m]
            actual_steps[m - 1] = float(rs - act + 1)
            reached[m - 1] = True
    return actual_steps, reached


def raw_features_for_trajectory(xy_traj, actions, z_traj, wp_xy, s0_xy, ruler, rho, horizon):
    """對一條給定的（可能是真實、也可能是刻意打壞過的）軌跡陣列組，算三個原始特徵
    （值越大越差）+ 診斷用的 reach_frac。純陣列運算，不碰模擬器。

    dawdle_steps＝逐腿重新偵測到的『實際步數』（見 leg_reach_steps）取平均——不除以
    ruler 查表的典型值（見檔頭『踩過一次雷』說明：除法在小距離腿上會除出離群值）。
    因為每條腿的弧長本身幾乎是常數（≈DELTA_SUB），『相對同距離分佈的分位』這件事
    在後面 build_score_table() 對這個原始量做全 corpus 分位排名時就自動成立，不需要
    再用 ruler 查表額外正規化一次。leg_typical_steps() 仍算出來附在回傳值裡供對照
    （n_table），但不參與 dawdle_steps 本身。
    """
    n_table = leg_typical_steps(ruler, s0_xy, wp_xy, rho)
    actual_steps, reached = leg_reach_steps(xy_traj, wp_xy, rho, horizon)
    dawdle_steps = float(np.mean(actual_steps))
    if len(actions) > 1:
        jerk_mean = float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))
    else:
        jerk_mean = 0.0
    fall_line = ruler["torso_z"]["fall_line_p1"]
    fall_frac = float(np.mean(z_traj < fall_line))
    reach_frac = float(np.mean(reached))
    return dict(dawdle_steps=dawdle_steps, jerk_mean=jerk_mean, fall_frac=fall_frac,
                reach_frac=reach_frac, n_table=n_table.tolist(), actual_steps=actual_steps.tolist(),
                reached=reached.tolist())


def percentile_of(value, sorted_population):
    """value 在 sorted_population（已排序的 1D array）裡的『比多少比例的母體小』
    位置，[0,1]。母體不含 value 自己（給待驗證的對照樣本用，不重新把自己算進母體，
    這樣『p10 門檻』的定義才乾淨：p10 完全只由『真實母體』決定）。
    """
    n = len(sorted_population)
    if n == 0:
        return 0.5
    rank = np.searchsorted(sorted_population, value, side="left")
    return float(rank) / float(n)


def build_score_table(raw_dawdle, raw_jerk, raw_fall):
    """三個原始量（各自 1D array，同長度＝corpus 樣本數，值越大越差）→ 各自的
    經驗分位（0~1，越大越差）→ 取三者**最大值**（最差的那一關說了算）當 badness
    → score_quality=1-badness。回傳 dict，含每個樣本的 pct_rank 與 score_quality，
    以及三個排序好的母體陣列（供之後對『外來樣本』如 adversarial 例子算一致定義
    的分位用，見 percentile_of）。

    ⚠️ 合成方式也踩過一次雷、修過一次：第一版用三者『平均』當 badness，跑驗收①
    （adversarial 必須顯著低於真實 p10）第二次還是 FAIL——診斷發現 reverse/shuffle
    這兩種構造各自只打壞『一關』打到 100 百分位（reverse 只打壞磨蹭、jerk 甚至還
    正常；shuffle 只打壞亂動、磨蹭沒有明顯變差），平均下來被另外兩個正常/良好的
    分位拉回中段（badness≈0.3~0.4），score_quality 完全沒有靠近 p10。改用『取最大值』
    （哪一關最差就以它為準）之後，任何單一維度被打到接近 100 百分位就會讓整體
    badness 接近 1、score_quality 接近 0，兩種構造都應聲落底——這也不只是「改到
    測試過」而已，對『好教材』這個目的本身也更合理：一集動作平滑、步速正常，
    但常常翻倒，不該因為前兩關均分掉而被算成『普通』教材，翻倒這一關本身就該
    否決它。三個原始量各自的診斷分位仍然全部保留在回傳值/輸出檔裡，可以另外
    再拿平均或其他合成方式對照，只是最終 score_quality 用 max。

    ⚠️ 驗收②抓到、修過的第三個雷：這裡的分位算法本來跟 percentile_of()（給外來
    樣本，如 val／adversarial 用）不是同一條公式——這裡原本用『中點分位』
    `(嚴格小於的個數+0.5*相等的個數)/n`，percentile_of() 用『嚴格小於的個數/n`。
    對連續量（磨蹭步數、亂動均值）兩者只差 O(1/n)，可忽略；但 fall_frac
    有~96%的樣本恰好等於 0（零膨脹），中點分位版本會把這 96% 的樣本全部打到
    pct_fall≈0.48（≈半數同分「打平」的中點），害得他們的 score_quality 被
    這個人工天花板卡住（診斷：19.4% 的 train 樣本卡在同一個 score_quality
    上限、無法被加權方案分出高下），而外來樣本（如 adversarial 底本，一個真的
    train episode 但改用 percentile_of() 算)算出來的 fall_frac=0 卻直接得
    pct_fall=0（全對照組最好），同一個原始值在兩套公式下天差地遠——這正是造成
    「adversarial 底本分數 0.577 高於全體 train 樣本能拿到的上限 0.519」這個
    自相矛盾數字的根源。修法：兩邊統一只用『嚴格小於的個數/n』（=percentile_of
    的定義）——這個計數本身『self 是否被算進母體』完全不影響結果（self 不會
    嚴格小於自己），只有分母 n 對 n-1 差 O(1/n)，可忽略，所以『population 含
    自己』與『population 不含自己（真正的 leave-one-out）』在這個嚴格小於定義下
    幾乎等價，不需要為了 LOO 另外寫一條中點公式——直接呼叫 percentile_of() 本身
    即可，內部 vs 外來樣本用同一支函式，不會再有兩套定義互相矛盾的問題。
    """
    raw_dawdle = np.asarray(raw_dawdle, dtype=float)
    raw_jerk = np.asarray(raw_jerk, dtype=float)
    raw_fall = np.asarray(raw_fall, dtype=float)
    n = len(raw_dawdle)
    assert len(raw_jerk) == n and len(raw_fall) == n

    sorted_dawdle = np.sort(raw_dawdle)
    sorted_jerk = np.sort(raw_jerk)
    sorted_fall = np.sort(raw_fall)

    # 向量化版的 percentile_of()：對整個陣列一次算，用法跟 percentile_of() 完全
    # 同一條公式（searchsorted side="left" / n），刻意不重複定義第二套。
    def pct_vec(values, sorted_pop):
        return np.searchsorted(sorted_pop, values, side="left") / len(sorted_pop)

    pct_dawdle = pct_vec(raw_dawdle, sorted_dawdle)
    pct_jerk = pct_vec(raw_jerk, sorted_jerk)
    pct_fall = pct_vec(raw_fall, sorted_fall)

    badness = np.maximum(np.maximum(pct_dawdle, pct_jerk), pct_fall)
    score_quality = 1.0 - badness
    return dict(
        pct_dawdle=pct_dawdle, pct_jerk=pct_jerk, pct_fall=pct_fall,
        badness=badness, score_quality=score_quality,
        sorted_dawdle=sorted_dawdle, sorted_jerk=sorted_jerk, sorted_fall=sorted_fall,
    )


def score_quality_of_external(dawdle, jerk, fall, sorted_dawdle, sorted_jerk, sorted_fall):
    """給一個『不在原本母體裡』的外來樣本（例如 adversarial 假軌跡）算 score_quality，
    用 build_score_table() 已經算好、來自『真實訓練集』的三個排序母體當參照 ——
    這樣『它比真實分佈的第幾百分位』這個講法對 adversarial 樣本也有意義（它沒有
    被混進母體去影響分位定義本身，p10 門檻乾淨地只由真實資料決定）。
    """
    pd = percentile_of(dawdle, sorted_dawdle)
    pj = percentile_of(jerk, sorted_jerk)
    pf = percentile_of(fall, sorted_fall)
    badness = max(pd, pj, pf)
    return dict(pct_dawdle=pd, pct_jerk=pj, pct_fall=pf, badness=badness,
                score_quality=1.0 - badness)


def make_adversarial(xy_traj, actions, z_traj, mode, seed):
    """把一條真實軌跡陣列組打壞，構造工單驗收①要的『人工爛軌跡』。
    mode='reverse'：整段（xy,actions,z）三個陣列一起做時間反轉。
    mode='shuffle'：整段（xy,actions,z）三個陣列套同一個隨機排列（固定 seed）。

    ⚠️ 記錄一個推導清楚、故意做過數值驗證的設計決定：『reverse』單獨不會動到
    亂動分數（jerk）—— 因為相鄰差的 L2 範數在時間反轉下是不變量（反轉後相鄰對
    只是原本相鄰對的方向掉過來，範數對稱），會亮的是磨蹭分（因為 wp_xy 的順序
    沒有跟著反轉，反轉後的軌跡等於『反著』去追一個正著排的路標序列，多半直接
    在第一條腿就找不到、觸發滿懲罰）。『shuffle』兩個分都會亮（相鄰配對被打散
    直接推高 jerk；路徑連續性被破壞，磨蹭分也會惡化）。這是刻意設計成兩種
    不同性質的反例，不是隨手挑的擾動，NOTE 會附這兩者分開報的數字驗證這個推導。
    """
    n = len(actions)
    assert len(xy_traj) == n and len(z_traj) == n
    if mode == "reverse":
        return xy_traj[::-1].copy(), actions[::-1].copy(), z_traj[::-1].copy()
    if mode == "shuffle":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return xy_traj[perm].copy(), actions[perm].copy(), z_traj[perm].copy()
    raise ValueError(f"⛔ 不認得的 mode={mode!r}")


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    s1 = float(w.sum())
    s2 = float(np.sum(w ** 2))
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def pick_beta_for_ess(scores, target_ess_frac=0.5, beta_hi0=50.0, tol_frac=1e-3, max_iter=60):
    """找 beta 使 weights=exp(beta*(score-score.max())) 的 effective sample size
    (ESS) / n ≈ target_ess_frac。beta=0 時 ESS/n=1（均勻權重）；beta 越大權重越
    集中在高分樣本，ESS/n 越小 —— 對 beta 做二分搜尋（單調遞減，指數傾斜的標準
    性質）。回傳 (beta, weights(未正規化,但已用 max 平移避免 overflow), ess, ess_frac)。
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    target_ess = target_ess_frac * n
    smax = scores.max()

    def ess_at(beta):
        w = np.exp(beta * (scores - smax))
        return effective_sample_size(w)

    lo, hi = 0.0, beta_hi0
    tries = 0
    while ess_at(hi) > target_ess and tries < 30:
        hi *= 2.0
        tries += 1
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e = ess_at(mid)
        if abs(e - target_ess) < tol_frac * n:
            lo = hi = mid
            break
        if e > target_ess:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    w = np.exp(beta * (scores - smax))
    ess = effective_sample_size(w)
    return beta, w, ess, ess / n
