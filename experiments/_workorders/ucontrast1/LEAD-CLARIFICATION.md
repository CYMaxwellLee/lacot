# lead 澄清（2026-09-28；對 IMPL-REPORT.md 的三個阻塞項）

_實作官（codex 一次性 exec）已收工無法回訊，本檔為決策記錄，供檢察官與後續執行者。_

## 1. ⭐ .6680 錨=工單錯置（實作官的懷疑正確、lead 認錯）

lead 親驗：.6680 出自 `~/Projects/lacot-gensft/experiments/gen_sft/bcodec_probe_bon_20260914/bon_t1p0_n16_full200_bon_summary.json`——它是 **bcodec 行為字典線**（ant、gensft 模型 `bcodec_u2s_g16k16_s20260913`、尺=per_leg_untimed_reach）的帳。「s13」=seed 20260913 縮寫，不是本件 pointmaze ckpt s33/s35 家族。工單把兩條線的 u 混了。

**修正後的正向校準錨（取代 .6680 復現）**：
(a) flag-off 逐位元等價——CPU 已驗 9 tests，GPU 版跑 3 題確認即可；
(b) A 臂 draw0 的 200 題 R1 應落在 j_signal noclimb s33=.350 的抽樣誤差內（同 ckpt 同題集同尺）。draw0 與原版 rollout 的 RNG 消耗若不同導致逐題差異屬預期——比整體率不比逐題。smoke 階段 25 題先看 sanity 帶（.15~.6）。
(b) 不過才是真異常。

**戰場後續**：bcodec 線的殺手對照（.9451 那把、9/27 CONVERGENCE 裁的本尊）另立工單補班（復用 `lacot-gensft/experiments/gen_sft/bcodec_probe_bon2_eval.py` 加噪音臂）；本件 pointmaze 版照走——它直接餵 v3 selector 的候選供給前置（DESIGN-v3 實驗 1 的 candidate validity 格）。

## 2. B 凍 u 語義：實作正確

cache key=(task,episode)、全 draws 復用首 draw u＝工單要的語義。照此走。

## 3. provenance / mtime gate：對凍結輸入資產不適用

s33/s35 是 j_signal README 指定「不得替換」的凍結資產，本件不產新 ckpt。pick-artifacts 雙條件（mtime 晚於起跑＋編號最大）是防撈錯【新產物】的規則；對固定輸入改為：SHA256（已記）＋路徑逐字對 README 表（已對）=滿足。provenance.json 的起跑時間格填 null＋註明「凍結輸入資產、不適用」。

## 4. sbatch 連線被拒=codex 沙盒結構性限制，不是錯

校準/mutant/smoke 的 sbatch 由 lead 側代發（F5 發射權：smoke 級代發、正式親手）。執行順序與 PASS 判準以實作包 README 的 gate 次序為準，逐關過了才走下一關。
