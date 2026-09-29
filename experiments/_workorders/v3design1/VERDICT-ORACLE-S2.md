# S2 檢察判決：v3 mod-oracle 第一批（Wφ＋Aω＋資料隔離契約）

在證明什麼：本批交付的 oracle 程式與留痕，是否如實實作 DESIGN-v3 (c) 資料隔離契約與 oracle 增量 1-5、(d) 實驗 1 判準；判準＝必抓清單 1-9 逐條可重現的證據。

檢察官：Claude opus（S2，與實作方 GPT astra 不同家族）。判官不動手：未改任何受檢檔、未 commit、未跑 GPU/sbatch。唯一寫入 repo 的是本檔；重跑產物全在 `/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/37e41886-10cf-41d0-9aae-34351f439be3/scratchpad/`（下稱 `$SP`，session 暫存，可能被清；關鍵數字已抄入本檔）。

受檢源碼 SHA256（與 `oracle-smoke/final-audit.log` 一致，檢察期間未變）：refine_models `1bbd1f07…`、refine_quality `e5a3e275…`、train_offline `29c0ea69…`、test `919e9eef…`。`experiments/scratch_lacot_rollout.py` 的並行改動未審、未碰；只讀了 HEAD 版 `make_batch` 做口徑對照（見 6）。

## VERDICT: FAIL(8)

- 範圍要講準：FAIL 只在第 8 項，而且是**來源層**。**權重層**的隔離成立：metadata 字串、檔名、cache 都進不了權重（已實測，見 8-A）。來源層的破口有兩類。第一類是資料白名單自己背書，domain／內容都沒綁，toy 可以洗成 offline 並拿到訓練授權。第二類是 CLI 把 split seed 綁在 model seed 上，跑 3 seeds 就會得到 3 套不同的 split。
- 其餘：1、3、4、6、9 PASS；2、5、7 PASS-with-notes。
- 修完下方「上機前必改」M1-M3 後，本批可轉 PASS-with-notes，不需要重做整批。M4 屬已知 F5 範圍。

## 逐點證據

### 1. 留痕自證：PASS

- 重算方法：讀 `*-final/oracle.pt` 的原始 state_dict，自寫 float64 numpy forward（不呼叫 production 的 ConsequenceEnsemble／PairedController／evaluate）。toy 資料用 seed 0 重建，dataset_hash 對上 report.json，每 split 各 5 窗逐元素對原始 ACT/OBS。initial loss 用 RNG replay：manual_seed(0) 後依 MLP.__init__ 順序建 nn.Linear。腳本在 `$SP/recompute_item1.py`，log 在 `$SP/rerun/item1-recompute.log`。
- 結果：兩域共 32 格，最大相對差 1.5e-7（float32 捨入級）。對報告表格如下：

| 欄位 | 我算 | report.json／報告表 |
|---|---:|---:|
| pointmaze W probe 前→後 | 7.834375 → 0.007112 | 7.834375 → 0.007112 |
| ant W probe 前→後 | 5.423855 → 0.018360 | 5.423854 → 0.018360 |
| pointmaze A probe 後 | 0.635704 | 0.635704 |
| A held-out NMSE pm／ant | 0.798916／1.056742 | 0.798916／1.056742 |
| ant h32 xy world／obs-only NMSE | 0.013890／1.262672 | 0.013890／1.262672 |

- 另有兩處報告文字不準，數字本身無誤：「12 個 metadata 欄位缺一拒收」，實際 `META_FIELDS` 是 11 個；「7 個 injection 子檢查全 FAIL」，實際是 4 FAIL＋3 ERROR（見 5）。

### 2. 判準格自由度：PASS-with-notes

- `qualification_reasons`（refine_quality.py:285-308）的每一格都由量測值比較產生。程式裡沒有寫死必過的門檻，也沒有 clamp、豁免或預設 True。另有兩項佐證：toy 標籤會被擋；`generated_candidates` 目前恆為 None，所以任何資料都過不了 authorize。
- 結構性近恆真（不是 code bug，是 DESIGN (d) 門檻設計本身，需要 lead 裁，見 L1/L2）。實測把 W 換成 action-blind 的 obs-only（xy NMSE 惡化 130 倍，0.0050 → 0.6533）：
  - PI 覆蓋率仍落在 0.867-0.915 → 過 85-95% 門檻。原因是 split conformal 在可交換的 calibration/test 上本來就會給出約 90%。
  - data validity 仍是 0.906-0.916 → 過 ≥90% 門檻。
  - 只有「NMSE ≤0.9×obs-only」與「配對 CI>0」擋下這個 teacher。
  - 真資料 5-step preflight（幾乎沒訓練）也一樣：PI 覆蓋率 0.896-0.908，data validity 0.886-0.939。
  - 結論：held-out 真資料上的 6 格判準裡，有 2 格不分辨 teacher 好壞。
- CLI 內真正恆真的一格：`load_offline_npz(args.data, permit=OfflinePermit(args.data, …))` 的路徑白名單比較，是拿 args.data 跟自己比（歸入 8-B）。

### 3. 預註冊：PASS

- 逐項對 DESIGN (d) 第 137 行：
  - NMSE：`> .9*obs_only` 判不過
  - 配對 CI：下界 `<= 0` 判不過
  - PI：`not .85 <= cov <= .95` 判不過
  - Aω：NMSE `> .1` 判不過，以 train action std 正規化
  - data validity：`< .9` 判不過
  - generated validity 與 nominal＋1 覆蓋：`< .8` 或 None 判不過
- validity 規則對 (c) 第 78 行：support 與 disagreement 各取 calibration p95，再加 tuple 合法。
- 沒有任何一格被放寬。有兩處比設計更嚴：ant 三個子空間都得過；多一道 controller disagreement p95。
- optional：門檻目前是散在程式裡的字面常數，沒有版本化的預註冊區塊，也沒有 hash 進 readings（O9）。

### 4. 尺的出處：PASS

- `subspaces('ant',29)` 為 xy=[0,1]、yaw=[3,4,5,6]、gait=[2,7..28]。這跟 `experiments/bcodec_shat_probe/shat_probe.py:66-79` 的 `subspace_indices` 逐索引相同。
- 正規化也一致。FIX3 的「位移正規化」算的是 `(s_next−s_start)/obs_sd`，誤差等於 `(pred−target)/state_sd`。本批的做法是 `current + delta*state_scale`，誤差是 `(pred−future)/state_norm.scale`，兩者同一把尺。
- 差異只有一處：obs_dim<29 時，FIX3 退成單一 "all" 子空間，本批 pointmaze 則拆成 xy(＋velocity)；ant 本批要求 dim 剛好 ==29，比 FIX3 嚴。

### 5. mutant 走真路徑：PASS-with-notes

- red-controls：用 unittest.mock.patch 在記憶體裡替換 `lacot.refine_models.validate_record`。`train_models` 是用 module global 去查這個名字，所以 patch 打到的正是 production 路徑。
  - 我在最終源碼上重跑 `witness-controls.py.txt`：7 個 injection 子檢查全部不過（4 FAIL＋3 ERROR），shuffle 對照 6/6 FAIL，EXPECTED_FAILURES_DETECTED: True。log 在 `$SP/rerun/red-controls-rerun.log`，traceback 行號 350/218 對得上最終源碼。
  - 3 個 ERROR 的來源：注入的是長度 2 的 tensor，`subset` 取樣時 IndexError。這仍算偵測到。但如果注入的是長度對齊的欄位，訓練根本不讀它，會安靜跑完，然後才 FAIL。守門靠的是 schema，不是 loss。
- red-trained：obs-only 取代 W 之後走真的 `evaluate`，已被上面的重跑覆蓋並重現。
- red.log：跑的是 permissive scaffold（`red-models.py.txt`），不是 production。其中 action-blind 那條 red 是恆假斷言（`action_witness(p,p,t)` 恆為 0，卻斷言它 >0.1），不能算 mutant 證據（見 7）。報告有寫明是 scaffold，但把它列成 red 證據算高估。
- 繞過手法沒有留在 production：四檔 grep `mock|patch|bypass|environ|getenv|skip|force|unsafe|__debug__|assert ` 全乾淨。唯一的旁路是 `load_checkpoint` 用 `OfflineCalibrator.__new__` 還原 calibrator。這條路合理，但沒有再驗一次 restored metadata（O3，optional）。
- 時序注記：`*-final` toy 產物（11:57）與 red-controls（12:00）都早於最終源碼（refine_models 12:01、refine_quality 12:02）。我用最終源碼重跑後 bit 等同／結論相同（見 9），不影響結論。

### 6. 維度/口徑對源碼：PASS（附 M3）

- Aω target 是原始動作，不是插值差分：
  - 組裝：refine_models.py:182 `a[:available] = act[r:r+available]`
  - 訓練：:353 取 `rec["actions"][None,:,:4]`
  - 評估：refine_quality.py:239 取 `record["actions"][:, :4]`
  - 驗證：我的獨立重算用 raw ACT 對上 A NMSE；mutation D（`act[r+1:…]` off-by-one）被 `test_episode_split_and_original_action_pairing` 抓到 FAIL。
- T_CAP 與 h 沒有互換：
  - W 只接 HORIZONS={4,16,32} 物理步，按 4 步 chunk 遞推。
  - path 是 `linspace(r, g, T_CAP)` 插值，`hi=min(lo+1,g)`。
  - rollout 以 `horizon//4` 次 chunk 呼叫 executor；controller 只吃 [B,T_CAP,2]。
  - 對 HEAD 版 `make_batch`（scratch:621-685）逐項比過：起點條件 `te−r≥CHUNK`、終點均勻抽＋CHUNK clamp、插值公式、`ACT[r:r+CHUNK]` 全部一致。production 的 T_CAP＝`min(LACOT_TCAP=128, MAX_TRAIN_T)`，CLI 預設 128 相符。
- 新發現（W 側，列 M3）：窗尾不足 4 步的 partial chunk 以零動作補齊，再餵進 W。
  - `build_records` 把 `available` 之後的 action 補 0。`ConsequenceEnsemble` 是一次吃 4 個動作、吐 4 個狀態，所以 partial chunk 裡有 mask 監督的步，條件其實是**虛構的零動作**。mask 只遮了 target，沒有遮 input。
  - 官方檔實測：`*-large-stitch-v0` 有 10.66% 的候選窗踩到（navigate 2.11%）。
  - evaluate／calibration 只用完整窗，所以讀數不受污染；受影響的是訓練分佈。這跟 (c)「不補虛構未來」的精神衝突。

### 7. 斷言方向：PASS-with-notes

- 恆真斷言 1 條：`test_action_blind_witness_is_detected` 的 `assertEqual(action_witness(zero, zero, target), 0)`，展開就是 x−x=0。`action_witness` 在 production 沒有任何呼叫者，是死碼。
  - 實證：mutation C 把 W 改成 action-blind（action 輸入乘 0），這條照樣 ok。同一個 mutant 讓 `test_shuffled_actions_significantly_worse` 6/6 FAIL，真正在守這件事的是後者。
- 隨機抽 2 條（seed 3345908813），各弄壞一格：
  - `test_normalization_train_only`：讓 evaluate 用 held-out 重擬 normalizer（mutA2）→ 該測試自己的斷言 FAIL（mean 差 1e4）。另一個全面版 mutA 在 setUpClass 就被 teacher fingerprint 擋下 ERROR。
  - `test_decoder_pairing_and_short_horizon_refusal`：拿掉 `gap <= min_shuffle_gap` → FAIL（ValueError not raised）。
- 未改的 sandbox 副本上，同一組測試全 ok（對照組）。log 在 `$SP/rerun/mutations.log`，全部 mutation 都只做在 `$SP/mut/` 的副本上。

### 8. 資料隔離繞道搜索：FAIL（來源層）

**A. 權重層：成立**
- 訓練只讀 state／actions／future／mask／path。metadata 只拿 domain 決定子空間。
- 實證一：同 seed 下，把 metadata 改成 `candidate_id="oracle_winner=3;success=1;reward=9"`、dataset_hash／noise_seed／iteration 全換，三個模型權重仍然 bit-identical。
- 實證二：同義欄位名 16 次注入全被 allowlist 拒收：is_success／Reward／returns／oracle／winner／infos 各注入 record 頂層與 metadata 共 12 次，另外 npz 注入 is_success／Rewards／infos／next_observations 4 次。
- 實證三：split 換標籤攻擊，把 train 改標成 calibration 或 test，都被 episode 交集檢查拒收。
- 檔名與 cache 沒有進入任何模型輸入的路徑。
- 以上 log 在 `$SP/rerun/item8-probes.log`。

**B. 來源層：破口（屬 (c) 明列需求「訓練 loader 僅白名單原始 offline dataset」）**

1. 白名單自己背書。CLI 用 `OfflinePermit(args.data, args.dataset_hash, args.domain)` 建 permit：路徑比較是自己比自己，恆真；hash 則是拿操作者輸入的值去比。任何只含 observations/actions/terminals 的 npz 都會被收成 `offline_dataset`，包括評估 rollout、只留成功 episode 的子集。
2. 內容沒綁。pointmaze 對 obs 維度沒有約束（真 pointmaze obs 是 2 維）。實測把 per-episode success 旗標當成 `observations[:,2]` 塞進去，會被接受，變成 "velocity" 子空間，也成了 W/A 的輸入欄。
3. domain 沒綁。官方 antmaze-large-stitch 檔用 `--domain pointmaze` 可以被接受（29 維變成 xy＋27 維 velocity）。「ant R>0 不支援」那道守門只看 domain 標籤，因此可以被誤標繞過。
4. provenance 由呼叫者自報。`build_records(toy_arrays, source_kind="offline_dataset")` 會被接受。再手填 `readings["generated_candidates"]=candidate_coverage(全 True)` 之後，**`authorize_training(4)` 放行**，這是 toy 資料。readings 的內容不在 teacher_fingerprint 覆蓋範圍，指紋只比對字串欄位。報告寫「toy…無法取得 production training 資格」，這句只在 CLI 路徑成立。
5. `report(prediction=…)`、`tuple_valid`、`rollout(executor)` 都接受呼叫者自備、沒有出處的輸入。實測 `report` 會直接用外部給的後果張量算 cost，這是評估後果回流成品質標籤的潛在管道（O4）。

**C. split seed 綁在 model seed 上（屬 (d)「split hashes 固定；模型 seeds=3」）**
- CLI 把 `--seed` 同時傳給 `build_records`（決定 episode split）和 `train_models`。
- toy 實測：seed1 的 test 與 seed0 的 train 交集 4/6 個 episode，seed1 的 calibration 與 seed0 的 train 交集 3 個。
- 照設計跑 3 個 teacher seeds，就會拿到 3 套不同的「sealed test」，而且互相污染。目前也沒有算 split hash。

### 9. 重跑：PASS

- green suite：`unittest discover` 跑出 19 tests OK，1.881 s（`$SP/rerun/green-rerun.log`）。重跑時設了 `PYTHONDONTWRITEBYTECODE=1`，repo 內的 __pycache__ 沒被寫。
- toy 兩域各跑到新目錄 `$SP/rerun/{pointmaze,ant}`：report.json 除了 elapsed 以外，所有數值與結構欄位 0 差異；metadata.json 完全相同；teacher_fingerprint 相同（b35f4469…／62049c13…）。bit 級可復現。
- 額外做了真資料 CPU preflight，僅驗接線：`~/.ogbench/data` 的 pointmaze-large-stitch 與 antmaze-large-stitch，各 5 steps，exit 0。
  - 官方檔的 qpos/qvel 在白名單內，會被接受。
  - pointmaze 的 2 維 obs 只產生 xy 子空間。
  - 報告宣稱的「未 import gym/gymnasium/ogbench/mujoco」屬實。
  - ⚠️ preflight 的讀數只證明接線，不是 teacher 品質，不得引用成實驗結果。

## 上機（實驗 1 正式訓練）前必改

- **M1｜split 與 seed 解耦**（8-C）
  - 新增獨立的 `--split-seed`，3 個 teacher seeds 共用同一值。
  - split hash＝sha256(dataset_hash, split_seed, 各 split 排序後的 episode ids)，寫進 metadata／checkpoint／calibration fingerprint 與 readings。
  - 加一條測試：不同 model seed 的 split hash 必須相同。
- **M2｜釘死資料清單**（8-B1/2/3，對應 (c)「僅白名單原始 offline dataset」）
  - 提交一份 manifest：`sha256 → (domain, 檔名, obs_dim, act_dim)`，只收實驗 1 要用的那幾個官方檔。
  - loader 只接受 manifest 裡的 hash；domain 與維度從 manifest 取，不從 CLI 取；維度不符就拒收。
  - 參考值（`$SP/rerun/local-ogbench-sha256.txt`，這是本機副本，**UNVERIFIED 對 upstream**）：pointmaze-large-stitch-v0 `9add335e…`、antmaze-large-stitch-v0 `f6887591…`。
- **M3｜partial chunk 不餵虛構動作**（6）
  - `available = (min(32, end−r)//4)*4`，或整段 mask 掉 partial chunk。
  - 加一條測試：mask 為真的步，其 chunk 內 4 個動作全都是真資料。
- **M4｜F5 範圍（報告已承認）**：GPU 路徑，以及預註冊資料量／步數。
  - CLI 的 `--max-windows 1024` 對三個 split 一體適用，`--steps 200` 是 toy 尺度。真資料 train 有約 59 萬候選窗（3000 episodes×197），train 與 calibration/test 需要分別設定。
  - obs-only baseline 要用同預算，並記錄 plateau；否則「贏過沒訓完的 obs-only」會讓 NMSE 比值格變寬。

**gate 使用前必改**（實驗 1 出讀數不需要；任何 `authorize_training`／`quality()` 上線前需要）：
- G1：`offline_dataset` 標籤只能由通過 manifest 驗證的 loader 產生（例如回傳 VerifiedArrays 憑證），`toy_arrays` 只能產生 `toy_offline`。
- G2：readings 與 split hash 綁進 authorize 的核對（hash 或重算）。generated-candidate 讀數由記錄候選出處（parent hash、candidate ids）的 API 產生，不收呼叫者自己給的 bool 矩陣。

## 需 lead 裁（DESIGN (d) 門檻設計層，實作已忠實照做）

- **L1｜PI 覆蓋 85-95% 這格結構上幾乎必過**：split conformal 本來就會給約 90%，action-blind W 也照樣通過。建議改成以寬度對 obs-only 設門檻，或降格為 split 健全性檢查。
- **L2｜data validity ≥90% 近似擲硬幣**：兩個 calibration-p95 過濾器在同分佈資料上的聯合通過率，本來就在 0.90-0.95（兩者獨立時約 0.9025）。實測 0.886-0.941 跟 teacher 好壞無關。可選：每個過濾器改用 p97.5，或改成對 calibration 自身聯合率做差異比較。
- **L3（建議，optional）｜Wφ chunk 內非因果**：4 個動作一起映到 4 個狀態，所以 ŝ_{t+1} 對 a_{t+1..t+3} 有偏導。toy 上是 a_t 敏感度的 2.1-5.4%（pointmaze 約 2%，ant 約 5%），而 toy 的真動力學這項是 0。
  - 在真資料上，行為策略的相關性可能被學成捷徑。shuffled-action 對照抓不到，因為打亂後的 chunk 仍然是行為策略產生的。
  - 建議實驗 1 加一道因果 witness（擾動 a_{t+1..t+3}，量 Δŝ_{t+1}），或在 chunk 內做因果遮罩。

## optional

- O1：刪掉或重寫 `test_action_blind_witness_is_detected`；移除死碼 `action_witness`。
- O2：修正報告文字：11 個 metadata 欄位；4 FAIL＋3 ERROR；red.log 屬 scaffold，其中 action-blind red 恆假。
- O3：`load_checkpoint` 對 restored metadata 再跑一次 `validate_metadata`。
- O4：`report(prediction=…)` 改成私有 helper；`tuple_valid` 與 `executor` 的出處寫進契約。
- O5：`quality()` 對 invalid 回傳 inf。objective 若直接 `.mean()`，loss 會變 inf，GradScaler 會靜默跳過含 invalid 的每一步。F5 objective 批必須先 mask 再 reduce，並附測試。
- O6：Aω 吃的是用 train-split 統計正規化的原始 XY；主線 D(u) 是用全資料 MU_XY/SD_XY 正規化。decoder adapter 必須反正規化，並在真 decoder 上跑 pairing preflight。
- O7：arrival 的 scale 用 path 終點當代理 goal 做校準，跟部署時的 goal 距離分佈不同。接 refine objective 時要再檢視權重平衡。
- O8：`check_decoder_pairing` 的 1e-8 只抓得到完全常數的 decoder（它本身也標 `qualified: False`，這是設計如此）。
- O9：門檻常數集中成一個版本化的預註冊區塊，並 hash 進 readings。

## 自我懷疑（非空）

- item 1 的資料重建沿用 production 的 `build_records`，我只抽了每 split 5 窗對原始陣列，不是全量；initial loss 的 RNG replay 借用了 torch 的 nn.Linear 初始化。
- L3 的捷徑風險在真資料上還只是假說：我只在 toy 量了非因果敏感度，而 toy 根本沒有行為策略捷徑可學。
- 我沒查 OGBench pointmaze 的物理：2 維 obs 是否 Markov、qvel 是否可由動作決定。這會影響 Aω 可辨識性，標 UNVERIFIED。
- 真資料 preflight 用的是本機 `~/.ogbench/data` 副本，跟 upstream 是否一致沒有驗證；5-step 讀數只是接線證據。
- L2「擲硬幣」的定性假設 calibration 與 test 可交換；episode 群聚會讓變異更大。確切的通過機率我沒算。
- mutation 只做了 5 個 mutant（A、A2、B、C、D）加 1 個未改的對照副本，不是完整的 mutation campaign。實際被 mutant 或 negative control 打過的測試有 6 條，其餘 13 條的敏感度只靠閱讀判斷。
- 與 make_batch 的口徑對照只看了 HEAD 版。工作樹的並行改動依指示未看；如果它動到 make_batch，對照結論只適用 HEAD。
- 我把 FAIL 限定在第 8 項。另一種讀法是把 8-B 全部歸到「操作者紀律」，給 PASS-with-notes。我不採用這個讀法，理由是 (c) 契約的目的就是讓這類失誤在機制上不可能發生，而 8-C 在照設計跑 3 seeds 時一定會觸發，不需要任何人失誤。

STATUS: DONE
