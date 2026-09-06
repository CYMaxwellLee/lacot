# DESIGN — explore×verify BoN：推論期「探索×驗證器」臂（rung 0.5 設計卡 v0、2026-09-06 呈裁）

_設計使魔（Fable 級）。唯讀輸入：THEORY-0906-information-ledger（帳本）、DESIGN-0906-
grpo-thoughts（GRPO 卡）、experiments/probe_grpo_headroom_report.txt＋probe_grpo_headroom.py
（rung0 實測與儀器、唯讀）、THEORY-0906-verifier-taught-latent-thinking（敘事）、
PLAN-0906-forward（時鐘與佇列）。_
_⛔ 本卡只設計、不施工不跑：所有門檻數值標【猜測】＝待 rung 0.5 實測替換；
所有「預期」句的數字皆從 rung0 報告推導、逐項附出處。_
_引用紀律：只引 repo docs 已有件；外部文獻一律標「經○○檔攜帶」、⛔ 無新 cite。
本卡回應的正是帳本檔呈裁點①（「部署期 BoN 臂要不要排 — rung 0 儀器順手可量」）：
這裡把那句話落成完整設計、排程仍呈裁（§7）。_

---

## 0. 一句話 ＋ ⓪ 四格自檢

**一句話**：推論期對每題 (s,g) 抽 N 條自生 latent 計畫 z（＝帳本語言的 u）、各自 decode
成路徑、用「佔據圖合法性＋BFS 可達性」的免費外部驗證器打分（沿用 GRPO 卡 §2.1 三項
reward＋C8 乘法閘、rung0 的 roundtrip-decode 校準語義）、選最好那條出計畫 — 免訓練地
兌現帳本 Prop L2 的 selection channel：每次推論注入 ≤ log N bits，是主人「靠 embedding
space 探索得到資訊」那問的**可跑肯定式**（探索自身零資訊、探索×驗證器有資訊），同時
是 rung1 GRPO 的 test-time 上界 baseline 與 paper 效率軸（T2）素材。

**⓪ 四格**（灑批前自檢慣例、先答在卡頭）：

| 格 | 答案 |
|---|---|
| 對準哪一缺？ | 不在三缺主鏈上 — 對準的是 (a) 帳本理論的實驗面（channel (iii) transient 消費格、L2 從定理落到可量）(b) T2 效率軸素材 (c) rung1 該打敗的判準線。它是帳本呈裁點①的落地，不是 T1 的格 |
| 推進還是轉圈？ | 推進 — 免訓練、CPU 分鐘級，量完就是 rung1 的量尺；且 zero 模式 headroom（V0）是 GRPO 卡 R-zero 主臂缺的那格（rung0 只量了 on 模式），一魚兩吃（⚠️ 9/6 深審 M1：兩錶只在 reward 二值時重合、dense reward 下 gap 是單向訊號） |
| 證明／證偽什麼？ | 證明「探索×外部驗證器」部署期就能兌現增益（plan 層預期 +.14~.19@N=16 on 模式、由 rung0 數字＋支配引理釘死 — §3.2）；真判決在 (i) zero 模式 headroom（V0）與 (ii) R0 兌現率。證偽面：V0 <.05 ⇒ 部署 BoN 無物可選、臂降級 — 且對 rung1 R-zero 主臂是同源壞消息，要一起報 |
| 為什麼是現在？ | rung0 儀器（GrpoReward、C8 校準、U1–U5）現成、三顆 ckpt 全過 ≥.15 預釘線（.169/.183/.192）＝獎品已證實存在；免訓練 ⇒ 不佔 P0–P2 判決鏈。**排程仍呈裁**（§7、⛔ 卡不寫死） |

---

## 1. 機制

### 1.1 流程（每題 (s,g)、eval 期）

```
1. cond 束 c：
     on 模式:   condvec(s,g,route-ix)      （rung0 探針同款 eval 語意 — 診斷格）
     zero 模式: ι=0                         （部署格＝主 claim 格；語義見 §6.3）
2. 抽 N 條 z ~ Flow.sample(c)              （@no_grad；專用 torch.Generator —
                                             GRPO 卡 §3.1 加固技同款 ⇒ N=1 臂與
                                             pass@1 golden 成對；N-scan 用 32 條
                                             一次抽、取前綴 ⇒ 各 N 配對可比）
3. 各自 decode：pts_i = u_dec(z_i) → [T,2]  （deterministic；⛔ 不套 E_geo refine —
                                             rung0 探針同紀律：獎品量 prior 原樣；
                                             refine×BoN 合成是另一格、本卡不含）
4. 驗證打分：r_i = GrpoReward.score(pts_i, s_cell, g_cell)
     ＝ N(z)·(w1·r_legal + w2·r_reach + w3·r_hit)、w=(1/3,1/3,1/3)
     合法性＝穿牆檢查（r_legal：逐點自由格比例、佔據圖＝資料建 _EOCC —
       ⑤'' 帶查(map) 合法部署語意、⛔ 非 privileged env.maze_map；經 GRPO 卡 §2.1 攜帶）
     可達性＝BFS 到目標（r_reach/r_hit：從 goal cell 的 grid_bfs dist 場、per-goal
       _BFS_CACHE 快取）
     N(z)＝C8 乘法閘（arclen floor＋鄰步距 cap）— 防刷分設計整包沿用 GRPO 卡
       §2.1/§2.2；校準語義整包沿用 rung0：⛔ 只准在 roundtrip-decode 空間校準
       （ρ_len=p5、δ_step=p95、per-ckpt per-day；raw 插值空間會閘掉 100% 的
       decode 產物、含真軌跡 roundtrip — 兩空間差 3.2×，rung0 report §2/§3 實測）
5. 選擇：J = argmax_i r_i（平手含全 0 群 ⇒ 獨立噪音均勻挑 ⇒ 退化群下 BoN 分佈上
   恰等於 pass@1 — 結構上不會更差）
6. 出計畫：pts_J → executor（R0 眼）；plan 層讀數＝pts_J 的 raw hit
```

儀器重用：`probe_grpo_headroom.py::GrpoReward`（`calibrate/dist_from/score/score_terms`）
與 U1–U5 儀器 gate 原樣搬 — rung0 報告已驗「這把尺對這顆、當天、在這台是活的」；
任何新 ckpt／新模式讀數前，U1–U5 必須先全綠、否則 INSTRUMENT INVALID 不放行。

### 1.2 支配引理（⭐ 本卡唯一新命題；一行證、可證偽）

**Lemma（閘內命中支配）**：對上式 score，若 **w3 ≥ w1**，則任一過閘命中候選
（N=1、r_hit=1）的 r 嚴格大於任一未命中候選的 r。
_證_：記 ρ(d)=max(0, 1−d/max(D_s,1))（＝r_reach、對 d 非增；d=∞ ⇒ 0）。命中者
d_end ≤ 1 且末格自由 ⇒ r ≥ w1·(1/T) + w2·ρ(1) + w3；未命中者 d_end ≥ 2 ⇒
r ≤ w1·1 + w2·ρ(2) + 0（未過閘者 r=0 更小）；差 ≥ w2[ρ(1)−ρ(2)] + w3 − w1(1−1/T)
> 0 當 w3 ≥ w1（前項 ≥0 由 ρ 非增）。（D_s=0 題 r_reach=1[d=0]，同式成立。）∎

**推論（實現值＝rung0 已量之數）**：argmax-r 在「存在過閘命中候選」的每一題必選中
過閘命中 ⇒ **realized BoN-hit@N ＝ gated-hit pass@N**（偏差僅來自全 0 群均勻挑 —
rung0 實測全 0 群 0.0% — 與閘誤咬真解，U4：真軌跡 roundtrip 7.4% 被閘）。
⇒ plan 層的 BoN 實現增益**不是新實驗、是 rung0 gated 欄的重述**（§3.2 錨）；
本臂真正的新資訊只有兩格：zero 模式 headroom（V0）與 R0 兌現率（c 格）。
w=(1/3,1/3,1/3) 釘死＝rung0 實跑值（錨全在它之下量的）；w3 ≥ w1 升格為設計約束。

### 1.3 三格實驗（rung 0.5 的內部結構；⛔ 描述、不施工）

| 格 | 內容 | 量級 |
|---|---|---|
| **a（V0）zero 模式 headroom** | rung0 探針加 ι=0 分支重跑（探針檔頭 78-79 行自己釘「zero 是另一格、不量不外推」— 這格補上它；同時餵 GRPO 卡 R-zero 主臂）。C8 對 zero 模式**重新校準**＋U1–U5 重過 | 探針 patch ~50 行【猜測】、CPU 分鐘級 |
| **b（N-scan）plan 層實現值** | N∈{1,4,8,16,32} 前綴協定（§1.1 步 2）；讀 realized BoN-hit、oracle pass@N、gated pass@N、退化群率、閘率 per N per 模式 | 探針 patch ~100 行【猜測】、CPU 分鐘級 |
| **c（R0 錶）兌現率** | executor 沿 BoN 選出計畫走、標準 R0 協定、on/zero 兩模式；env 旗 `LACOT_BON_N`（0=off ⇒ **golden：現行為逐位元零差**、DIV_W 慣例）＋`LACOT_BON_MODE(on|zero)`；報表開 **BoN@N 專欄** — ⛔ 不寫進 base R0 格、不進 T1 欄（⑤'' 口徑紀律；BoN 是部署策略變更、得有自己的欄） | eval 路徑 patch ~100 行【猜測】＋eval 節點批次 |

---

## 2. 資訊帳（對照帳本檔、逐命題引節號）

| 帳本命題 | 內容 | 本臂讀法 |
|---|---|---|
| §2.1 Prop L1(b) | 純自生：任意 N 條、任意深度聯合恆 0 bits | **對比敘述的那半**：N 條 z 自己一分錢不進帳 — 「多抽幾條就多知道路」被定理殺死；沒有步 4-5，本臂＝零資訊的計算 |
| §2.2 Prop L2(i) | 選擇預算：I(W;u*\|s,g) ≤ H(J\|s,g) ≤ **log N** | **每次推論的注入上界**：N=4/8/16/32 ⇒ ≤2/3/4/5 bits。bits 全部經 J 進場、一條 z 都不帶 |
| §2.2 Prop L2(ii) | 驗證器存量帽：≤ I(W;M_V\|s,g) | M_V＝資料建 E 圖＋BFS 場：查得動的以它能表達者為限（§6.1）；從驗證器擠不出它沒有的 |
| §2.2 Prop L2(iii) | 自驗證＝0 新 bits | **禁用孿生**（§6.3）：拿 flow 自己的 log_prob 或任何 θ-可測 critic 當分數 ⇒ 注入恰 0（部署期帳本條件在固定 θ 上）。我們的 E 圖不是 θ 的函數、本身就是 W 變數 ⇒ 真 (iii) 注入 — 帳本 L4.5 防火牆：sharpening 批評（2412.01951／2504.13837、經帳本攜帶）不適用於本臂 |
| §2.2 L2(v) Remark | ⚠️ 兩種 log N 別混 | W-bits 界（上排）之外另有**分佈傾斜**定理：KL(π_BoN‖π_base) ≤ log N − (N−1)/N（2401.01879 Thm 3.1、經帳本 §2.2(v) 攜帶；可達性 2404.01730 同處）。本卡兩者都用、⛔ 分開記帳 |
| §2.2 L2(vi) Remark | 驗證器分離：verifier-free 可證地虧、有 M_V 可證地贏 | 本臂的正向錨（2502.12118、經帳本攜帶）：BoN 增益不是僥倖 — 兩側佐證之一（⚠️ 9/6 深審 M7：bits 上界與 performance 標度分離量綱不同、非數學夾擠） |
| §0 Prop L0＋合成律 R2 | 地板/赤字分解；refine/BoN＝把有效溫度往 0 壓（R2 定性、經敘事 §1(7) 攜帶） | BoN 不動 base sampler 的 NLL 地板 — 增益直接以 T→0 成功率貨幣兌現、跳過 NLL 記帳。⛔ 別把「≤ log N bits」讀成 R0 界（帳本 §6 匯率紀律） |
| §2.4 Prop L4＋§5 | 同一批 (iii) bits 兩條出路：推論期當場消費（transient）或經 GRPO 打進 θ（persistent） | **本臂＝transient 那半**、rung1＝persistent 那半 — 主人兩問的兩半在同一組命題裡（§4） |

**有效注入折扣〔啟發式〕**：退化群（G 條 r 全同）內 J ⊥ W ⇒ 有效注入
≤ (1−q_deg)·log N。rung0 on 模式實測 q_deg=29.7~34.4%（G=16）⇒ 名目 4 bits、
有效 ~2.6–2.8 bits；zero 模式 q_deg 未知（V0 量）。
**容量對齊〔啟發式、帳本 §2.4 Remark〕**：instance 級 route 資訊 ~2.5 bits（⑰、經帳本
攜帶）⇒ N=8 的 3 bits 已在「一次選擇夠指定一條 route」的刻度上 — N-scan 測的就是
這條對齊之外還剩多少（§3.3 形狀預期）。

**一句話（帳本 §5 原句）**：探索是油門，驗證器才是油箱 — 本臂是那句話的可跑版。

---

## 3. 判準與預期（預釘；⛔ 跑前寫死、不事後挑）

### 3.1 儀器 gate（讀數前置、任一沒過 ⇒ 該格 INSTRUMENT INVALID）

- U1–U5 per ckpt **per 模式**全綠（zero 模式 C8 重校準後重過）；
- N=1 golden：與 pass@1 配對差在 1 SE 內（同 generator 前綴協定使然 — 超出＝儀器 bug）；
- `LACOT_BON_N=0` golden：現 eval 行為逐位元零差。

### 3.2 預期增益錨（全部從 rung0 報告推導；on 模式、G=16、64 題 dev）

| 量 | f27n s40 | idpxm s40 | f27nL s40 | 出處 |
|---|---|---|---|---|
| oracle raw-hit headroom（pass@G−pass@1） | .183 | .192 | .169 | rung0 §5 判準行（≥.15 預釘線三顆全過） |
| gated-hit pass@G | .938 | .953 | .906 | rung0 §3 各節 |
| raw pass@1 | .771 | .761 | .769 | 同上 |
| **⇒ 預測 realized 增益@N=16**（支配引理：gated pass@G − raw pass@1） | **+.182** | **+.192** | **+.137** | §1.2 推論 |
| 難題切片 L_BFS≥3（n=28、⛔ 診斷列不進判準 — rung0 同紀律） | +.366 | +.382 | +.328 | rung0 §3（oracle） |

- **BoN 可兌現上限**＝pass@1→pass@G 的差（GRPO 卡 §3.4「獎品」原句；NF-CoT
  pass@k-vs-k 診斷、經 GRPO 卡攜帶）— rung0 已量、上表照抄。
- N=8 預期：log 形內插 ⇒ 上表 ×0.7~0.85【猜測】≈ +.10~.16 plan 層（on 模式）。
- **zero 模式：⛔ 無錨不猜數** — rung0 探針明文不外推（檔頭 78-79 行）；V0 先量、
  判準線沿用 rung0 預釘同一把尺（≥.15 值得、<.05 降級、其間邊際 — 不另發明線）。
- R0 層：無錨（rung0 全在 plan 層）；兌現率＝c 格的本體讀數。已知天花板：U4
  roundtrip hit 82.8%（decoder 端點保真度）壓住一切 pass@* 讀數。

### 3.3 判讀樹（岔先釘）

1. **V0（zero headroom）**：≥.15 ⇒ zero 格全開（部署 claim 成立要件）｜.05–.15 ⇒
   邊際 — on 格照跑（有實測錨）、zero 格降診斷｜<.05 ⇒ **zero 模式無物可選** ⇒
   本臂降級為 on 模式 demo＋資訊帳敘事；⭐ 同時通報 rung1：R-zero 主臂同源壞消息
   （同一個 prior、同一批候選 — GRPO 卡 §4 rung 0 gate 的 zero 版本）。
2. **b 格支配檢**：|realized − gated pass@N| ≤ .02【猜測】⇒ 引理成立、plan 層照錨讀｜
   超出 ⇒ 先修儀器（w3≥w1？平手噪音？閘 bug？）⛔ 修好前不讀任何下游格。
3. **c 格主判（primary endpoint、唯一釘死格）：zero 模式 R0(BoN@8) − R0(@1)**：
   ≥ +.05【猜測】⇒ 效率軸素材強、進 T2 一列｜+.02~.05 ⇒ 邊際、次要報告不 headline｜
   < +.02 且 plan 層增益 ≥ .10 ⇒ **proxy-gap 判決**：瓶頸在 decoder/executor
   （U4 82.8% 天花板；GRPO 卡 §5.1 風險 3 同型）⇒ 檔案給 D 線／decoder 工作、
   **⛔ 不加 N**（N 治不了兌現率）。
4. **N 形狀檢**：gain(4) ≥ 0.4·gain(16)【猜測】（log N 報酬遞減形；帳本 L2(v) KL 預算
   ＋覆蓋啟發式 N ≳ 1/p_min）。違反且 realized≈oracle ⇒ 覆蓋受限（難題 p_min 小）⇒
   記給 proposal 多樣性工作（溫度／intent 噪聲 — GRPO 卡 §3.4 藥單）、不無腦加 N；
   realized 隨 N 掉而 oracle 升 ⇒ §6.2 盲點選入訊號。
5. **多重比較紀律**：N×模式×ckpt×雙尺格子很多 — primary 只有第 3 條那一格，
   其餘一律 diagnostic；⛔ 不准跑完在格子裡挑好看的當結論。

### 3.4 R0 錶（c 格報表形；on/zero 兩模式都量）

列＝ckpt（f27nL 主、f27n/idpxm 對照）；欄＝R0@1(on)、R0-BoN@8(on)、R0@1(zero)、
R0-BoN@8(zero)、配對差×2、plan 層 realized×2、q_deg、閘率。⑤'' 口徑：BoN 欄獨立、
⛔ 不混 base 格、不進 T1；配對協定＝同 generator 前綴 ⇒ @1 與 @8 逐題成對。

---

## 4. 與 GRPO 臂（rung1）的關係

- **BoN＝GRPO 的 test-time 上界 baseline**：GRPO 的賣點＝把 BoN 收益**內化進權重**、
  免 N 倍推理成本（敘事 §1(6)：「把 pass@G 裡已存在但低機率的成功搬進 pass@1」；
  帳本 L4：同批 bits 的 persistent 出路）。⇒ rung1 該打敗／逼近的線就是本臂的數：
  **內化效率 η_int ≔ [pass@1(post-RL) − pass@1(base)] / [BoN@G(base) − pass@1(base)]**，
  G 對齊訓練 G=8、模式對齊 R-zero。η_int ≥ 0.5【猜測】＝「至少半個獎品搬進了權重、
  推理成本 1/N」；≈1＝全內化；>1 ⇒ 先過 generic-T 指紋檢（⑱' 齊漲型；GRPO 卡
  §3.3、step-matched 對照鐵則不變）再談泛化紅利。
- **指紋二（預註冊）**：若 RL 真內化，post-RL 的 BoN headroom 應**收縮**
  （headroom(post-RL) < headroom(base)【猜測】— 獎品被消費掉了）；不縮 ⇒ RL 搬的
  質量不在這批成功上、內化敘事要重查。與敘事 §3 預測 11（zero 漲多於 on）成對。
- **效率軸素材（T2）**：內化 vs test-time search 的對比在本域是毫秒級對毫秒級
  （§5 實測 ~2 ms/候選）— 誠實寫法：**軸論證是結構性的**（per-query ×N vs ×1、
  攤銷交叉點 Q* = C_train/c_extra；c_extra 毫秒級 ⇒ 本域 Q* 天文數字、BoN 在本域
  幾乎免費），**戲劇性版本屬一般敘事**（候選＝完整 LLM/robot rollout 的域）。T2 表
  （PLAN §1.2：我們 ms/plan vs ECD 8~25 s、C-MCTD 37~530 s）加一列 BoN@8 —
  仍在 ms 級、與秒級搜索系同格對比；GRPO 列保 1×。兩列都是「搜索買多少、RL 攤
  多少」三分帳（GRPO 卡 §3.4 與路線二分工）的實測面。

---

## 5. 成本

- **免訓練、eval 級算力、CPU 可跑**。實測錨（rung0 report）：三顆 ckpt 全探針 7.2 s
  CPU（zeldajr、8 threads）；headroom 段 64 題×G=16 ≈ 2 s/顆 ⇒ **~2 ms／候選**
  （抽樣＋decode＋打分）。BFS 場 per-goal 一次 O(格數)、`_BFS_CACHE` 快取 —
  N 條與跨題共用、近乎免費（GRPO 卡 §2.4 同帳）。
- **N=8 單 eval 時間倍率**：
  - plan-proxy eval（無 rollout）：×8 於毫秒級步驟 ⇒ 64 題約 +1 s CPU — 絕對值可忽略；
  - 完整 R0 eval：只有選出那條進 executor ⇒ rollout 端 ×1、plan 端 ×8 ⇒
    倍率 ≈ 1 + 7·t_plan/t_eval1；t_plan≈2 ms 實測、t_eval1（單題 rollout）未實測 ⇒
    **估 ×1.0~1.2【猜測；界＝1×（rollout 主導）到 8×（純 plan-proxy、絕對值仍秒級）】**，
    c 格順手實測替換。
- N=32（scan 上限）：plan 端 64 題 ≈ +4 s CPU — scan 整格仍分鐘級。

---

## 6. 風險與邊界

### 6.1 驗證器表達力邊界（哪些格會失真）

M_V 查得動的**以佔據圖語言能表達者為限**：逐點合法性、端點 BFS 可達性、C8 長度/步距
sanity。查不動的：動力學可行性、executor 可跟隨性（幾何之外）、路徑效率（合法繞遠路
r_reach 只看端點 ⇒ 滿分）。⇒ 失真格＝「V 眼裡好、executor 眼裡爛」— plan 層增益高估
R0 增益；錶＝§3.3 岔 3 的 proxy-gap 判決格。另 E 圖是資料建的（free=491/1295）：
資料覆蓋薄處 V 自己就是錯的 — L2(ii) 存量帽是實帽不是客套（⑤'' 語意接受此 tradeoff，
本卡不修；rung0 route 無路 fallback 0 題＝dev 集上目前沒咬）。

### 6.2 盲點選入（BoN 的 hacking 同族、無梯度版）

無訓練 ⇒ 不會「學會」刷分，但 argmax over N 會**系統性選進**驗證器的未知盲點、
且選入率隨 N 升。已知洞已閘（N3 教訓：短計畫刷分 U1、teleport U2 — 單元測 r=0）；
未知洞的錶＝**R0-vs-plan gap 對 N 的斜率**：plan 增益隨 N 升而 R0 增益平/降 ⇒
正在選進盲點（§3.3 岔 4）。C8 閘＋w 支配條件（w3≥w1）是設計內的擋法。

### 6.3 zero 模式語義（⛔ 明寫、防兩個混淆）

- **BoN-zero ＝ 自生 proposal × 外部驗證器**：ι=0、z 純從 prior 自生（(ii) 通道全關）、
  bits 只經選擇 J 進場 ≤ log N。這**不與「自生零資訊」矛盾** — L1(b)＋L2(i) 的分解
  正是這句話：z 們 0 bits、J 帶 ≤ log N bits。這是部署誠實格＝主 claim 格
  （鏡像 GRPO 卡 R-zero 主臂）。
- **⛔ 禁用孿生**：分數換成 flow 自己的 log_prob、或任何 θ-可測 critic ⇒ L2(iii)
  自驗證＝0 新 bits — 那是另一個臂（提取赤字改善）、⛔ 不准與本臂同名同表。
- zero 模式 prior 較弱 ⇒ q_deg 可能高於 on 模式實測的 30~34%、有效注入折扣更重 —
  V0 順手量、⛔ 不先猜。
- 附註：BoN 輸出分佈已非 π_base（KL ≤ log N − (N−1)/N）— 任何下游假設
  「計畫 ~ π_base」的檢查（密度型 OOD 等）語義跟著變，報表標 BoN@N 即為此。

### 6.4 題集與儀器紀律（沿 rung0 原樣攜帶）

- dev 題集 47% 為 L=0 平凡題 ⇒ 全集 headroom 被稀釋；難題切片只當診斷（⛔ 不進判準）；
  stitch/teleport 難題集是真靶（GRPO 卡 §5.1 風險 3 的藥）— 後續格、本卡不外推。
- C8 校準 per-ckpt per-day per 機、⛔ 只在 roundtrip-decode 空間；U1–U5 沒全綠不讀數。
- 多重比較：primary 一格釘死（§3.3 岔 3）、其餘 diagnostic — 格子多就是誘惑多。

---

## 7. 時鐘定位（呈裁）

免訓練 ⇒ **技術上可排 9/18 abstract 前當附加格**：a/b 格 CPU 分鐘級、c 格 eval 節點
批次 — 與 GRPO 卡 §5.3「rung 0/1 塞日間零碎/P4」同足跡等級，⛔ 不佔 P0–P2 判決鏈、
不搶 eval 節點雙 eval 隊列（PLAN §4 鐵則）。**但排程屬呈裁事項、卡上不寫死**：

1. a（V0 zero headroom）＋b（N-scan）是否准排 9/18 前零碎時段 — 順手且餵 rung1 判準；
2. c（R0 錶）eval 節點批次的佇列位 — 建議 P4、與 F3/T2 效率儀器化（PLAN D2/D11）併批；
3. BoN@8 是否作為 T2 效率表一列進 abstract 證據包、或留 9/25 full／discussion。

---

## 附：引用清單（全 repo 內；外部件標攜帶來源）

| 件 | 用在 | 級別 |
|---|---|---|
| THEORY-0906-information-ledger §0/§2.1/§2.2/§2.4/§4/§5/§6 | 資訊帳全節（L0/L1b/L2(i)-(vi)/L4/L4.5、對應表 row 1、匯率紀律） | 定理級（證在該檔）＋Remark |
| DESIGN-0906-grpo-thoughts §1.2/§2.1/§2.2/§2.4/§3.1/§3.3/§3.4/§4/§5.1/§5.3 | reward 三項＋C8 閘、hacking 表、成本帳、generator 加固、指紋、獎品句、rung 形式、proxy 風險、時鐘慣例 | 設計卡 v0 |
| experiments/probe_grpo_headroom_report.txt | 全部數字錨（§3.2 表）、C8 roundtrip-decode 校準、U1–U5、預釘判準線、q_deg、模式註記 | 實測（2026-09-06、zeldajr CPU） |
| experiments/probe_grpo_headroom.py（唯讀） | GrpoReward API（calibrate/dist_from/score/score_terms）、_BFS_CACHE、no-refine 紀律、zero 不外推註記（:75、:78-79） | 儀器現況 |
| THEORY-0906-verifier-taught-latent-thinking §1(6)(7)/§3 預測 6·11 | transient/persistent 兩半、溫度座標攜帶、rung0 判準與 GRPO 指紋 | 敘事（只織不添） |
| THEORY-0905-composition-law R2、Lemma 1/2 | BoN＝壓有效溫度；bits→R0 匯率斷 | 經敘事 §1(7) 攜帶 |
| NOTE-0906-context-taxonomy CT-1 | 自生零資訊 | 經帳本攜帶 |
| FINDINGS-0905 ⑤''/⑰/⑱' | E 圖合法部署語意與 R0 口徑／2.5 bits／base 未收斂＋齊漲指紋 | 經 GRPO 卡＋帳本攜帶 |
| PLAN-0906-forward §0/§1.2/§4 | 9/18/9/25 時鐘、T2 效率表、P0–P4 佇列與 eval 節點鐵則 | 計畫（待裁） |
| 2401.01879 Thm 3.1／2404.01730 §4.2 | KL 傾斜 ≤ log N−(N−1)/N 與可達性 | 經帳本 §2.2(v) 攜帶【隊友正文級】 |
| 2502.12118 §5.2 | 驗證器分離（正向錨） | 經帳本 §2.2(vi) 攜帶【隊友正文級】 |
| 2412.01951／2504.13837 | 自驗證防火牆（本臂不適用面） | 經帳本 L4.5 攜帶【隊友正文級】 |
| 2606.06447 NF-CoT | pass@k-vs-k 診斷（獎品量測形） | 經 GRPO 卡攜帶（lab 正文核驗） |
| 2502.07202 MCTD | 最近鄰 test-time search（novelty 站位：無內化） | 經帳本 §4 攜帶 |

_設計卡完。呈裁點＝§7 三件＋一件設計裁量：w=(1/3,1/3,1/3) 釘 rung0 實跑值
（支配引理條件 w3 ≥ w1 升格為約束）是否照准。_
