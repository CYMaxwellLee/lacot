# PLAN — 2026-09-06 forward（戰略規劃使魔 v1、全文待主人裁）

_任務：把「內化 paper」從三缺推到投稿證據包。唯讀產出；不含任何已執行的變更。_
_輸入：FINDINGS-0905 全、DESIGN-0905 兩份、RELATED-WORK-0905、THEORY-0905 兩份、NOTE-composition-law、FINDINGS-0904、當下 queue 狀態（idp01 4/8 已收＋4 跑、idpxm 排隊中）。_

---

## 0. 時鐘（本計畫唯一新增的外部事實）

**ICLR 2027：abstract 9/18、full paper 9/25（AoE）**〔web 查證 iclr.cc/Conferences/2027/CallForPapers、mldeadlines.com 同值；會期 2027-04-26~30 加州〕。

⇒ 今天是 D0。**14 天計畫的終點＝abstract deadline**；到 full 還有 +6 天 buffer。
⇒ 結構性後果兩條：
1. **paper 落筆不是尾節點、是平行軌**——method＋theory 節 D4 就開工（不依賴新實驗的部分先寫）。
2. **D7（9/12）設 go/no-go 檢查點**：「搖籃 Int>0 立住」＋「ant 資料在手」二者缺一 ⇒ 當天跟主人談轉場（下一場 venue，時間全數投高維）。⛔ 這個分岔是主人的裁決，本計畫只把判準預釘。

---

## 1. 勝利條件反推：ICLR 最小完整證據包

### 1.1 claim（NOTE §三已有，一句話）

> 訓練時查得到的路線知識（BFS oracle 或 hindsight 摘要）可攤銷進 flow planner 的條件 prior；合成律讓 BFS＝溫度族的凍結極限（T→0）、字典搜索＝一般化；內化程度由 intent-dropout 錶直接可量（Int 定義＋(Int, ε) 診斷對）。

敘事站位（對照 RELATED-WORK 競品格局、⑥ 空地修正後）：**「同格誠實對標＋免搜索快 3~4 個量級＋內化度量獨有」**取代「沒人做過 humanoid stitch」。內化車道（訓練查／推論免查＋量化殘留）三隻 sweep 獨立確認無先占；ECD/CD/CDGS/GSC 全是秒級推論期搜索／校正系、無 latent 無內化 ⇒ 我們不在它們那一欄競分數、在自己那一欄立度量。

### 1.2 表格清單（投稿最小集；標 ★＝缺了就掉回 maze study）

```
T1 ★ 主結果表（內化主 claim；每 env 三臂×8 顆、雙 eval）
   列：env ∈ {pointmaze-{medium,large}-stitch ★, antmaze-{medium 或 large}-stitch ★,
              humanoidmaze-large-stitch（stretch、遲到就標 partial）}
   欄（每 env）：base（無 intent）／ref（p=0 全曝光、f27n 形）／idp-best（勝出藥方）
   指標：subgoal raw、R0 帶查(map)、R0 免查、配對差、Int（Def 1.4）、ε_rel（⑬ 探針）
   teacher 軸小欄：route vs hindsight（O-agnostic；搖籃已有 ER .918≈hindsight .928 兩點）
   協定：⑤''（內化只在 R0 報、subgoal 內化欄 undefined、配對差主指標、⛔不寫 oracle）
T2 ★ 效率表：我們 ms/plan（要自己實測、儀器化）vs ECD 8~25s、C-MCTD 37~530s、
   ChronoForest 91.9s；同格分數誠實並列（humanoid 格 ECD 64±4、TMD 23.0）
T3 ★ 字典驗收＋合成表（缺②本體）：D1 utilization／D2 pairability（f27n 2×2 已是實錘）
   ／D3 round-trip+interp 三關讀數；字典 DP（Remark R4 形 (a)）vs 連續 intent 對照
T4   劑量–反應表（文獻空白素材）：p ∈ {0, 0.1, 0.3, 0.5}×8 的 (Int, ε) 曲線
   ＋鎖死探針（B/A=1.1%）＋guidance 無效（w 掃）＋warm-start 判決 — Conj 2.6 的實驗面
T5   幾何表（路線一）：before(.705/.212)→after、三 loss ablation、d_time↔d_bfs 交叉評
附錄：FSQ 失敗解剖 2×2＋fsq 全變體帳（③）＝「recon 好≠可學」方法學；zero 探針
   儀器無效判定；per-seed 全表；SVA 對比（同權重推論期開關 vs 重訓 — 我方更乾淨）
```

### 1.3 理論節定理清單（兩份 THEORY 已有 v0、Rei 磨嚴）

- 合成律：Lemma 1/2（log-semiring T=1、BFS＝T→0 凍結極限、量化差距 ≤HT·logK）＋Prop 3（定點迭代＝BFS）＋ D1–D3 假設×可量代理 ＋ Prop 6／Conj 7（字典 DP 恢復最優／學到字典的近似版）。⛔「generalizes BFS」只准按定理級／Conjecture 級／Open 三層拆開陳述。
- 內化：Def 1.4（Int＋(Int,ε) 診斷對）＋Prop 1.1–1.3（ε→W₂→Δsucc 望遠鏡＋預算恆等式）＋Prop 2.1／Cor 2.2／Prop 2.3（鎖死＝合法全域最優＋guidance 無效＋穩定駐點）＋Conj 2.6（p 臨界值）＋Prop 3.1/3.4/3.7（三藥打三層：動力學／結構在場／資訊供給）。
- 差異化引用已釘：OKBE 帶限定詞當旁證；TMD 零插值量測、無重疊；HDFlow 五軸差異表；DAPD「privilege illusion」由 idp 零錨 eval 正面回答。

### 1.4 三缺 → 最小達標線

```
缺①高維實證   最小＝antmaze stitch 一格內化線重現（Int>0）；stretch＝humanoid 誠實對標 64±4
缺②字典本體   最小＝intent 層小字典過 D1–D3 驗收＋字典 DP 在 stitch 上 ≥ 連續臂−ε
              ＋合成出訓練沒見過的長路的直接證據（哪怕 medium 一格）
缺③內化數字   最小＝搖籃 (Int, ε) 立住（分母≥κ·SE、8 顆配對、R0-200ep 硬化）
              ＋T4 劑量曲線；強化＝多路線 (s,g) 設定下的 Int（⑫⭐：maze 冗餘壓低
              intent 價值 ⇒ 內化度量要立得漂亮、多路線設定可能是必要條件而非對沖）
```

---

## 2. 7~14 天計畫（D1=9/6 … D14=9/19；+buffer 至 9/25）

六條主線。每項：〔補缺｜前置｜算力｜判準〕＋⓪自檢（align 大方向？推進還是轉圈？證明什麼？）。

### 主線 A：內化線（缺③；本週主戰場）

- **A1 劑量二臂判決**〔缺③｜前置：idp01 8/8＋idpxm 收齊（今晨）｜0（已付）｜判讀樹 ⑪：idp01 R0on 回 .42+ ⇒ 增益回、直接讀零模式殘留；idpxm 回 ⇒ 病=曝光量；皆不回 ⇒ dropout 結構妨礙 ⇒ 訓練期藥〕⓪三枝都推進：各自指定下一步藥，不轉圈。
- **A2 warm-start 半天測**〔缺③｜前置：stage2 續訓小 patch（裁示中）｜~2 GPU-h｜f27n 續訓 p=0.3 數千步：增益黏住 ⇒ 退火可行；退回 ⇒ p=0.3 是另一吸子、需常駐 L_div（Prop 3.1/Conj 3.2 的實驗面）〕⓪這是「退火 vs 常駐」的分岔判官、藥方選擇的前置。
- **A3 藥方臂**〔缺③｜前置：A1×A2 枝｜每臂 8 顆 ≈3.2 GPU-h 訓＋eval｜臂庫（⑭ 定）：退火 p schedule／L_div floor（Def 3.3、margin 錨 f27n cond 層差 .6046）／雙保險。判準：R0on ≥ .42 且 Int 分母 ≥ κ·SE 且 ε_rel 脫離 1% 級〕⓪每臂對應理論節一條命題 — 贏了是結果、輸了是 T4 的機制格，皆非白跑。
- **A4 Int 錶正式數字**〔缺③收官｜前置：勝出臂｜R0 ep≥200 × 8+8 雙 eval（eval 節點）｜⑤'' 全協定＋per-seed 全表＋8v8 分佈對照；產出 T1 搖籃欄＋(Int,ε) 落格〕⓪這就是「內化可量」四個字的現金。
- **A5 多路線 (s,g) 設定（根治＋強化缺③）**〔缺③（可能是必要條件）｜前置：eval-set 設計（teleport／counterfactual 構造、C-i~iii 條件對表）｜資料生成 CPU＋pilot 1 顆＋×8 ≈6 GPU-h｜判準：C-i 的 H(R|s,g)>0 先在資料上量到、再看 Int 是否站起來（Prop 3.7 預測：兩根支柱同拆）〕⓪與 stitch 本義合流 — 不是繞路、是把「組合出沒見過的路」做成可量設定。
- **A6 T4 補點**〔缺③素材｜前置：無｜p=0.5×8 ≈3.2 GPU-h＋teacher 光譜第三點（劣化 teacher）×1 顆｜劑量曲線第四點＋O-agnostic 第三點〕⓪便宜、直接變 paper 圖。

### 主線 B：路線一 v2 階梯（缺③支撐＋高維可攜；T5）

- **B0 E-canon 補讀**〔novelty 防線｜前置：無｜0（使魔 fan-out）｜QRL/MRN/IQE/CRL/SoRB/HIQL 升【正】——RELATED-WORK E 類【缺】、路線一動筆前必補〕⓪防 reviewer 一刀。
- **B1 patch v2**〔前置：主人核 v2 規格（DESIGN-route1 已定形）｜0（CPU smoke）｜子空間拆分 e_m‖e_d、三 loss env 開關 default 0＝zero-diff golden、C8 非退化欄進探針〕
- **B2 rung1 單顆 stage1**〔~2 GPU-h【猜測：stage1 訓一顆 1~2h、docs 未載明】｜判準：插值合法率 > .757（贏亂猜）、rho 交叉形式評 ≥.35、recon gate〕
- **B3 rung2 疊 stage2 單顆**〔0.4 GPU-h｜subgoal/R0 不退步 gate（張力錶：幾何搶容量 vs R0 卸貨）〕
- **B4 rung3 ×8**〔≈16 GPU-h（stage1×8＋stage2×8）｜T5 成表〕
  ⓪整梯自檢：任一 gate 掛＝回設計，⛔不硬調權重刷過；此線是「幾何買多少、搜索補多少」分帳的幾何端，輸了也是誠實 ablation。

### 主線 C：B 階段字典（缺②；★需主人開題——原順序「押後、等 Rei/主人開」，deadline 下建議提前，此為本計畫最大的順序變更提案）

- **C0 N1 補量**〔缺②前置｜3 行 probe patch 等主人核｜10 分鐘 CPU｜出貨 codebook × 真資料分佈的每維格點利用直方圖——「字典本身好不好」至今未驗，這格不量其他都是猜〕
- **C1 設計定稿**〔缺②｜前置：THEORY D1–D3＋⑥共識（字典住 intent 層、K 小、Hydra 64 同向）｜0｜產出：spec＋驗收關（D1 utilization／D2 配對可學性／D3 round-trip+interp）寫死在動工前〕⓪驗收關先於實作＝f27n 2×2 教訓的制度化。
- **C2 最小實作 pilot**〔缺②｜前置：C1＋A 線藥（intent 通路要活著才有東西可離散化）｜pilot 2 顆 ≈1 GPU-h｜intent latent 上小字典（k-means init 或 VQ）＋合成律 DP/beam（R4 形 (a)、從 ⊥ 起算最小定點語意）；先 medium-stitch〕
- **C3 放量判決**〔缺②收官｜前置：C2 過 D1–D3｜×8 ≈4 GPU-h｜判準：字典臂 ≥ 連續 intent 臂 −ε 且展示「組合出訓練沒見過的長路」至少一格直接證據〕⓪這格是「字典搜索一般化 BFS」從定理級落到實驗級的唯一通道。

### 主線 D：高維 A0（缺①；**長桿、D1 就要開**）

- **D1 資料活路三管齊發**〔缺①｜前置：無｜0｜①lab 內問（frieren 有人跑 HIQL——最快活路）②issue #49 +1＋Berkeley IT ③rail 低頻監控續掛。**同日請主人裁自訓 expert**（main_sac online-ant-xy；成本【猜測：單卡 1~2 天/env】——同分佈不同位元、寫 paper 時如實揭露）〕⓪裁了當晚就開：它是全計畫最長的單桿，晚一天開＝終點晚一天。
- **D2 ant 管線移植**〔缺①｜前置：資料到位（任一活路）｜stage1 ×2 顆＋stage2 pilot ≈6 GPU-h｜口徑五件套照 DESIGN-0904；teacher 只剩 hindsight ⇒ O-agnostic claim 被自然測試〕
- **D3 ant 內化線 ×8**〔缺①收官（最小版）｜前置：D2 pilot 方向對｜≈8 GPU-h｜判準：**Int>0 重現、非分數 SOTA**——「分數是搖籃裡的，能帶走的只有內化那條線」（主人 9/5 定調）〕
- **D4 humanoid（stretch）**〔缺①強化｜前置：D3 成＋D10 主人 go｜stage1＋×8 ≈20 GPU-h【猜測】｜誠實對標 ECD 64±4＋效率軸；遲到就進 buffer 週、再遲標 partial〕
  ⓪高維判準統一用「在高維還存在嗎」濾每個零件（teacher／subgoal 腿／字典）。

### 主線 E：理論 converge → Rei 交接

- **E1 洞清單定稿**〔理論節｜前置：兩份 THEORY v0（已有）｜0｜給 Rei 的洞：(S1) 分佈版強度、Conj 7 √δ 換算、Conj 2.6 toy model（線性 adapter＋二次 loss、Rei 出手點）、Def 1.4 與 KL 關係、定點唯一性條件、quasimetric loss 形＝路線一 v2 的理論面〕
- **E2 交接包**〔前置：E1＋B 線 rung1 有 ablation 雛形｜0｜theory ×2＋before 尺＋d 蒸餾資產＋v2 ablation；時點看 Rei 忙完（主人 9/5 裁：她忙完前ルナ先推）〕⓪交接記「為什麼」不只「做了什麼」。

### 主線 F：paper 落筆（平行軌）

- **F1 骨架＋claim 四軸重排定稿**〔前置：主人核 ⑥ 提案（④空地→內化度量軸）｜D1 提裁〕
- **F2 method＋theory 節初稿**〔D4 開工、D7 交檢查點——不依賴新實驗〕
- **F3 效率儀器化**〔T2 前置｜eval 節點半天｜我們的 ms/plan wall-clock 按可比協定量好記錄（對 ECD Table 6）〕⓪沒有自己的實測數字、效率軸只是口號。
- **F4 abstract（9/18）→ full（9/25）**〔D13 提交 abstract；buffer 週收 T1 高維格＋複驗＋完稿〕

### 日程格（依賴序；過夜批見 §4）

```
D1  9/6   A1 判決｜裁：warm-start/N1/A0自訓/四軸重排/R0-200/C1 提前｜A2 跑｜夜：A3+A6+D1(自訓)
D2  9/7   A2 判決→定藥｜B0 收｜A5 eval-set 設計｜F3 效率儀器
D3  9/8   A3 收→Int 初讀｜C0+C1｜B1 patch｜A6 teacher 第三點
D4  9/9   A4 硬化跑｜B2 rung1｜A5 資料生成｜F2 開工
D5  9/10  B2 判｜C2 pilot｜A5 pilot｜E1 洞清單
D6  9/11  B3｜C2 判(D1–D3)｜A5 判｜pointmaze-large 移植開跑
D7  9/12  ★檢查點①：搖籃證據包凍結 v0（T1 搖籃欄+T4+T5 雛形）＋ant 資料判＝go/no-go
D8  9/13  D2 ant stage1+pilot｜C3 ×8｜B4 ×8（夜）
D9  9/14  D2 判→D3 ×8（夜）｜pointmaze-large 收
D10 9/15  D3 收→ant Int｜humanoid go/no-go（主人裁）
D11 9/16  D4 stage1 或 ant-large 補格｜T2 效率表收
D12 9/17  D4 batch｜全表 ⑤'' 協定複驗＋per-seed
D13 9/18  ★abstract 提交｜缺格盤點＋buffer 計畫定
D14 9/19  證據包 v1 凍結｜writing 全力
buffer 9/20–25  humanoid 收尾（誠實欄）｜rebuttal 素材｜9/25 full 提交
```

---

## 3. 風險與對沖（每主線一條「爛掉怎麼辦」）

- **A 內化線：p 掃全爛**（A1 三枝走到「皆不回」＋A2 塌回）⇒ 藥梯順序：**L_div 先上**（結構藥、patch 最小、正中 ⑬ 量到的 cond 層塌格、Prop 3.4 保「可喚醒性」）→ 若 ε 保住而 Int 仍 0 ⇒ A1 假設（資訊冗餘）實錘 ⇒ **重心整個移 A5 多路線設定**——maze 冗餘本身升格為 finding（Prop 1.3 的實測版＋兩個文獻空白），Int 數字改在多路線／stitch 設定立。備援蒸餾梯（訓練期強迫用）：Context Distillation＋PDM 修正 → π-Distill 雙模式 → ReGuide。缺③不死、換地基。
- **B 路線一：階梯 gate 掛**（幾何傷 recon 或 R0）⇒ 降級為 negative-result ablation＋before 尺方法學（「latent 空間今天沒有測地線結構」本身是乾淨量測）；theory §4 留 conjecture；⛔不硬調權重、不擋其他線——paper 主幹不押它。
- **C 字典：intent 層字典也毒** ⇒ 退守「字典的科學」：D1–D3 驗收關＋2×2 失敗解剖＋per-token vs intent 層的層級選擇證據；claim 重心移「溫度族合成律＋內化度量」。⚠️ 這會把缺②弱化成方法學貢獻——**是否仍夠 ICLR 是主人的戰略裁決**，D6 判決日當場提。
- **D A0：rail 一直死＋lab 無人有** ⇒ 自訓 expert 是唯一保底、故 **D1 裁、當夜開**（長桿先開）；成本【猜測：單卡 1~2 天/env，SAC online】、同分佈不同位元要在 paper 揭露。最壞情境：高維只有 ant-medium ⇒ humanoid 降 discussion＋效率外推，缺①半殘——**全計畫最大單點風險**，對沖＝三管齊發＋最早開跑＋D7 檢查點強制攤牌。
- **E/D4 humanoid 內化線打不到 ECD 64** ⇒ 敘事站位：**不同欄競爭**——ECD 系全是秒級推論期搜索／校正、無訓練查/推論免查之分、Int/ε 對它們甚至無定義；我們給同格誠實數字＋Pareto（score × latency × search-free）＋獨有度量欄。⛔不 claim 分數 SOTA；賣的是新軸、內化車道查無先占（三隻 sweep 獨立確認）。
- **F 時程：9/18 abstract 證據不齊**（D7 判準任一缺）⇒ 轉場 fallback（下一場 venue【猜測：ICML 2027 一月】），多買的時間全投高維＋字典；判準已預釘在 D7、⛔不拖到 D13 才發現。

---

## 4. 算力預算（4 訓練＋1 eval；24 分/顆 stage2；stage1 【猜測 1~2 h/顆】）

可用量（粗）：4 節點 × 14 天 × ~16 h（過夜 12h＋日間零碎）≈ **900 GPU-h**。
計畫用量（粗排、含 20% 重跑餘裕）：

```
A 內化線   藥方 3 臂×8＋p=0.5×8＋warm-start＋A5 pilot/×8＋teacher 點   ~35 GPU-h
B 路線一   rung1/2 單顆＋rung3 ×8（stage1 重）                         ~25 GPU-h
C 字典     N1(CPU)＋pilot＋×8                                          ~ 8 GPU-h
D 高維     expert 自訓（若裁）~50–100【猜測】＋ant stage1/2＋×8        ~100 GPU-h
D4 stretch humanoid stage1＋×8【猜測】                                  ~40 GPU-h
複製格     pointmaze-large 三臂×8                                       ~15 GPU-h
eval 節點  雙 eval 常備＋R0-200ep 硬化（決賽臂專用）＋F3 效率儀器      （獨立、不佔訓練）
──────────────────────────────────────────────  合計 ~250–320 GPU-h ≈ 可用量 1/3
```

⇒ **瓶頸不是 GPU-h、是判決串行**（每個過夜批要等前一個判決選臂）。因此過夜批優先序固定為：
**P0 內化判決鏈（A1→A3→A4）＞ P1 expert 自訓（長桿、一開就常駐一節點）＞ P2 路線一階梯 ＞ P3 字典 pilot ＞ P4 補點（p=0.5、teacher 光譜、複製格）**。
日間零碎塞 pilot 與 smoke；eval 節點永遠留給雙 eval 隊列＋硬化批，⛔訓練不搶 eval 節點。

---

## 5. 明天（9/6）的具體牌

**早：收與裁（不開新火）**
- 收劑量二臂（idp01 8/8＋idpxm 1/1、雙 eval 配對）→ 走 ⑪ 判讀樹、當場定藥方枝。
- 一次呈裁六件：①warm-start patch（半天測）②N1 補量 probe（10 分鐘）③A0 自訓 expert（成本估＋同分佈揭露條款）④claim 四軸重排（⑥ 提案）⑤R0-200ep 協定＋p=0.5 補點 ⑥C 線提前開題（順序變更提案）。（deadline 事實一併報：9/18/9/25。）

**中：分岔執行**
- warm-start patch＋跑（若核）；A5 多路線 eval-set 設計落 spec；B0 E-canon fan-out 派出；F3 效率儀器在 eval 節點排上。
- 若 A1 走「增益回」枝（idp01 R0on≥.42）：直接排 idp01 的 R0-200ep 硬化——那就是第一個正式 Int 數字，其他藥方降為 T4 機制格。

**晚：過夜批（按 §4 優先序）**
- P0：藥方臂（枝定哪臂上哪臂；L_div 臂在任何枝都有資訊量、預設在列）＋p=0.5×8 補點。
- P1：expert 自訓開跑（若裁）——長桿今晚不開、D8 的 ant 就沒資料保底。
- P2：B1 patch smoke（CPU、zero-diff golden）過了才排 rung1。
- 裁而未跑的一律記錄在案、⛔不搶跑（做實驗前先問主人——這條是鐵的）。

---

_⓪ 全計畫自檢：每一項都掛在三缺之一或其直接前置上；沒有一項是「因為便宜所以跑」。最不確定的三格已標【猜測】：stage1 訓練時長、expert 自訓成本、humanoid 總帳——D1/D2 用實測數字替換。_
