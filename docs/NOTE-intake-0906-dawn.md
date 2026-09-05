# NOTE — 9/6 凌晨加派四隻的收件清單（寫給早上的我；補 handoff ②-3.5 之後的新事件）

_00:13 主人接住重啟＋恢復記憶（TG 5663-5665）→ 00:19 授權加派（「離四點還有時間」5668）→
00:24 新定調句（5671：「學到的不是info而是function」）→ 00:26 前四隻全部出擊 → 01:00 前全交。
產物【全部未 commit】、等ルナ驗收後才入庫。主人已就寢（00:25「快跑來抱著睡」）。_

## 四隻產物與收件動作

1. **BoN 設計卡** `docs/DESIGN-explore-verify-bon.md`（283 行）
   - 細讀驗收；複核「閘內命中支配引理」（它使 plan 層實現值被 rung0 數字釘死、新實驗縮成 zero headroom＋R0 兌現率兩格）。
   - 呈裁×4：a+b 排不排 9/18 前／c 的 eval 佇列位（它建議 P4 與 F3/T2 併批）／BoN@8 進不進 abstract T2 表／w=(1/3,1/3,1/3) 釘 rung0 實跑值照不照准。

2. **深審官** `docs/REVIEW-theory-audit-0906.md`（致命 2＋重要 9＋建議 11＋穩件 10＋需回原文 6 組）
   - 四級複判（採納/修訂/降級/駁回）逐條；⛔ 它是假設清單不是判決書。
   - 致命 2 條先看：F1＝R0 溫度座標跨檔矛盾（合成律判 T=1、帳本/CT/敘事寫 T→0、已進英文 claim 段）；F2＝sharpening 防火牆條件變數錯位（該用 I(W;M_V|s,g,θ)、檔內判準會給自家錯誤安心、真防火牆其實是假設而 rung0 正是其操作量測）。
   - ⚠️ M1 與 rung0 判讀直接相關：「無 headroom ⇒ 幫浦無輸入」在 dense reward 下有反例 — 收 GRPO 相關格前帶著這條、照帳本壓縮句讀會錯砍臂。
   - 🚨 時效件 M6：**收表前先 `ls slurm/logs/IDP005-*`** 對 idp005 批實跑狀態（敘事檔「無臂」vs 夜審「已在跑」矛盾）。
   - M2：A1 舊敘事殘留在周邊檔（PAPER-SKELETON 六處、DESIGN-multiroute 病根段、PLAN L140）— 排清理。

3. **乙檔 v2** `docs/THEORY-2026-09-05-internalization-formal.md`（+152/−64）＋ `docs/REVISION-NOTES-internalization-v2.md`
   - diff 驗收後 commit；pre-registered 檔（postA1）確認未動。
   - ⚠️ 它自行採納了深審 M3 一條（postA1 Prop 2.1′ 證明橋、它獨立重推同意才吃）— 驗收時複核這條。
   - TODO(驗收) 5 處；接口矛盾 4 條（postA1 勘誤註待補／帳本 §2.4 建議補 1-bit 拆分半句／合成律側丙 D4 互引未落／κT「債已還完」標疑維持）。

4. **info vs function** `docs/THEORY-2026-09-06-info-vs-function.md`（33.8KB、F1–F7、主人 5671 定調句逐字卷首）
   - 細讀驗收；關鍵件：Remark F1.5（單環境下 info/function 之刀數學上失明 — 帳本沒切不是疏漏）；Prop F2 記憶化缺口恆等式；Prop F5（GRPO group centering 打不進題級常數事實、附 reward 平移免費單元測）；FP-1 設計紀律（只跑 zero-shot 分不了兩假說、要關通道對照臂）。
   - 呈裁×3：**多圖訓練臂＝新資料軸**（與 multiroute 正交、現有資料不供給）／ξ 兌現效率錶進不進儀器清單／claim「general」拆 teacher-agnostic 與 environment-generic 兩軸。

## 收件順序建議（併 handoff ② 用）

M6 時效件（一行 ls）→ 大收表（handoff ②-1、判讀帶著深審 M1）→ 理論四件驗收（先深審複判、再乙檔 diff、再兩張新卡）→ 呈裁 batch（handoff ②-3 六件＋這波新增 7 件）一次呈主人。

_另：昨晚施工隻 smoke 親跑（handoff ②-3.5d）仍在清單上、未動。_
