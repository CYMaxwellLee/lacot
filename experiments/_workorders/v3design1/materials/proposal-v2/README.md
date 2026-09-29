# v2 交件索引

u 編碼想像軌跡。這是待 lead／主人裁的返工提案；已完成設計、CPU 玩具證據及考場，尚未修改／合併 lacot production。

1. [ARCH.md](ARCH.md)：三候選、主案取捨、四模組所有權、零等待、四判準自檢。
2. [contracts/CONTRACT.md](contracts/CONTRACT.md)、[FAKE-CERT.md](contracts/FAKE-CERT.md)：v2 接口／梯度／dtype／R0／教師與 gauge 契約、自證。
3. 四個獨立工單：[M1](mod-objective/TASK.md)、[M2](mod-actor/TASK.md)、[M3](mod-verification/TASK.md)、[M4](mod-oracle/TASK.md)，各附 solution stub、neighbors、contracts snapshot、tests。
4. [integration/](integration/)、[EVIDENCE.md](EVIDENCE.md)：親跑 wiring PASS、current mainline top-level FAIL、reference CPU訓練、six mutants、R2；[施工草圖](integration/WIRING-SKETCH.md)。
5. [ROUTING.md](ROUTING.md)：四級路由建議，未派工。

另附必交 [CHANGE-ORDER.md](CHANGE-ORDER.md)：v1→v2契約變更、blast radius、沿用與重考範圍；[判決逐條回應](REVIEW-RESPONSE.md)。

重跑（只寫本目錄、CPU）：

```sh
/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/run_evidence.py
```

頂層 `integration/integration_test.py` 不規定consistency公式；`consistency_test.py` 是待裁 EMA 主案專屬，mutations.py 是v2接線考場。current_mainline 的 exit=1 是預期**語意 FAIL**並保存原輸出，不被洗成PASS。

限制：reference 最終小MLP加toy decoder sign feedback才全seed通過；無feedback的兩次失敗與真RefineOperator額外診斷FAIL原樣留存。因此尚不能聲稱production RefineOperator、真maze教師或真部署已修好。沒有GPU／真訓練；8份唯讀關鍵源檔前後hash一致。模組self-check只驗fixture，worker stub刻意失敗；裁後仍須真三入口合併gate。

STATUS: DONE
VERIFIED: 五件套＋變更單、wiring PASS、現行主線頂層 FAIL、CPU reference／six mutations／B1／EMA／R2／M3獨立考場親跑；詳見EVIDENCE。
