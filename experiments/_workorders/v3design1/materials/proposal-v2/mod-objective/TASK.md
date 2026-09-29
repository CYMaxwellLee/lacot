# M1：generated-plan 目標

u 編碼想像軌跡。負責新 lacot/refine_objective.py 與對應契約測試，L3；不碰 scratch/model/oracle。

讀本地 C1–C4，從 solution.py 填 refinement_terms；同 sample 主臂及 sample+jitter 輔臂逐輪 quality，EMA 按待裁 C4，保留 cond/full BPTT。R0、autocast mixed dtype、nonfinite propagation 必過；不再要求 cond dtype=clean，也不 hot-path finite raise。

neighbors.py 和 contracts 都在本地，不等 M2/M4。現有 fake 是 executable spec，不能複製其小玩具限制冒充 production oracle；production quality 由 callback 提供。teacher 停梯度、loss graph 保留、無 action/data target 接口。

獨立命令（本目錄）：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B tests.py --self-check`；交件必另跑 `tests.py --candidate solution.py`。回交 patch、候選 PASS、API hash、gradient 與 dtype/AMP 範例。裁案前不派工。
