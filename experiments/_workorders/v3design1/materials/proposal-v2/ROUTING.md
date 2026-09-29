# 四級路由建議（不派工）

u 編碼想像軌跡。沿用 v1 明示尺度：L1 機械修改，L2 邊界明確工程，L3 多梯度／數值契約，L4 對抗科學與跨域主線整合。不是自行指定模型或越權裁案。

| 模組 | 檔位 | 主要難點 | 起跑 |
|---|---|---|---|
| M1 objective | L3 | full BPTT、teacher detach、cond/autocast、R0 | t=0 local analytic oracle |
| M2 actor/mainline | L4 | scratch 三語意空間、主線各 loss、EMA/AMP/resume、兩 actor | t=0 certified objective/oracle fake |
| M3 verification | L4 | B1/off-manifold、多 seed訓練、mutation、baseline、公平歸因 | t=0 local harness＋唯讀真 baseline |
| M4 oracle | L4 | frozen decoder OOD、同 sample action teacher、外部 validity | t=0 synthetic plans與既有 decoder/geometry，只讀 caller |

四 worker 零等待並行；合體需接口/teacher 認證，並非前置排程。採案與實際檔位由 lead／主人決定。
