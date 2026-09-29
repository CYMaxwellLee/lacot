# M2 接線施工草圖（不是已合併 patch）

u 編碼想像軌跡。以下固定 features/clean 表示的分工，全部 scratch hunk 歸 M2。主線三種 clean 不合併成一顆含糊 u_target。

```python
# Keep existing density branch 1695–1703, _REAL_W and _ca logic.
# C6 expects density() already normalized by its OWN space dimension.
services = make_refine_services(decoder=_dec, undo_intent=_intent_inv,
                               s=s, g=g, anchors=anc, frozen_quality=quality_model,
                               frozen_action_teacher=action_teacher)
adapter = Adapter(
    density=lambda: l_nf,  # _et_nf uses DIM or K*fsq.d exactly as before
    features=lambda clean: (_ca, clean),
    anchor=lambda features: _wmse(ahead(*features), act),
    sample=lambda: flow.sample(B, flow_cond(cond, anc)),
    exposure=lambda u: services.head_exposure(ahead, cond, u),
)
# head_clean is _u_head (not _et_nf); sample returns deployment u, including z->up.
# draw noise using a dedicated stream and same latent dtype; never use actions here.
base_total, logs, states = compose(
    adapter, refine, refine_ema, services.quality_factory, cond, _u_head,
    rounds=R_TRAIN if LEARNED_REFINE else 0,
    noise=refine_noise, lam_cons=CONS_WEIGHT,
)
# Replace 1779, do not add l_nf/l_anchor again.
total = base_total + (0.0 * l_bc if BC_INDEP else l_bc)
# Existing DIV_W/GRPO_W additions continue unchanged outside this sketch.
```

`quality_factory(sampled,cond)` 在 sample 抽出後才绑定 initial mode/support context；避免對另一條 sample 的 target 閉包。只有已驗證 representations 能進 compose，provider 不接受「coordinates 就是 action」的捷徑。inference hard quantization 与 training proxy 的偏差由M4/M3驗，不用改 `_q` 來偷過。

model.py state adapter features=`cat([cond,clean.flatten(1)],-1)`；image adapter=`clean.flatten(1)`。anchor 同原 head.nll，density 同原 flow.nll/(K*D)。每個 actor 都調用 compose，R>0 provider 由可選 training services 提供；R0 在任何 provider 建構/讀屬性前短路，確保 R2 的 SimpleNamespace fixture 仍能跑。

provider 初始化在首次 `_stage2_loop(STEPS2)` 前；resume 載入完成後 restore provider fingerprint 與 refine_ema 再跑 `_stage2_loop(CONT_STEPS)`。舊 eval GEO 段改同 builder/reuse instance，禁止訓練與 eval 另造两張不同牆圖。opt2 梯度有限、成功更新才 EMA；不得因 opt_bc skip 与否误判 opt2，需個別optimizer skip fixture。

三入口 spy gate：在可寫合併 checkout 給 compose 加測試 spy（測試依賴注入，無 production fallback），各入口一次 call；回放同 clean/sample/noise，數值及梯度等式成立；分別使用state/image/scratch的原features。主線另用非均勻_REAL_W、COND_DROP fixed mask、FSQ z≠u width、intent per-token cond 案例，total不重加。
