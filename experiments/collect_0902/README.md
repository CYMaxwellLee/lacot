# 9/2 收表腳本（全部 cwd 會切到 ~/Projects/lacot；讀 results/night_0902/*）
- collect_0902.py A|B — V8 失敗地理（死點密度分位、口袋/密區）與 dz2 重現
- collect_ebfs_c.py — ebfs 分辨器 ＋ C 批（方差溯源、舊配方）
- collect_esel.py — E 選計畫 esel16
- collect_d.py — D 批（定版配方方差拆分）
- collect_uora.py — oracle-u 探針
- collect_emb.py — embedding 批（長 stage1 / VQ64）＋ rt_gate 往返尺
- collect_gate.py — 往返尺校準（V8 八顆）
- collect_flowprobe.py — flow 探針第一版校準；第二版（進度）用 collect_flowprobe 的邏輯改 flowprobe2_v8/
- collect_snap.py — 路標吸附（snap_ma2 / snap_ma1）
- collect_guard.py — 守門批（guard_HG / guard_DA / guard_HGS）
⏳ 9/3 早上要新寫：dstart_{hard,soft}（對 V8）、dshard_{HG,DA}、heldout2_{base,hard}、medium_hard —— 照 collect_emb.py / collect_guard.py 的樣子改目錄即可。
