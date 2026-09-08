"""共用模組 —— humanoidmaze 版「照譜跳」驗證（段獨立重放 ＋ 教材接力）。

對應 docs/NOTE-2026-09-08-humanoid-relay.md。方法論「形狀」照抄
experiments/walk_verify/p1_replay.py 與 experiments/walk_verify/teacher_relay/run_teacher_relay.py
（ant 版，見兩份 note：NOTE-2026-09-08-walk-verify.md §二、NOTE-2026-09-08-teacher-relay.md），
換成 humanoidmaze-medium-stitch-v0 資料 ＋ experiments/humanoid_dict/ 的
FactorizedCondVQVAE 字典（G16K16 / G1K32，皆 seg_len=4，ckpt 已由使魔訓好，本次不訓練）。

⛔ 只 import 同目錄上一層的 hd_common.py（沿用 humanoid_dict 既有紀律：不跨 ant 的
walk_verify/ 目錄 import，見 hd_common.py 開頭 docstring 第 13 行）。ant 版三支檔案
（wv_common / p1_replay / p2_analyze_traces）只作方法論參考，程式碼不 import。
⛔ 不修改 hd_common.py 或任何既有檔案。
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HD_DIR = os.path.dirname(HERE)  # experiments/humanoid_dict
sys.path.insert(0, HD_DIR)
import hd_common as hc  # noqa: E402

DD_DEFAULT = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
CKPT_DIR = os.path.join(HD_DIR, "ckpt")
RULER_DEFAULT = os.path.join(HD_DIR, "results", "ruler_pack.json")

# ------------------------------------------------------------------
# DELTA_SUB / rho：沿用 ant 數值，不重新校 —— 理由見下方長註解
# ------------------------------------------------------------------
RHO_DEFAULT = 1.875   # = 0.25*DELTA_SUB，跟 ant 的 lacot/subgoal.py:230 同定義、同數值
DELTA_SUB = 7.5        # 跟 ant（walk_verify/teacher_relay/run_teacher_relay.py）完全同值

# 從 humanoidmaze-medium-stitch-v0-val.npz 的 qpos xy 全範圍量出（一手：x -1.93~21.93,
# y -1.90~21.92 => 中點 10.00,10.01）。ant 對應值 9.42,10.24（p2_analyze_traces.py:37,
# 從 antmaze 的資料 xy 範圍量出）。
MAZE_CENTER = (10.0004, 10.0064)

"""
DELTA_SUB 要不要重新校？（任務要求：尺度明顯不同就要校，並在報告標理由與數值）

查證鏈（一手，grep 對過原始碼）：
  1. ogbench/locomaze/maze.py 的 MazeEnv.__init__ 對 maze_unit 的預設值是 4.0
     （maze.py:40）。
  2. ogbench/locomaze/ant.py 與 ogbench/locomaze/humanoid.py 兩份檔案裡都
     grep 不到 "maze_unit" 字樣 ⇒ 兩者都沒有覆寫它，都吃預設值 4.0。
  3. 'medium' 這個迷宮佈局字串在 maze.py 只定義一份（maze.py 三處
     elif self._maze_type == 'medium'，同一個 MazeEnv 類別內，ant/humanoid
     用同一份，不是各自一份）。
  ⇒ humanoidmaze-medium 跟 antmaze-medium 是「同一個迷宮、同一個座標尺度」，
    只是站在裡面走的軀體不同。上面量出的 MAZE_CENTER（10.00,10.01）跟 ant
    版本（9.42,10.24）量級幾乎一樣，數值上佐證了這個判斷（不是巧合）。
  ⇒ DELTA_SUB=7.5（沿路標弧長，原始 xy 座標單位）在 humanoid 上代表的仍然是
    同一段物理迷宮距離，不需要因為「humanoid 走得比較慢」重新校——humanoid
    比較慢的是【時間尺度】（量尺包：每步位移 0.048 vs ant 0.131，只有 37%），
    不是【空間尺度】。時間尺度的差異已經由 N_data／N_table 兩把限時尺各自
    處理（humanoid 量尺包自己的 distance→steps 表本身就量得比 ant 慢很多，
    例如 d=2 時 p50=43 步 vs ant 15 步），不需要再靠改 DELTA_SUB 重複處理一次。
    rho=1.875=0.25*DELTA_SUB 沿用同一個推論一併不改。
  ⚠️ 副作用（如實記）：humanoid 一個 episode（401 步）的平均總弧長明顯短於
    ant 一個 episode（201 步）（humanoid 步速只有 37%，episode 只長 2 倍，
    2×0.37=0.74 倍，總弧長反而更短）⇒ 每條 episode 能切出的路標數 M 預期比
    ant 少（ant p50=3），本檔案的 build_tasks 會如實印出實測的 M 分佈，數字
    見報告，不是漏做。
"""


def load_dict(ckpt_path, map_location="cpu"):
    """讀 humanoid_dict 的 FactorizedCondVQVAE ckpt（見 humanoid_dict/run_cell.py 存檔格式）。

    回傳 (model, cfg, obs_mu, obs_sd)。obs_mu/obs_sd 是 (1,69) torch tensor，
    推論時要自己正規化 obs（ckpt 沒有把它們存成 model buffer，跟 ant 的
    wv_common.CondVQVAE 不同，是 humanoid_dict/run_cell.py 既有的存檔慣例）。
    """
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    seg_dim = cfg["seg_len"] * hc.ACT_DIM
    assert seg_dim == cfg.get("seg_dim", seg_dim), "⛔ seg_dim 對不上 cfg"
    model = hc.FactorizedCondVQVAE(
        seg_dim, hc.OBS_DIM, cfg["G"], cfg["K"],
        D=cfg["group_dim"], hidden=cfg["hidden"], decay=cfg["decay"],
        dead_steps=cfg["dead_steps"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    obs_mu = torch.as_tensor(np.asarray(ck["obs_mu"]), dtype=torch.float32)
    obs_sd = torch.as_tensor(np.asarray(ck["obs_sd"]), dtype=torch.float32)
    return model, cfg, obs_mu, obs_sd


def norm_obs(obs, obs_mu, obs_sd):
    return (obs - obs_mu) / obs_sd


def encode_idx(model, x):
    """動作段 (B,seg_dim) -> 每組字 index (B,G)。跟 obs 無關（encoder 只吃 x，跟 ant 的
    CondVQVAE 同設計，見 hd_common.FactorizedCondVQVAE.forward）。"""
    with torch.no_grad():
        z_e = model.encoder(x)
        _, idx, _ = model.vq(z_e, training=False)
    return idx


def decode_from_idx(model, idx, obs_raw, obs_mu, obs_sd):
    """(B,G) 字 index ＋ 原始（未正規化）obs (B,69) -> 動作段 (B,seg_dim)。"""
    with torch.no_grad():
        o = norm_obs(obs_raw, obs_mu, obs_sd)
        return model.decode_codes(idx, o)


def make_env(data_dir=DD_DEFAULT):
    """不含 renderer 的純模擬 env（量測用；render 版另見 hd_common.make_render_env）。"""
    import ogbench
    env = ogbench.make_env_and_datasets(hc.DATASET_NAME, dataset_dir=data_dir, env_only=True)
    env.reset(seed=0)
    return env


def sim_obs(u):
    """humanoid 的 obs 不是 concat(qpos,qvel)（跟 ant 不同，ant 才能手動 concat）——
    humanoid 的 69 維是 xy+joint_angles+head_height+extremities+torso_vert+com_vel+qvel
    的自訂特徵（ogbench/locomaze/humanoid.py:get_ob）。直接呼叫環境自己的
    get_ob(ob_type='states')，逐位元對應資料集的 observations 欄位
    （MazeEnv.get_ob 預設 ob_type 走 render 分支，這裡明確傳 'states' 避免踩到）。
    """
    return np.asarray(u.get_ob(ob_type="states"), dtype=np.float32)


def episode_bounds(terminals):
    return hc.episode_bounds(terminals)


def timed_reach(first_hit, N, mult, horizon):
    budget = np.ceil(np.asarray(N, dtype=float) * mult)
    evaluable = budget <= horizon
    reached = (first_hit > 0) & (first_hit <= budget)
    return reached, evaluable


def q(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {str(x): None for x in qs}
    return {str(x): float(np.percentile(a, x)) for x in qs}


def load_ruler(path=RULER_DEFAULT):
    with open(path) as f:
        return json.load(f)


def ruler_steps_for_distance(ruler, d, qq="50"):
    return hc.ruler_steps_for_distance(ruler, d, qq)


def save_json(obj, path):
    hc.save_json(_np_clean(obj), path)


def _np_clean(o):
    if isinstance(o, dict):
        return {k: _np_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def arclength(xy):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])
