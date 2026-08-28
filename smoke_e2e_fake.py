"""smoke：拿【假資料】把主線 scratch_lacot_rollout.py 真的跑起來（⛔ 不需要 /archive、不需要 GPU）。

🚨 為什麼要有這一支（2026-08-28）：那天的 code review 抓到的六個坑，全部是
   「不報錯、只安靜給出錯誤結論」—— 而它們全都在【只有真的跑起來才會經過】的路徑上：
     R=0 的爬坡快取、官方 rollout 的跨集汙染、三個 arm 的檔名、
     decoder 探針的位置、幾何 value 的健康檢查、subgoal 的落點。
   ⇒ 純函式的 smoke（smoke_rollout_fixes.py）驗得了邏輯，⛔ 驗不了「這條路真的走得通」。

⭐ 做法：用迷宮的 BFS 最短路造一份形狀對的假資料 ⇒ 佔據圖真的是那張迷宮
   ⇒ GeoValue / BFS subgoal / arc_subgoal 走的都是真的程式碼。
⛔ 數值沒有任何意義（只訓幾十步）—— 這支驗的是【接線】，⛔ 不是成績。
⛔ 產物一律寫進 tempdir（LACOT_OUT_DIR），⛔ 不准落進 results/。

    MUJOCO_GL=osmesa ./.venv/bin/python smoke_e2e_fake.py     # ~30 秒
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")

try:
    import gymnasium
    import ogbench  # noqa: F401
    import ogbench.locomaze  # noqa: F401
    from lacot.dev_eval import _bfs_from, _passable_cells
    _env = gymnasium.make("pointmaze-medium-v0")
except Exception as e:                                  # pragma: no cover
    print(f"⚠️ SKIP：這台機器起不了 mujoco / ogbench 環境（{type(e).__name__}: {e}）")
    print("   ⛔ 這【不是】通過 —— 只是這支沒跑到。純邏輯的部分請跑 smoke_rollout_fixes.py")
    sys.exit(0)

fails = []
ENV_NAME = "pointmaze-medium-navigate-v0"


def make_fake_dataset(out_dir):
    """沿迷宮的 BFS 最短路造軌跡 ⇒ 佔據圖真的長得像那張迷宮。"""
    u = _env.unwrapped
    cells = _passable_cells(_env)
    rng = np.random.default_rng(0)

    def path(s, g):
        d = _bfs_from(_env, g)
        if s not in d:
            return None
        cur, out = s, [s]
        while d[cur] > 0:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cur[0] + di, cur[1] + dj)
                if n in d and d[n] == d[cur] - 1:
                    cur = n
                    out.append(n)
                    break
        return out

    OBS, ACT, TERM = [], [], []
    for _ in range(260):
        s, g = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        p = path(tuple(s), tuple(g))
        if p is None or len(p) < 4:
            continue
        xy = np.array([u.ij_to_xy(c) for c in p], np.float64)
        fine = []
        for a, b in zip(xy[:-1], xy[1:]):
            for t in np.linspace(0, 1, 4)[:-1]:
                fine.append(a * (1 - t) + b * t)
        fine.append(xy[-1])
        fine = np.array(fine) + rng.normal(0, .12, (len(fine), 2))
        OBS.append(fine)
        ACT.append(np.clip(np.diff(fine, axis=0, append=fine[-1:]) / 1.5, -1, 1))
        t = np.zeros(len(fine), bool)
        t[-1] = True
        TERM.append(t)
    OBS = np.concatenate(OBS).astype(np.float32)
    ACT = np.concatenate(ACT).astype(np.float32)
    TERM = np.concatenate(TERM).astype(np.float32)
    for suffix in ("", "-val"):
        np.savez(os.path.join(out_dir, f"{ENV_NAME}{suffix}.npz"),
                 observations=OBS, actions=ACT, terminals=TERM)
    return OBS.shape[0], int(TERM.sum())


BASE = dict(
    MUJOCO_GL="osmesa", LACOT_ENV=ENV_NAME,
    LACOT_STEPS1="40", LACOT_STEPS2="20",
    LACOT_DEV_EVAL="1", LACOT_DEV_PER_TIER="1", LACOT_DEV_MIN_DIST="3",
    LACOT_EVAL_EPISODES="1", LACOT_EVAL_MAXH="8", LACOT_EVAL_RS="0,1",
    LACOT_GRAD_R="2", LACOT_GRAD_R_WARM="1",
)
ARMS = {
    # ⭐ 三個要互相對打的配置 —— 它們以前會產生【同一個檔名】
    "S0(bfs)": dict(LACOT_SUBGOAL="bfs"),
    "flat-grad": dict(LACOT_ENC_OBJ="recon", LACOT_LEARNED_REFINE="0", LACOT_GRAD_REFINE="1"),
    "S1(latent)": dict(LACOT_ENC_OBJ="recon", LACOT_LEARNED_REFINE="0", LACOT_GRAD_REFINE="1",
                       LACOT_SUBGOAL="latent"),
}


def run(name, extra, data_dir, out_dir):
    env = dict(os.environ, OGBENCH_DATA_DIR=data_dir, LACOT_OUT_DIR=out_dir, **BASE, **extra)
    p = subprocess.run([sys.executable, "-u", str(ROOT / "experiments" / "scratch_lacot_rollout.py")],
                       env=env, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        print(p.stdout[-2500:])
        print(p.stderr[-2500:])
        fails.append(f"{name} 跑掛了（returncode={p.returncode}）")
        return None, None
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("寫入 ")]
    if not line:
        fails.append(f"{name} 沒有寫出 json")
        return p.stdout, None
    return p.stdout, json.load(open(line[-1][3:].strip()))


def want(cond, name, msg):
    print(f"      {'✓' if cond else '🚨'} {msg}")
    if not cond:
        fails.append(f"{name}: {msg}")


data_dir = tempfile.mkdtemp(prefix="lacot_fake_data_")
out_dir = tempfile.mkdtemp(prefix="lacot_fake_out_")
try:
    n_row, n_traj = make_fake_dataset(data_dir)
    print(f"假資料：{n_row} row / {n_traj} 條軌跡（沿 medium 迷宮的最短路造的）")
    print(f"⛔ 產物全部寫到 {out_dir}，⛔ 不碰 results/\n")

    tags = {}
    for name, extra in ARMS.items():
        print(f"── {name} ──  " + " ".join(f"{k.replace('LACOT_', '')}={v}" for k, v in extra.items()))
        stdout, js = run(name, extra, data_dir, out_dir)
        if js is None:
            continue
        tags[name] = js["env"] and [ln for ln in stdout.splitlines() if ln.startswith("寫入 ")][-1]
        tags[name] = os.path.basename(tags[name][3:].strip())

        # #4：設定要讀得出來
        want(js.get("subgoal") == extra.get("LACOT_SUBGOAL", ""), name,
             f"json 讀得出 subgoal={js.get('subgoal')!r}")
        want(js.get("grad_refine") == int(extra.get("LACOT_GRAD_REFINE", 0)), name,
             f"json 讀得出 grad_refine={js.get('grad_refine')}")

        if extra.get("LACOT_GRAD_REFINE") == "1":
            # #8：幾何 value 的健康檢查真的跑了、而且過了
            want("幾何 value 健康檢查通過" in stdout, name, "GEO.health() 跑過並通過")
            want("格心 round-trip" in stdout, name, "印出格心 round-trip 誤差")
            # #6：decoder 的 ctx_usage gate 真的跑了
            want("decoder 讀得到 u" in stdout, name, "decoder 讀 u 的 gate 跑過並通過")
            # #15：X 對照不適用 ⇒ 要說出來，⛔ 不准照樣印判決
            want(js.get("x_refine_direction", {}).get("applicable") is False, name,
                 "X 對照標成不適用（GRAD_REFINE=1 ⇒ _RDIR 沒有作用點）")
            want("這個對照不適用" in stdout, name, "X 對照跳過時有印說明")
        else:
            want(js.get("x_refine_direction", {}).get("applicable") is True, name,
                 "有 learned refine ⇒ X 對照【適用】")

        if extra.get("LACOT_SUBGOAL"):
            # #2：分段 arm 的 R 要跟 LaCoT 一樣（⛔ 不是 0）
            want("分段" in stdout and "LaCoT (R=1)" in stdout, name, "分段 arm 與 LaCoT (R=1) 都跑了")
            # #11/#12：subgoal 的落點診斷
            d = js.get("subgoal_diag", {})
            want(d.get("n_plans", 0) > 0, name, f"subgoal 診斷有 {d.get('n_plans')} 筆")
            want(d.get("dsub_med") is not None, name,
                 f"‖sub − 現在‖ 中位 {d.get('dsub_med'):.2f}（DELTA_SUB=7.5）")
            want(d.get("sub_cap_chunks") == 10 and d.get("sub_stuck_chunks") == 3, name,
                 f"cap/stuck 以 chunk 記錄（{d.get('sub_cap_chunks')}/{d.get('sub_stuck_chunks')}）")
            if extra["LACOT_SUBGOAL"] == "latent":
                want(d.get("d0_med") is not None, name,
                     f"有量『路的第 0 點離現在多遠』（中位 {d.get('d0_med'):.2f}）")
        print()

    # ⭐ #4 的真正驗收：三個 arm 的結果檔名互不相同
    print("── #4 三個 arm 的實際檔名 ──")
    for k, v in tags.items():
        print(f"    {k:<11} {v}")
    want(len(set(tags.values())) == len(tags), "#4",
         f"{len(tags)} 個 arm ⇒ {len(set(tags.values()))} 個相異檔名（⛔ 相同就會互相覆蓋）")
finally:
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)

print()
if fails:
    print(f"🚨 FAIL（{len(fails)} 項）")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("✅ E2E PASS")
