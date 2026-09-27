#!/usr/bin/env python3
"""C1: frozen-code four-step counterfactuals on official antmaze val states."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
for p in (EXP / "bcodec_v0", EXP / "walk_verify", EXP / "gk_scan"):
    sys.path.insert(0, str(p))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import bcodec_common as bc  # noqa: E402
import wv_common as wv  # noqa: E402
from p1_replay import sim_obs  # noqa: E402

DATASET = "antmaze-medium-stitch-v0"
PREREGISTERED = {"edelta_lt": 0.5, "discriminability_rms_fraction_gte": 0.10,
                 "paired_bootstrap_ci_level": 0.95, "paired_benefit_gt": 0.0}
SPACES = ("xy", "yaw", "gait")


def load_model(path):
    # Both trained families use this one public adapter throughout the probe.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "G" in ck["config"]:
        return bc.load_bcodec_gk_ckpt(path)
    return bc.load_bcodec_ckpt(path)


def as_code(idx):
    return tuple(int(v) for v in np.asarray(idx).reshape(-1))


def decode(model, code, obs_norm):
    shape = (1, model.G) if hasattr(model, "G") else (1,)
    idx = torch.tensor(code, dtype=torch.long).reshape(shape)
    with torch.no_grad():
        ah, sh = model.decode_codes(idx, torch.from_numpy(obs_norm[None].astype(np.float32)))
    return (ah.numpy().reshape(4, model.act_dim).copy(),
            sh.numpy().reshape(4, model.obs_dim).copy())


def encode_pool(model, raw, starts, mu, sd, batch=512):
    """Encode whole donor segments once; a GxK tuple always comes from one segment."""
    out = []
    for off in range(0, len(starts), batch):
        ix = starts[off:off + batch, None] + np.arange(4)[None, :]
        ss = ((raw["observations"][ix] - mu) / sd).reshape(len(ix), -1).astype(np.float32)
        aa = raw["actions"][ix].reshape(len(ix), -1).astype(np.float32)
        with torch.no_grad():
            _, idx, _ = model.vq(model.encode(torch.from_numpy(ss), torch.from_numpy(aa)), training=False)
        out.extend(as_code(row) for row in idx.numpy().reshape(len(ix), -1))
    return out


def wrappers(env):
    out = []
    cur = env
    while True:
        out.append(cur)
        if cur is cur.unwrapped:
            return out
        cur = cur.env


def snapshot(env):
    """Capture after env.reset()+set_state, matching run_reset_chunks' starting state."""
    u = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(u.model, spec), dtype=np.float64)
    mujoco.mj_getState(u.model, u.data, state, spec)
    objects = wrappers(env)
    attrs = []
    for obj in objects:
        attrs.append({key: copy.deepcopy(vars(obj)[key]) for key in
                      ("_elapsed_steps", "cur_goal_xy", "cur_task_id", "cur_task_info")
                      if key in vars(obj)})
    rng = []
    for obj in objects + [env.action_space]:
        gen = vars(obj).get("_np_random")
        rng.append(copy.deepcopy(gen.bit_generator.state) if gen is not None else None)
    return dict(state=state, warmstart=u.data.qacc_warmstart.copy(), attrs=attrs, rng=rng)


def restore(env, snap):
    u = env.unwrapped
    mujoco.mj_setState(u.model, u.data, snap["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(u.model, u.data)
    u.data.qacc_warmstart[:] = snap["warmstart"]
    objects = wrappers(env)
    for obj, attrs in zip(objects, snap["attrs"]):
        for key, value in attrs.items():
            setattr(obj, key, copy.deepcopy(value))
    for obj, state in zip(objects + [env.action_space], snap["rng"]):
        if state is not None:
            vars(obj)["_np_random"].bit_generator.state = copy.deepcopy(state)


def reset_to_dataset(env, raw, s):
    env.reset()
    env.unwrapped.set_state(raw["qpos"][s].copy(), raw["qvel"][s].copy())


def execute_code(env, snap, model, code, mu, sd, shat_override=None):
    """Same frozen decoder and clipped four-step env path in tests and real runs."""
    restore(env, snap)
    obs0 = sim_obs(env.unwrapped)
    ah, shn = decode(model, code, (obs0 - mu) / sd)
    # ŝ is observation only. The optional override tests this one-way wiring.
    if shat_override is not None:
        shn = np.asarray(shat_override, dtype=np.float32).copy()
    sh = shn * sd[None, :] + mu[None, :]
    if not (np.isfinite(ah).all() and np.isfinite(sh).all()):
        raise RuntimeError(f"Nonfinite decoder output for code {code}")
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    actions = np.clip(ah, lo, hi).astype(np.float64)
    actual = []
    for action in actions:
        env.step(action)
        actual.append(sim_obs(env.unwrapped).copy())
    if not np.isfinite(actual).all():
        raise RuntimeError(f"Nonfinite simulator state for code {code}")
    return dict(code=list(code), obs_start=obs0.tolist(), actions=actions.tolist(),
                shat_norm=shn.tolist(), shat=sh.tolist(), actual=np.asarray(actual).tolist())


def yaw(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def project(obs, sd):
    """xy metres; yaw radians; gait = SE(2)-quotiented, training-scale shape/velocity."""
    o = np.asarray(obs, dtype=np.float64)
    angle = yaw(o[3:7])
    w, x, y, z = o[3:7] / max(np.linalg.norm(o[3:7]), 1e-12)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    vx, vy = o[15:17]
    c, s = np.cos(angle), np.sin(angle)
    gait = np.concatenate(([(o[2] / sd[2]), roll, pitch], o[7:15] / sd[7:15],
                           [(c * vx + s * vy) / sd[15], (-s * vx + c * vy) / sd[16]],
                           o[17:29] / sd[17:29]))
    return {"xy": o[:2], "yaw": np.array([angle]), "gait": gait}


def squared_delta(a, b, space):
    d = np.asarray(a) - np.asarray(b)
    return float(np.sum(wrap(d) ** 2 if space == "yaw" else d ** 2))


def start_stats(cells, teacher_end, sd, permutation, teacher_start=None):
    projected = [{kind: project(np.asarray(c[kind])[-1], sd) for kind in ("shat", "actual")}
                 for c in cells]
    base = project(cells[0]["obs_start"] if teacher_start is None else teacher_start, sd)
    teacher = project(teacher_end, sd)
    sums = {}
    for space in SPACES:
        pred = [p["shat"][space] for p in projected]
        truth = [p["actual"][space] for p in projected]
        num = den = perm_num = 0.0
        pairs = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                yd = wrap(truth[i] - truth[j]) if space == "yaw" else truth[i] - truth[j]
                pd = wrap(pred[i] - pred[j]) if space == "yaw" else pred[i] - pred[j]
                pp = (wrap(pred[permutation[i]] - pred[permutation[j]]) if space == "yaw"
                      else pred[permutation[i]] - pred[permutation[j]])
                dn = squared_delta(pd, yd, space)
                dd = float(np.sum(yd ** 2))
                pn = squared_delta(pp, yd, space)
                num += dn; den += dd; perm_num += pn
                pairs.append(dict(codes=[i, j], difference_error_sq=dn,
                                  true_difference_sq=dd, permuted_error_sq=pn))
        absolute = [squared_delta(pred[i], truth[i], space) for i in range(len(cells))]
        for cell, err in zip(cells, absolute):
            cell.setdefault("absolute_error_sq", {})[space] = err
        sums[space] = dict(numerator=num, denominator=den, permutation_numerator=perm_num,
                           absolute_error_sq=sum(absolute), teacher_change_sq=squared_delta(teacher[space], base[space], space),
                           pairs=pairs)
    return sums


def derangement(rng, n=4):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p.tolist()


def summarize(starts, rng, n_boot=2000):
    result = {}
    for space in SPACES:
        rows = [s["stats"][space] for s in starts]
        den = np.array([r["denominator"] for r in rows])
        num = np.array([r["numerator"] for r in rows])
        perm = np.array([r["permutation_numerator"] for r in rows])
        teacher = np.array([r["teacher_change_sq"] for r in rows])
        pair_rms = float(np.sqrt(den.sum() / (6 * len(rows))))
        teacher_rms = float(np.sqrt(teacher.mean()))
        discriminable = bool(pair_rms > 0 and
                             pair_rms >= PREREGISTERED["discriminability_rms_fraction_gte"] * teacher_rms)
        ed = float(num.sum() / den.sum()) if den.sum() > 0 else None
        pe = float(perm.sum() / den.sum()) if den.sum() > 0 else None
        by_kind = {}
        for kind in ("clean", "drift"):
            kind_rows = [s["stats"][space] for s in starts if s["kind"] == kind]
            kind_den = sum(r["denominator"] for r in kind_rows)
            by_kind[kind] = (float(sum(r["numerator"] for r in kind_rows) / kind_den)
                             if kind_den > 0 else None)
        benefit_ci = None
        if den.sum() > 0:
            ix = rng.integers(0, len(rows), size=(n_boot, len(rows)))
            bden = den[ix].sum(axis=1)
            benefits = (perm[ix].sum(axis=1) - num[ix].sum(axis=1)) / np.maximum(bden, 1e-30)
            benefit_ci = np.quantile(benefits, [.025, .975]).tolist()
        verdict = ("此視窗不可辨識" if not discriminable else
                   "通過" if ed is not None and ed < .5 and benefit_ci[0] > 0 else "未過線")
        result[space] = dict(E_delta=ed, e_delta_clean=by_kind["clean"],
                             e_delta_drift=by_kind["drift"], permutation_E_delta=pe,
                             absolute_error_rmse=float(np.sqrt(sum(r["absolute_error_sq"] for r in rows) / (4 * len(rows)))),
                             true_pair_rms=pair_rms, teacher_four_step_rms=teacher_rms,
                             discriminability_ratio=pair_rms / teacher_rms if teacher_rms > 0 else None,
                             discriminable=discriminable, paired_benefit_ci95=benefit_ci,
                             preregistered_thresholds=PREREGISTERED, verdict=verdict)
    return result


def choose_donors(pool, pool_codes, pool_obs_norm, pool_actions, episode_of, s, obs0, original, mu, sd):
    # Pose/velocity only, measured in the bcodec checkpoint's own obs normalization.
    query = ((obs0 - mu) / sd)[2:]
    dist = np.sum((pool_obs_norm[:, 2:] - query) ** 2, axis=1)
    order = np.argsort(dist, kind="stable")
    selected = []
    seen = {original}
    source_action = pool_actions[np.searchsorted(pool, s)] if s in pool else None
    for j in order:
        j = int(j)
        code = pool_codes[j]
        if episode_of[int(pool[j])] == episode_of[s] or code in seen:
            continue
        if source_action is not None and np.linalg.norm(pool_actions[j] - source_action) < 0.5:
            continue
        selected.append(dict(code=code, segment=int(pool[j]), normalized_obs_distance=float(np.sqrt(dist[j])),
                             action_distance=(float(np.linalg.norm(pool_actions[j] - source_action))
                                              if source_action is not None else None)))
        seen.add(code)
        if len(selected) == 3:
            break
    return selected


def run(args):
    torch.set_num_threads(max(1, args.threads))
    model, cfg, mu_t, sd_t = load_model(args.ckpt)
    if cfg["seg_len"] != 4 or cfg["obs_dim"] != 29 or cfg["act_dim"] != 8:
        raise ValueError(f"Unexpected checkpoint dimensions: {cfg}")
    if float(cfg["lam_s"]) != 1.0:
        raise ValueError(f"C1 requires the λ1 bcodec checkpoint, got lam_s={cfg['lam_s']}")
    mu, sd = mu_t.numpy().reshape(-1), sd_t.numpy().reshape(-1)
    raw = wv.load_npz(args.data_dir, DATASET, "val")
    pool = wv.cut_segments(raw, 4)
    starts_ep, ends_ep = wv.episode_bounds(raw["terminals"])
    episode_of = np.empty(len(raw["actions"]), dtype=np.int32)
    for ei, (a, b) in enumerate(zip(starts_ep, ends_ep)):
        episode_of[a:b + 1] = ei
    pool_codes = encode_pool(model, raw, pool, mu, sd)
    pool_obs_norm = (raw["observations"][pool] - mu) / sd
    pool_actions = raw["actions"][pool[:, None] + np.arange(4)].reshape(len(pool), -1)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(starts_ep))
    env = wv.make_env(args.data_dir, DATASET)
    rows = []
    drift_selfconsistent = []
    coverage = {kind: dict(attempted=0, accepted=0, donor_shortage=0) for kind in ("clean", "drift")}
    for ei in order:
        if len(rows) >= args.n_starts:
            break
        kind = "clean" if len(rows) % 2 == 0 else "drift"
        s0, e0 = int(starts_ep[ei]), int(ends_ep[ei])
        s = s0 if kind == "clean" else s0 + 4
        if s + 4 > e0:
            continue
        coverage[kind]["attempted"] += 1
        if kind == "clean":
            reset_to_dataset(env, raw, s)
        else:
            reset_to_dataset(env, raw, s0)
            warm = snapshot(env)
            idx0 = np.searchsorted(pool, s0)
            assert pool[idx0] == s0
            code0 = pool_codes[idx0]
            execute_code(env, warm, model, code0, mu, sd)
        snap = snapshot(env)
        obs0 = sim_obs(env.unwrapped)
        idx = np.searchsorted(pool, s)
        assert pool[idx] == s
        original = pool_codes[idx]
        donors = choose_donors(pool, pool_codes, pool_obs_norm, pool_actions, episode_of,
                               s, obs0, original, mu, sd)
        if len(donors) < 3:
            coverage[kind]["donor_shortage"] += 1
            continue
        codes = [original] + [d["code"] for d in donors]
        cells = [execute_code(env, snap, model, c, mu, sd) for c in codes]
        start_id = f"val_ep{ei}_s{s}_{kind}"
        for ci, (cell, donor) in enumerate(zip(cells, [None] + donors)):
            cell.update(start_id=start_id, code_id=ci,
                        donor_segment=None if donor is None else donor["segment"],
                        donor_obs_distance=None if donor is None else donor["normalized_obs_distance"],
                        donor_action_distance=None if donor is None else donor["action_distance"])
        permutation = derangement(rng)
        stats = start_stats(cells, raw["observations"][s + 4], sd, permutation,
                            teacher_start=raw["observations"][s])
        rows.append(dict(start_id=start_id, episode=int(ei), segment=int(s),
                         kind=kind, permutation=permutation, cells=cells, stats=stats))
        if kind == "drift":
            drift_selfconsistent.append(original == code0)
        coverage[kind]["accepted"] += 1
    if len(rows) < args.n_starts:
        raise RuntimeError(f"Only {len(rows)}/{args.n_starts} starts with three distinct donor tuples; coverage={coverage}")
    summary = summarize(rows, rng, args.bootstrap)
    result = dict(protocol="C1 E_delta", checkpoint=str(Path(args.ckpt).resolve()), config=cfg,
                  seed=args.seed, dataset=raw["_path"], coverage=coverage, starts=rows,
                  summary=summary, preregistered_thresholds=PREREGISTERED,
                  drift_selfconsistency_rate=float(np.mean(drift_selfconsistent)),
                  scale="smoke" if args.n_starts < 200 else "main",
                  n_starts=len(rows), n_codes=sum(len(r["cells"]) for r in rows),
                  n_steps=4 * sum(len(r["cells"]) for r in rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"saved: {out}")
    if args.n_starts < 200:
        print("SMOKE ONLY: preregistered thresholds are displayed; no scientific conclusion at this scale")
    for space, rec in summary.items():
        print(f"{space}: E_delta={rec['E_delta']:.6g} permutation={rec['permutation_E_delta']:.6g} "
              f"absolute_rmse={rec['absolute_error_rmse']:.6g} discriminability={rec['discriminability_ratio']:.6g} "
              f"benefit_CI95={rec['paired_benefit_ci95']} verdict={rec['verdict']} "
              f"[preregistered: E_delta<.5, pair RMS>=.1 teacher RMS, paired benefit CI95 lower>0]")
        print(f"{space}: e_delta_clean={rec['e_delta_clean']:.6g} e_delta_drift={rec['e_delta_drift']:.6g}")
    print(f"drift_selfconsistency_rate={result['drift_selfconsistency_rate']:.6g}")
    print(f"coverage={coverage}")
    print(f"n_starts={result['n_starts']} n_codes={result['n_codes']} n_steps={result['n_steps']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="bcodec λ1 checkpoint path; no inferred path")
    ap.add_argument("--data-dir", default=os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.n_starts < 2 or args.n_starts % 2:
        ap.error("--n-starts must be even and at least 2 (equal clean/drift coverage)")
    run(args)


if __name__ == "__main__":
    main()
