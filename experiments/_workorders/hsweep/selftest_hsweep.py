"""T1–T6: real local data, CPU only; mutations live in memory, artifacts in /tmp.

No package installation, GPU, Slurm, results writes, or cleanup/deletion.
"""
import os
import sys

# Set these before importing torch, ogbench, or either trainer's source.
sys.dont_write_bytecode = True
for key in list(os.environ):
    if key.startswith("LACOT_"):
        del os.environ[key]
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    "MUJOCO_GL": "osmesa", "LACOT_ENV": "pointmaze-large-stitch-v0",
    "OGBENCH_DATA_DIR": os.path.expanduser("~/.ogbench/data"),
    "LACOT_ENC_OBJ": "recon_ictr", "LACOT_LEARNED_REFINE": "0",
    "LACOT_COND_DROP": "0.1", "LACOT_BC_INDEP": "1",
    "LACOT_TEACHER_MIX": "0.5", "LACOT_K": "8",
    "LACOT_TCAP": "128", "LACOT_DEC_START": "soft",
    "LACOT_EMA_W": "0.999", "LACOT_WARMUP": "500",
})

import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
M9 = ROOT / "experiments/_workorders/ucontrast1/smoke/mutants/M9/scratch_lacot_rollout.py"
TRAIN = HERE / "train_hsweep.py"
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT)
torch.set_num_threads(2)
FIELDS = ("traj", "mask", "s", "g", "act")
N_BATCHES = 8
SCRATCH = Path(tempfile.mkdtemp(prefix="hsweep-selftest-", dir="/tmp"))
os.environ["LACOT_OUT_DIR"] = str(SCRATCH)


def tree(source):
    return ast.parse(source)


def execute(nodes, ns, filename):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec"), ns)


def function(source, name):
    return next(n for n in tree(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def mutation_namespace(ns, source):
    # Rebind caller AND callees: copying a dict alone leaves Python functions'
    # __globals__ pointing at the original engine, bypassing the mutant.
    mutant = copy.copy(ns)
    execute([function(source, name) for name in
             ("_teacher_prefix", "_teacher_traj", "make_batch")], mutant, "<mutation-engine>")
    return mutant


def load_data_engine(path):
    # Execute the unchanged imports/data/config/pool and actual make_batch,
    # stopping before sota_mlp/model construction/training. No mocked dataset.
    nodes = tree(path.read_text()).body
    stop = next(i for i, n in enumerate(nodes)
                if isinstance(n, ast.FunctionDef) and n.name == "sota_mlp")
    ns = {"__file__": str(path), "__name__": "hsweep_cpu_data_engine"}
    execute(nodes[:stop], ns, path)
    ns["_BOOT_RNG"] = None
    assert ns["device"] == "cpu"
    assert ns["T_CAP"] == 128
    return ns


def arrays(ns, rng, mix):
    return [x.detach().cpu().numpy() for x in ns["make_batch"](rng, teacher_mix=mix)]


def equal_fields(left, right, context):
    for name, a, b in zip(FIELDS, left, right):
        assert np.array_equal(a, b), f"{context}: {name} is not np.array_equal"


def t1(ref, candidate):
    candidate["TEACHER_H"] = None
    count = 0
    for seed in (33, 20261004):
        for mix in (0.0, 0.5):
            ra, rb = np.random.default_rng(seed), np.random.default_rng(seed)
            for batch in range(N_BATCHES):
                equal_fields(arrays(ref, ra, mix), arrays(candidate, rb, mix),
                             f"T1 seed={seed} mix={mix} batch={batch}")
                assert np.array_equal(ref["_REAL_W"][0].numpy(),
                                      candidate["_REAL_W"][0].numpy())
                count += 1
    return count


def killer(label, check, expected_error=None):
    try:
        check()
    except AssertionError as exc:
        if expected_error is not None:
            assert expected_error in str(exc), f"{label}: failed the wrong assertion: {exc}"
        print(f"{label}: FAIL (expected; killer detected)", flush=True)
        traceback.print_exc(file=sys.stdout)
    else:
        raise AssertionError(f"{label}: killer survived; test did not detect defect")


def raw_length(p, sd):
    return float(np.linalg.norm(np.diff(p.astype(np.float64), axis=0) * sd, axis=1).sum())


def capture_prefix(ns):
    records = []
    original = ns["_teacher_prefix"]

    def capture(p):
        out = original(p)
        records.append((p.copy(), out.copy()))
        return out

    ns["_teacher_prefix"] = capture
    return records, original


def t2_baseline(ref, ns):
    # Feed every pool index to the actual M9 _teacher_traj, retaining its jitter,
    # normalized-arclength interpolation and final float32 cast unchanged.
    ns["TEACHER_H"] = None
    assert "LACOT_TEACHER_H" not in os.environ
    assert len(ref["_TCH"]) == 4096 and ref["_BOOT"] is None

    class EveryRouteRNG:
        def __init__(self):
            self.rng = np.random.default_rng(33)
            self.index = -1
            self.full = []

        def integers(self, high):
            assert high == 4096
            self.index += 1
            return self.index

        def uniform(self, low, high, size):
            assert (low, high, size) == (-0.5, 0.5, (1, 2))
            jitter = self.rng.uniform(low, high, size=size)
            self.full.append(ref["_TCH"][self.index] + jitter * ref["_tcell_norm"])
            return jitter

    rng = EveryRouteRNG()
    traj = ref["_teacher_traj"](rng, 4096)
    assert len(rng.full) == 4096 and rng.index == 4095
    shortages = np.array([1.0 - raw_length(t, ref["SD_XY"]) / raw_length(p, ref["SD_XY"])
                          for t, p in zip(traj, rng.full)])
    b = float(shortages.max())
    print(f"T2 baseline: H unset; all 4096 routes; actual M9 interpolation; "
          f"B={b:.12g} median={np.median(shortages):.12g} "
          f"p99={np.percentile(shortages, 99):.12g}", flush=True)
    return b


def t2(ns, baseline):
    rows = []
    for h in (3, 4, 8):
        ns["TEACHER_H"] = h
        rng = np.random.default_rng(33)
        records, trajs, goal_batches = [], [], []
        truncated_count = 0
        # Continue the same seed's stream until >=200 truncated samples; do not
        # filter out any previously drawn sample from the checks below.
        while truncated_count < 200:
            chunk, original = capture_prefix(ns)
            try:
                traj, goals = ns["_teacher_traj"](rng, 256)
            finally:
                ns["_teacher_prefix"] = original
            assert len(chunk) == 256
            records.extend(chunk)
            trajs.append(traj)
            goal_batches.append(goals)
            truncated_count += sum(raw_length(full, ns["SD_XY"]) > 4 * h
                                   for full, _ in chunk)
        traj, goals = np.concatenate(trajs), np.concatenate(goal_batches)
        errors, interp_errors, truncated = [], [], 0
        for i, (full, prefix) in enumerate(records):
            full_len = raw_length(full, ns["SD_XY"])
            prefix_len = raw_length(prefix, ns["SD_XY"])
            error = abs(prefix_len - min(4 * h, full_len))
            errors.append(error)
            assert error < 1e-6, f"T2 H={h} i={i} raw length error={error}"
            assert np.array_equal(goals[i], full[-1].astype(np.float32))
            if full_len > 4 * h:
                truncated += 1
                # Independent endpoint oracle: interpolate the FULL raw path at 4H.
                raw = full * ns["SD_XY"] + ns["MU_XY"]
                lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
                expected = np.array([np.interp(4 * h, lengths, raw[:, k]) for k in (0, 1)])
                assert np.allclose(prefix[-1] * ns["SD_XY"] + ns["MU_XY"],
                                   expected, rtol=0, atol=1e-10)
                assert np.array_equal(prefix[:-1], full[:len(prefix) - 1])
                rel = 1.0 - raw_length(traj[i], ns["SD_XY"]) / prefix_len
                interp_errors.append(rel)
                assert rel <= baseline + 1e-9, (f"T2 H={h} i={i} T_CAP relative shortage={rel} "
                                                f"> B={baseline} + 1e-9")
            else:
                assert np.array_equal(full, prefix), f"T2 H={h}: short path changed"
        assert truncated >= 200
        rows.append((h, len(records), truncated, max(errors), max(interp_errors)))
        print(f"T2 H={h}: PASS samples={len(records)} truncated={truncated} "
              f"max_raw_error={max(errors):.3e} "
              f"max_T_CAP_relative_shortage={max(interp_errors):.12g} B={baseline:.12g}", flush=True)
    print("T2 comparison: H samples truncated max_raw_error max_relative_shortage B", flush=True)
    for h, samples, truncated, raw_error, shortage in rows:
        print(f"{h} {samples} {truncated} {raw_error:.3e} {shortage:.12g} {baseline:.12g}", flush=True)
    # Exact boundary and duplicate vertices: no random calls involved.
    ns["TEACHER_H"] = 3
    for raw in (np.array([[0., 0.], [12., 0.]]),
                np.array([[0., 0.], [0., 0.], [12., 0.], [12., 0.], [16., 0.]]),
                np.array([[0., 0.], [2., 0.], [2., 2.]])):
        p = (raw - ns["MU_XY"]) / ns["SD_XY"]
        assert abs(raw_length(ns["_teacher_prefix"](p), ns["SD_XY"]) -
                   min(12., raw_length(p, ns["SD_XY"]))) < 1e-6


def t3(ns):
    ns["TEACHER_H"] = 3
    records, original = capture_prefix(ns)
    try:
        traj, mask, s, g, act = arrays(ns, np.random.default_rng(33), 0.5)
    finally:
        ns["_teacher_prefix"] = original
    n_r = ns["B"] - len(records)
    full_goals = np.stack([full[-1] for full, _ in records]).astype(np.float32)
    assert np.array_equal(g[n_r:], full_goals), "T3: g differs from full translated endpoint"
    assert np.array_equal(s[n_r:], traj[n_r:, 0]), "T3: s changed"
    distances = np.linalg.norm((g[n_r:] - traj[n_r:, -1]).astype(np.float64) * ns["SD_XY"], axis=1)
    assert distances.max() > 4, f"T3: no truncated sample with goal distance >4 ({distances.max()})"
    assert np.all(ns["_REAL_W"][0].numpy()[n_r:] == 0)
    assert not mask.any() and not act[n_r:].any()
    return float(distances.max())


class TraceRNG:
    """Records exact calls, order and arguments, while forwarding to a real RNG."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def __getattr__(self, name):
        target = getattr(self.rng, name)
        if not callable(target):
            return target

        def traced(*args, **kwargs):
            self.calls.append((name, repr(args), repr(kwargs)))
            return target(*args, **kwargs)
        return traced


def t4(ns):
    ra, rb = TraceRNG(20261004), TraceRNG(20261004)
    n_r = ns["B"] - int(round(ns["B"] * 0.5))
    for batch in range(N_BATCHES):
        ns["TEACHER_H"] = None
        a = arrays(ns, ra, 0.5)
        wa = ns["_REAL_W"][0].numpy().copy()
        ns["TEACHER_H"] = 3
        b = arrays(ns, rb, 0.5)
        equal_fields([x[:n_r] for x in a], [x[:n_r] for x in b], f"T4 batch={batch} real rows")
        assert np.array_equal(wa, ns["_REAL_W"][0].numpy())
        assert json.dumps(ra.bit_generator.state, sort_keys=True) == json.dumps(rb.bit_generator.state, sort_keys=True)
    assert ra.calls == rb.calls, "T4: RNG call name/order/arguments differ"
    return len(ra.calls), n_r


def t5(m9_source, source):
    old, new = {}, {}
    execute([function(m9_source, "_tag_extra")], old, M9)
    execute([function(source, "_tag_extra")], new, TRAIN)
    recipe = dict(ENC_OBJ="recon_ictr", TEACHER_MIX=0.5, LEARNED_REFINE=0,
                  COND_DROP=0.1, BC_INDEP=1, EMA_W=0.999, WARMUP=500, DEC_START="soft")
    for kwargs in ({}, recipe, dict(recipe, STEPS1=2, AMP=1, COMPILE=1)):
        assert old["_tag_extra"](**kwargs) == new["_tag_extra"](**kwargs)
    assert new["_tag_extra"](**recipe, TEACHER_H=4) == old["_tag_extra"](**recipe).replace("_tch0.5", "_tch0.5_th4")
    nodes = tree(source).body
    tag_index = next(i for i, n in enumerate(nodes)
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tag" for t in n.targets))
    tag_node = nodes[tag_index]
    old_tag = next(n for n in tree(m9_source).body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tag" for t in n.targets))
    assert ast.dump(tag_node) == ast.dump(old_tag), "T5: original tag assembly changed"
    mkdir_index = next(i for i in range(tag_index, len(nodes))
                       if isinstance(nodes[i], ast.Expr) and isinstance(nodes[i].value, ast.Call)
                       and ast.unparse(nodes[i].value.func) == "os.makedirs")
    common = dict(os=os, __file__=str(TRAIN), ENV_NAME="pointmaze-large-stitch-v0",
                  CONS="self", K=8, COND=256, CHUNK=4, STEPS2=8000, T_CAP=128, SEEDS=2, TAG_SEED=33,
                  LOAD_CKPT="", CONT_TRAIN=0)
    for h in (None, 4):
        ns = dict(common, _extra=new["_tag_extra"](**recipe, TEACHER_H=h))
        execute([tag_node], ns, TRAIN)
        expected = "ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu" + old["_tag_extra"](**recipe) + "_s33.pt"
        if h is None:
            assert f"ckpt_{ns['tag']}.pt" == expected
        else:
            assert "_th4" in ns["tag"]
        target = SCRATCH / f"ckpt_{ns['tag']}.pt"
        # First use the actual new destination/guard segment with no target present.
        execute(nodes[tag_index + 1:mkdir_index + 1], ns, TRAIN)
        target.touch(exist_ok=False)
        rejected = False
        try:
            execute(nodes[tag_index + 1:mkdir_index + 1], ns, TRAIN)
        except SystemExit as exc:
            rejected = True
            print(f"T5 H={h}: existing fresh target rejected: {exc}", flush=True)
        assert rejected and target.stat().st_size == 0
        assert not Path(ns["dst"]).exists(), "T5: refusal wrote rollout JSON"
        print(f"T5 H={h}: PASS filename={target.name}", flush=True)
    # Test the actual final guard too, before torch.save is allowed to run.
    ck_index = next(i for i, n in enumerate(nodes)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ck" for t in n.targets))
    save_index = next(i for i in range(ck_index, len(nodes))
                     if isinstance(nodes[i], ast.Expr) and isinstance(nodes[i].value, ast.Call)
                     and ast.unparse(nodes[i].value.func) == "torch.save")
    try:
        execute(nodes[ck_index:save_index], ns, TRAIN)
    except SystemExit as exc:
        print(f"T5 save-time recheck: PASS {exc}", flush=True)
    else:
        raise AssertionError("T5: final overwrite guard failed")


def t6():
    out_dir = Path(tempfile.mkdtemp(prefix="hsweep-cpu-smoke-", dir="/tmp"))
    env = dict(os.environ, LACOT_OUT_DIR=str(out_dir), LACOT_TEACHER_H="3",
               LACOT_STEPS1="2", LACOT_STEPS2="2", LACOT_SEED="33",
               LACOT_LOG_EVERY="1", LACOT_DIAG_TRAIN="1",
               LACOT_EVAL_RS="0", LACOT_EVAL_EPISODES="1", LACOT_EVAL_MAXH="1",
               LACOT_DEV_EVAL="0", LACOT_PREREQ="0", LACOT_REV_ARM="0")
    cmd = [sys.executable, "-u", str(TRAIN)]
    print("T6 subprocess:", " ".join(cmd), flush=True)
    print("T6 environment:", json.dumps({k: v for k, v in sorted(env.items())
          if k.startswith("LACOT_") or k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
          "ROCR_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OGBENCH_DATA_DIR", "MUJOCO_GL", "PYTHONPATH",
          "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")}), flush=True)
    # Execute the entire trainer, unmodified: two updates per stage and only the
    # minimal original rollout required to reach checkpoint saving (20 env steps).
    log_path = out_dir / "smoke-output.txt"
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=300)
    output = log_path.read_text()
    print("T6 full subprocess output BEGIN", flush=True)
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    print("T6 full subprocess output END", flush=True)
    print("T6 last 20 lines BEGIN", flush=True)
    print("\n".join(output.splitlines()[-20:]), flush=True)
    print("T6 last 20 lines END", flush=True)
    assert proc.returncode == 0, f"T6: subprocess exit={proc.returncode}; log={log_path}"
    assert "device: cpu" in output and "device: cuda" not in output
    assert "step 2  recon-mse" in output and "step 2  l_nf/dim" in output
    checkpoints = list(out_dir.glob("ckpt_*.pt"))
    assert len(checkpoints) == 1 and "_th3" in checkpoints[0].name
    ck = checkpoints[0]
    checkpoint = torch.load(ck, map_location="cpu", weights_only=False)
    print("T6 torch.load keys:", list(checkpoint.keys()), flush=True)
    required = {"cond_enc", "cond_head", "flow", "refine", "ahead", "bc_head",
                "traj_enc", "e_pooler", "u_dec", "ema", "cfg"}
    if env["LACOT_DEC_START"] == "hard":
        required.update({"s_embed", "dec_start"})
    else:
        assert "s_embed" not in checkpoint and "dec_start" not in checkpoint
    condition = next((i, line) for i, line in enumerate(TRAIN.read_text().splitlines(), 1)
                     if 's_embed = nn.Linear' in line)
    print(f"T6 s_embed construction {TRAIN}:{condition[0]}: {condition[1].strip()}", flush=True)
    print(f"T6 recipe DEC_START={env['LACOT_DEC_START']} required keys={sorted(required)}", flush=True)
    assert required <= checkpoint.keys(), f"T6: missing keys {required - checkpoint.keys()}"
    assert checkpoint["cfg"]["STEPS2"] == 2 and checkpoint["cfg"]["T_CAP"] == 128
    for name in required - {"cfg", "ema", "dec_start"}:
        assert all(torch.isfinite(v).all() for v in checkpoint[name].values()
                   if isinstance(v, torch.Tensor) and v.is_floating_point()), f"T6: nonfinite {name}"
    print(f"T6: PASS ckpt={ck} bytes={ck.stat().st_size}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=("T1", "T2"), default="T1",
                        help="T2 retains the prior T1 evidence for rework r1")
    args = parser.parse_args()
    digest = hashlib.sha256(M9.read_bytes()).hexdigest()
    assert digest == "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"
    assert not torch.cuda.is_available(), "CPU-only selftest requires CUDA unavailable"
    print(f"M9={M9}\nsha256={digest}\nPython={sys.version}\n"
          f"numpy={np.__version__} torch={torch.__version__} device=cpu\n"
          f"data={os.environ['OGBENCH_DATA_DIR']}/{os.environ['LACOT_ENV']}.npz\n"
          f"temporary artifacts retained at {SCRATCH}", flush=True)
    m9_source, source = M9.read_text(), TRAIN.read_text()
    ref, ns = load_data_engine(M9), load_data_engine(TRAIN)
    assert len(ref["_TCH"]) == len(ns["_TCH"])
    assert all(np.array_equal(a, b) for a, b in zip(ref["_TCH"], ns["_TCH"]))
    print("M9/hsweep teacher pools: np.array_equal for every route", flush=True)
    if args.start_at == "T1":
        count = t1(ref, ns)
        print(f"T1: PASS H unset; {count} batches, all five fields + real-mask np.array_equal", flush=True)
        mutant = mutation_namespace(ns, source)
        original_teacher = ast.get_source_segment(source, function(source, "_teacher_traj"))
        needle = "        p = _teacher_prefix(p)"
        assert original_teacher.count(needle) == 1
        mutated_teacher = original_teacher.replace(needle, "        rng.random()  # intentional killer\n" + needle)
        execute(tree(mutated_teacher).body, mutant, "<T1-RNG-killer>")
        # Insert at the shared post-jitter truncation insertion site, so flag-off T1
        # can exercise it; an H-only mutation would be invisible to H-unset T1.
        killer("T1 RNG killer", lambda: t1(ref, mutant))
    else:
        prior = (HERE / "selftest-output.txt").read_text()
        assert "T1: PASS H unset; 32 batches" in prior
        assert "T1 RNG killer: FAIL (expected; killer detected)" in prior
        assert hashlib.sha256(TRAIN.read_bytes()).hexdigest() == "f748e2cb844dd9af2d3d93a624ca2545635953c4ffdbd684cf69fe8b2ceb377e"
        print("T1: prior PASS and RNG killer FAIL retained from selftest-output.txt; trainer SHA256 unchanged; not rerun", flush=True)
    baseline = t2_baseline(ref, ns)
    t2(ns, baseline)
    mutant = mutation_namespace(ns, source)
    original_prefix = ast.get_source_segment(source, function(source, "_teacher_prefix"))
    needle = "limit = 4.0 * TEACHER_H"
    assert original_prefix.count(needle) == 1
    mutated_prefix = original_prefix.replace(needle, needle + " + 2.0  # intentional killer")
    execute(tree(mutated_prefix).body, mutant, "<T2-overlong-prefix-killer>")
    killer("T2 4H+2 killer", lambda: t2(mutant, baseline), expected_error="raw length error=")
    distance = t3(ns)
    print(f"T3: PASS full translated g preserved; max raw g-to-prefix-end distance={distance:.6f} >4", flush=True)
    mutant = mutation_namespace(ns, source)
    original_batch = ast.get_source_segment(source, function(source, "make_batch"))
    assert original_batch.count("xy_to_obs_slot(tg.astype(np.float64))") == 1
    mutated_batch = original_batch.replace("xy_to_obs_slot(tg.astype(np.float64))",
                                            "xy_to_obs_slot(tt[:, -1].astype(np.float64))")
    execute(tree(mutated_batch).body, mutant, "<T3-near-goal-killer>")
    killer("T3 near-goal killer", lambda: t3(mutant))
    calls, n_r = t4(ns)
    print(f"T4: PASS H=3 vs unset; {N_BATCHES} batches, {n_r} real rows/batch; "
          f"RNG states and all {calls} calls/order/arguments identical", flush=True)
    t5(m9_source, source)
    t6()
    assert hashlib.sha256(M9.read_bytes()).hexdigest() == digest
    print("T1-T6: ALL PASS; n_tests=6 n_killers=3; M9 SHA256 unchanged", flush=True)
    if args.start_at == "T2":
        print("r1 executed: n_tests=5 n_killers=2; plus retained T1 and RNG killer evidence", flush=True)


if __name__ == "__main__":
    main()
