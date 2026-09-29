"""External, paired flag-off proof for the official s33 flat R1 rollout.

The two source files are loaded through the same loader. Only the unrelated
top-level multi-arm evaluation is omitted; rollout and policy function bodies
are compiled verbatim from each source file.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/scratch_lacot_rollout.py"
sys.path.insert(0, str(WORK))
import run  # noqa: E402 -- use the gate's exact checkpoint and code hashes


def questions(value):
    try:
        pairs = [tuple(map(int, part.split(":"))) for part in value.split(",")]
        if (len(pairs) != 3 or len(set(pairs)) != 3
                or any(len(p) != 2 or not 1 <= p[0] <= 5 or not 0 <= p[1] < 50 for p in pairs)):
            raise ValueError("expected three distinct task:episode pairs (task 1..5, episode 0..49)")
        return pairs
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def source_module(path, name):
    """Import source setup without launching its unrelated main evaluation."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    cut = next(i for i, node in enumerate(tree.body)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "RT_GATE" for t in node.targets))
    tree.body = tree.body[:cut]
    ast.fix_missing_locations(tree)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    if not module.LOAD_CKPT or module.LOAD_EMA != 1 or getattr(module, "ORACLE_COLLECTOR", None) is not None:
        raise RuntimeError("Expected loaded s33 EMA with flag-off official path")
    if (module.ENV_NAME != "pointmaze-large-stitch-v0" or module.device != os.environ["FLAGOFF_DEVICE"]
            or module.SUBGOAL or module.BON_N or module.GRAD_REFINE or module.FINISH_R):
        raise RuntimeError("Unexpected environment, device, or policy variant")
    import torch
    if next(module.ahead.parameters()).dtype != torch.float32:
        raise RuntimeError("Expected float32 action head on both sources")
    return module


def array_digest(value):
    import numpy as np
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.digest()


class TracedEnv:
    """Identical external reset pin and trajectory trace for both sources."""

    def __init__(self, inner, task, episode):
        self.inner = inner
        self.task = task
        self.episode = episode
        self.seed = 1000 * task + episode + run.SWEEP_SALT
        self.trace = hashlib.sha256()
        self.resets = 0
        self.steps = 0

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def reset(self, *, seed, options):
        import numpy as np
        if (self.resets != 0 or seed != 1000 * self.task + self.episode
                or options.get("task_id") != self.task or options.get("render_goal") is not False):
            raise RuntimeError("Official rollout reset differs from selected question")
        self.resets += 1
        np.random.seed(self.seed)
        self.inner.action_space.seed(self.seed)
        obs, info = self.inner.reset(seed=self.seed, options=options)
        self.trace.update(json.dumps(["reset", self.seed, self.task, self.episode]).encode())
        self.trace.update(array_digest(obs))
        self.trace.update(array_digest(info["goal"]))
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.inner.step(action)
        self.steps += 1
        self.trace.update(array_digest(action))
        self.trace.update(array_digest(obs))
        self.trace.update(json.dumps([float(reward), bool(term), bool(trunc),
                                      bool(info.get("success", False))]).encode())
        return obs, reward, term, trunc, info


def trajectory(module, task, episode):
    """Select one real outer-loop pair; all policy/rollout code stays intact."""
    original_env = module.env
    native_range = range
    traced = TracedEnv(original_env, task, episode)
    calls = []

    def selected_range(*args):
        if sys._getframe(1).f_code is module.rollout.__code__:
            calls.append(args)
            if len(calls) == 1 and args == (1, module.N_TASKS + 1):
                return (task,)
            if len(calls) == 2 and args == (module.SEEDS,):
                return (episode,)
            raise RuntimeError(f"Unexpected official rollout loop: {args}")
        return native_range(*args)

    module.env = traced
    module.range = selected_range
    try:
        module.rollout(1, True, f"flag-off task={task} episode={episode}")
    finally:
        module.env = original_env
        del module.range
    if calls != [(1, module.N_TASKS + 1), (module.SEEDS,)] or traced.resets != 1 or traced.steps < 1:
        raise RuntimeError("Incomplete official trajectory")
    return traced.trace.hexdigest(), traced.steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=questions, default=questions("1:3,2:7,4:11"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", type=Path, default=WORK / "smoke/flagoff-proof.json")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    # Match run.py's isolated launch config. A parent LACOT_ORACLE_* cannot
    # silently turn either side into the opt-in oracle path.
    for key in tuple(os.environ):
        if key.startswith("LACOT_"):
            del os.environ[key]
    config = dict(ENV="pointmaze-large-stitch-v0", CONS="self", K=8, COND=256,
                  CHUNK=4, TCAP=128, ENC_OBJ="recon_ictr", LEARNED_REFINE=0,
                  COND_DROP=0.1, BC_INDEP=1, TEACHER_MIX=0.5, WARMUP=500,
                  DEC_START="soft", LOAD_EMA=1, CONT_TRAIN=0, DEV_EVAL=0,
                  PREREQ=0, EVAL_RS=1, GRAD_REFINE=0, SUBGOAL="", FINISH_R=0,
                  INTENT="", U_SOURCE="flow", SUB_ESEL=0, INTENT_GUID_W=0,
                  DEC_ANCHOR=0, BON_N=0, BOOT_DATA="", S1_FROM="", FLOW_PROBE=0,
                  DIAG_DUMP=0, LOAD_CKPT=run.CKPTS[33], SEED=33)
    os.environ.update({"LACOT_" + k: str(v) for k, v in config.items()})
    os.environ["FLAGOFF_DEVICE"] = args.device
    ckpt = ROOT / run.CKPTS[33]
    if run.sha(ckpt) != run.CKPT_SHA256[33]:
        raise ValueError("s33 checkpoint SHA256 mismatch")
    baseline = subprocess.check_output(["git", "show", "HEAD:experiments/scratch_lacot_rollout.py"], cwd=ROOT)
    current_sha = run.sha(SOURCE)
    results = {}
    # The temporary baseline lives beside the current source so its unchanged
    # __file__-relative checkpoint lookup resolves to the same repository.
    with tempfile.NamedTemporaryFile(dir=SOURCE.parent, prefix=".flagoff-head-", suffix=".py") as tmp:
        tmp.write(baseline)
        tmp.flush()
        for side, path in (("baseline", Path(tmp.name)), ("current", SOURCE)):
            module = source_module(path, f"_flagoff_{side}")
            try:
                results[side] = [trajectory(module, *pair) for pair in args.tasks]
            finally:
                sys.modules.pop(module.__name__, None)
                del module
                gc.collect()
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
    rows = []
    mismatches = []
    for pair, (bh, bs), (ch, cs) in zip(args.tasks, results["baseline"], results["current"]):
        rows.append(dict(task=pair[0], episode=pair[1],
                         baseline_trajectory_sha256=bh, current_trajectory_sha256=ch))
        print(f"task={pair[0]} episode={pair[1]} baseline={bh} ({bs} steps) "
              f"current={ch} ({cs} steps)", flush=True)
        if bh != ch or bs != cs:
            mismatches.append(f"{pair[0]}:{pair[1]} hash {bh[:12]} != {ch[:12]}; steps {bs} != {cs}")
    if mismatches:
        raise RuntimeError("Flag-off trajectory mismatch: " + "; ".join(mismatches))
    if run.sha(SOURCE) != current_sha:
        raise RuntimeError("Current rollout source changed during comparison")
    proof = dict(note="External reset pin and step trace; official flag-off rollout on both sources",
                 device=args.device,
                 baseline_source_sha256=hashlib.sha256(baseline).hexdigest(),
                 current_source_sha256=current_sha, code_hashes=run.code_hashes(),
                 ckpt_sha256=run.CKPT_SHA256[33], pairs=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(proof, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
