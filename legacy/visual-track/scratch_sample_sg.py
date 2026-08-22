"""Randomly sample (s, g) from rendered visual-pointmaze-medium (val) via the real WpmPipeline.
Prints hindsight metadata and saves [s | future slices | g] composite strips.
Run on jasmine:  OGBENCH_DATA_DIR=... WPM_CACHE_DIR=... MUJOCO_GL=osmesa .venv/bin/python sample_sg.py [seed]
"""
import os, sys
import numpy as np

sys.path.insert(0, "/archive/cymaxwelllee/fpo")
import wpm.data.pipeline as P

DS = "visual-pointmaze-medium-navigate-v0"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
outdir = "/archive/cymaxwelllee/fpo/outputs/sg_demo"
os.makedirs(outdir, exist_ok=True)

cfg = P.PipelineConfig(dataset_name=DS)
pipe = P.WpmPipeline(cfg, split="val")
rng = np.random.default_rng(seed)
b = pipe.sample_batch(rng, batch_size=4)

print("dataset:", DS, "| split: val | seed:", seed)
print("observation:", b.observation.shape, b.observation.dtype)
print("current s :", b.current.shape)
print("goal g    :", b.goal.shape, "| range", int(b.goal.min()), "-", int(b.goal.max()))
print("slices    :", b.slices.shape, "| offsets", b.slice_offsets)
print("actions   :", b.actions.shape)
print("steps_to_goal :", b.steps_to_goal.tolist(), "  <- hindsight distance (rows to the goal)")
print("goal_component:", b.goal_component.tolist(), "  <- which mixture arm the goal came from")
print("row / goal_row:", b.row.tolist(), "/", b.goal_row.tolist())
print("traj_id       :", b.traj_id.tolist(), "  <- s and its goal share a trajectory (hindsight)")

def up(frame, k=3):
    return np.ascontiguousarray(np.kron(frame, np.ones((k, k, 1), dtype=frame.dtype)))

try:
    from PIL import Image
    B = b.current.shape[0]
    for i in range(B):
        frames = [b.current[i]] + [b.slices[i, j] for j in range(b.slices.shape[1])] + [b.goal[i]]
        frames = [up(f) for f in frames]
        H, W = frames[0].shape[:2]
        pad = 6
        canvas = Image.new("RGB", (len(frames) * W + (len(frames) - 1) * pad, H), (240, 240, 240))
        x = 0
        for f in frames:
            canvas.paste(Image.fromarray(f), (x, 0)); x += W + pad
        canvas.save(f"{outdir}/sample{i}_s_slices_g.png")
    print("SAVED_PNG", outdir, "(each: [ s | future slices | g ])")
except Exception as e:
    print("PIL failed:", repr(e))
    np.savez(f"{outdir}/sg.npz", current=b.current, goal=b.goal, slices=b.slices)
    print("SAVED_NPZ", f"{outdir}/sg.npz")
