"""Visualize M4's REAL e_target trajectory: the s->g path (variable length, ENDS at goal).
Extracts obs frames row..goal_row from the same trajectory, subsamples for display.
Single labeled PNG."""
import os, sys, glob
import numpy as np

sys.path.insert(0, "/archive/cymaxwelllee/fpo")
import wpm.data.pipeline as P
from PIL import Image, ImageDraw, ImageFont


def get_font(size):
    cands = []
    try:
        import matplotlib
        cands.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"))
        cands.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans.ttf"))
    except Exception:
        pass
    cands += sorted(glob.glob("/usr/share/fonts/**/*.ttf", recursive=True))
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


DS = "visual-pointmaze-medium-navigate-v0"
cfg = P.PipelineConfig(dataset_name=DS)
pipe = P.WpmPipeline(cfg, split="val")
obs = pipe.arrays["observations"]  # (N,64,64,3) uint8 rendered frames
rng = np.random.default_rng(1)
b = pipe.sample_batch(rng, batch_size=3)

NSHOW = 6  # frames shown per trajectory (subsampled; always includes s and g)
samples = []
for i in range(b.current.shape[0]):
    r0, r1 = int(b.row[i]), int(b.goal_row[i])
    length = r1 - r0  # == steps_to_goal
    idx = np.unique(np.linspace(r0, r1, min(NSHOW, length + 1)).round().astype(int))
    samples.append((obs[idx], (idx - r0).tolist(), int(b.steps_to_goal[i]), int(b.traj_id[i])))

K = 4
FS = 64 * K
pad = 12
titleh = 44
rowlabelw = 165
maxcols = max(len(s[0]) for s in samples)
B = len(samples)
steplabelh = 26

W = rowlabelw + maxcols * FS + (maxcols - 1) * pad + pad
H = titleh + B * (FS + steplabelh + pad) + pad
canvas = Image.new("RGB", (W, H), (250, 250, 250))
d = ImageDraw.Draw(canvas)
ftitle, frow, fstep = get_font(21), get_font(15), get_font(14)

d.text((pad, 10), "M4 e_target = the s->g TRAJECTORY (variable length, ENDS at the goal). blue=s  red=g.",
       font=ftitle, fill=(20, 20, 20))

def up(frame):
    return Image.fromarray(np.ascontiguousarray(np.kron(frame, np.ones((K, K, 1), dtype=frame.dtype))))

for i, (frames, offs, s2g, tid) in enumerate(samples):
    y = titleh + i * (FS + steplabelh + pad)
    d.text((pad, y + FS // 2 - 20),
           f"sample {i}\ngoal {s2g}\nsteps away\n(traj {tid})", font=frow, fill=(20, 20, 20))
    n = len(frames)
    for j in range(n):
        x = rowlabelw + j * (FS + pad)
        canvas.paste(up(frames[j]), (x, y))
        is_s, is_g = (j == 0), (j == n - 1)
        border = (25, 80, 190) if is_s else ((185, 25, 25) if is_g else None)
        if border:
            d.rectangle([x, y, x + FS - 1, y + FS - 1], outline=border, width=3)
        lab = "s (now)" if is_s else ("GOAL g" if is_g else f"+{offs[j]}")
        col = (25, 80, 190) if is_s else ((185, 25, 25) if is_g else (90, 90, 90))
        d.text((x + 4, y + FS + 4), lab, font=fstep, fill=col)

outp = "/archive/cymaxwelllee/fpo/outputs/sg_demo/traj_to_goal.png"
canvas.save(outp)
print("SAVED", outp, canvas.size, "| trajectory lengths:", [s[2] for s in samples])
