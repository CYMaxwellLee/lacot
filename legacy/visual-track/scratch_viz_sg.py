"""One clear LABELED image of random (s,g) samples from visual-pointmaze-medium val.
Single PNG (so Telegram shows it as one image, not a 3x3 album grid)."""
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
    cands += sorted(glob.glob("/usr/share/fonts/**/DejaVuSans*.ttf", recursive=True))
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
rng = np.random.default_rng(1)
b = pipe.sample_batch(rng, batch_size=3)

K = 4
FS = 64 * K
pad = 12
titleh = 40
labelh = 30
rowlabelw = 165
cols = ["s  (now)", "future +8", "+12", "+16", "GOAL  g"]
ncol = len(cols)
B = b.current.shape[0]

W = rowlabelw + ncol * FS + (ncol - 1) * pad + pad
H = titleh + labelh + B * (FS + pad) + pad
canvas = Image.new("RGB", (W, H), (250, 250, 250))
d = ImageDraw.Draw(canvas)
ftitle, fcol, frow = get_font(22), get_font(19), get_font(15)

d.text((pad, 9), "visual-pointmaze-medium: random (s, g).  g is a HINDSIGHT future state on the SAME trajectory.",
       font=ftitle, fill=(20, 20, 20))

x0 = rowlabelw
for j, c in enumerate(cols):
    x = x0 + j * (FS + pad)
    col = (185, 25, 25) if "GOAL" in c else ((25, 80, 190) if "now" in c else (110, 110, 110))
    d.text((x + 4, titleh + 4), c, font=fcol, fill=col)

def up(frame):
    return Image.fromarray(np.ascontiguousarray(np.kron(frame, np.ones((K, K, 1), dtype=frame.dtype))))

for i in range(B):
    y = titleh + labelh + i * (FS + pad)
    d.text((pad, y + FS // 2 - 26),
           f"sample {i}\ngoal is\n{int(b.steps_to_goal[i])} steps\nahead\n(traj {int(b.traj_id[i])})",
           font=frow, fill=(20, 20, 20))
    frames = [b.current[i], b.slices[i, 0], b.slices[i, 1], b.slices[i, 2], b.goal[i]]
    for j, f in enumerate(frames):
        x = x0 + j * (FS + pad)
        canvas.paste(up(f), (x, y))
        border = (185, 25, 25) if j == ncol - 1 else ((25, 80, 190) if j == 0 else None)
        if border:
            d.rectangle([x, y, x + FS - 1, y + FS - 1], outline=border, width=3)

outp = "/archive/cymaxwelllee/fpo/outputs/sg_demo/sg_labeled.png"
canvas.save(outp)
print("SAVED", outp, canvas.size)
