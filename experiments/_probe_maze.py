"""驗證 pointmaze 的座標→格子轉換：拿【真實走過的點】去對牆，落在牆裡的比例必須 ≈0。

⛔ 不手算、不憑印象：參數對不對，讓資料自己說。
"""
import os, numpy as np
D = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

MAPS = {}
MAPS["medium"] = [
    [1,1,1,1,1,1,1,1],[1,0,0,1,1,0,0,1],[1,0,0,1,0,0,0,1],[1,1,0,0,0,1,1,1],
    [1,0,0,1,0,0,0,1],[1,0,1,0,0,1,0,1],[1,0,0,0,1,0,0,1],[1,1,1,1,1,1,1,1]]
# large 從套件讀，避免抄錯
import inspect, re
from ogbench.locomaze import maze as mz
src = inspect.getsource(mz)
for name in ("large", "giant"):
    m = re.search(r"elif self\._maze_type == '%s':\s*\n\s*maze_map = \[(.*?)\]\s*\n\s*elif" % name, src, re.S)
    if m:
        rows = re.findall(r"\[([0-9,\s]+)\]", m.group(1))
        MAPS[name] = [[int(v) for v in r.split(",") if v.strip()] for r in rows]

for mt in ("medium", "large"):
    M = np.array(MAPS[mt]); print(f"\n=== {mt}  maze_map {M.shape} ===")
    for row in M: print("   " + "".join("█" if v else "." for v in row))
    env = f"pointmaze-{mt}-stitch-v0"
    p = f"{D}/{env}.npz"
    if not os.path.exists(p):
        print(f"   ⛔ 沒有 {p}"); continue
    obs = np.load(p)["observations"][:, :2].astype(np.float64)
    for unit, off in ((4.0, 4), (1.0, 1), (2.0, 2), (4.0, 2)):
        i = ((obs[:, 1] + off + 0.5 * unit) / unit).astype(np.int64)
        j = ((obs[:, 0] + off + 0.5 * unit) / unit).astype(np.int64)
        oob = (i < 0) | (i >= M.shape[0]) | (j < 0) | (j >= M.shape[1])
        ii, jj = np.clip(i, 0, M.shape[0]-1), np.clip(j, 0, M.shape[1]-1)
        inwall = (M[ii, jj] == 1) & ~oob
        print(f"   unit={unit} off={off}:  落在牆 {inwall.mean()*100:6.2f}%   出界 {oob.mean()*100:6.2f}%"
              f"   x∈[{obs[:,0].min():.1f},{obs[:,0].max():.1f}] y∈[{obs[:,1].min():.1f},{obs[:,1].max():.1f}]")
