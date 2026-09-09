#!/usr/bin/env python
"""drift_analysis 專用畫圖：分位數帶折線圖（p25-p75 半透明色帶 + p50 實線）。

⛔ 不修改任何既有檔案。這台沒裝 matplotlib（見 wv_common.py 開頭註記），沿用同一個
慣例 —— 全部用 Pillow 手畫，只 import wv_common 的色盤常數（hex_to_rgb /
PALETTE_CATEGORICAL / INK_*／GRIDLINE／AXIS／SURFACE），不 import 也不呼叫
wv_common 裡帶底線的私有函式（自己內聯一份等價的 `_font()`，2 行，不算重複邏輯）。

現有 wv_common.draw_loss_curve 也能畫多線，但它把 x 軸標籤寫死成「train step」
（wv_common.py 裡 `dr.text((ml, H - 18), "train step", ...)`），不能借用來畫
「chunk index」——這是新寫一個小函式而不是硬套現成函式的原因，寫在這裡供對帳。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> walk_verify/
import wv_common as wv  # noqa: E402


def _font():
    from PIL import ImageFont
    return ImageFont.load_default()


def _finite_xy(xs, ys1, ys2=None, ys3=None):
    """回傳只保留『所有給定序列在該點都是有限值』的 (xs, ys1[, ys2, ys3]) 子集。

    n 隨 chunk index 遞減到 0 時（例如 completed 組後段沒有題目撐到那麼晚），
    band_stats() 對缺樣本的格點填 NaN——PIL 的 polygon/line 不能吃 NaN 座標
    （會畫出座標爆掉的雜線，不是乾淨地略過），必須先在畫圖前濾掉，只畫『真的有
    樣本』的那一段，不是把整條線硬拉到底。
    """
    xs = np.asarray(xs, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys1)
    if ys2 is not None:
        mask &= np.isfinite(ys2)
    if ys3 is not None:
        mask &= np.isfinite(ys3)
    out = [xs[mask], np.asarray(ys1, dtype=float)[mask]]
    if ys2 is not None:
        out.append(np.asarray(ys2, dtype=float)[mask])
    if ys3 is not None:
        out.append(np.asarray(ys3, dtype=float)[mask])
    return out


def draw_percentile_band(groups, out_path, title, xlabel, ylabel):
    """groups: list of dict(label=str, color_idx=int, xs=[...], p25=[...], p50=[...], p75=[...])

    每組畫一條 p50 實線 + p25~p75 半透明色帶（多邊形填色，經 RGBA overlay 疊到底圖後
    轉回 RGB，避免直接在 "RGB" 模式 Image 上用帶 alpha 的 fill 出錯）。ylabel 併進
    title 一行印（不另外在左上角疊字）——避免長 ylabel 跟 title 在窄左邊界裡疊字。
    """
    from PIL import Image, ImageDraw

    W, H = 980, 520
    ml, mr, mt, mb = 84, 230, 46, 56
    pw, ph = W - ml - mr, H - mt - mb
    ft = _font()
    full_title = f"{title}  [{ylabel}]"

    filt = [_finite_xy(g["xs"], g["p50"], g["p25"], g["p75"]) for g in groups]
    allx = np.concatenate([f[0] for f in filt if len(f[0])])
    ally_hi = np.concatenate([f[3] for f in filt if len(f[0])])
    x0, x1 = float(allx.min()), float(allx.max())
    if x1 <= x0:
        x1 = x0 + 1.0
    y0 = 0.0
    y1 = float(ally_hi.max()) * 1.12 if len(ally_hi) else 1.0
    if not np.isfinite(y1) or y1 <= y0:
        y1 = y0 + 1e-6

    def px(x):
        return ml + pw * (x - x0) / (x1 - x0)

    def py(y):
        return mt + ph * (1 - (y - y0) / (y1 - y0))

    base = Image.new("RGB", (W, H), wv.hex_to_rgb(wv.SURFACE))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odr = ImageDraw.Draw(overlay)
    for i, (g, (xs, p50, p25, p75)) in enumerate(zip(groups, filt)):
        col = wv.hex_to_rgb(wv.PALETTE_CATEGORICAL[g.get("color_idx", i) % 8])
        band_pts = [(px(x), py(v)) for x, v in zip(xs, p75)] + \
                   [(px(x), py(v)) for x, v in zip(xs[::-1], p25[::-1])]
        if len(band_pts) >= 3:
            odr.polygon(band_pts, fill=col + (48,))
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    dr = ImageDraw.Draw(base)

    for f in np.linspace(0, 1, 6):
        yy = mt + ph * (1 - f)
        dr.line([(ml, yy), (ml + pw, yy)], fill=wv.hex_to_rgb(wv.GRIDLINE))
        dr.text((6, yy - 6), f"{y0 + f * (y1 - y0):.3g}", fill=wv.hex_to_rgb(wv.INK_MUTED), font=ft)
    for f in np.linspace(0, 1, 6):
        xx = ml + pw * f
        dr.line([(xx, mt), (xx, mt + ph)], fill=wv.hex_to_rgb(wv.GRIDLINE))
        dr.text((xx - 12, mt + ph + 6), f"{x0 + f * (x1 - x0):.0f}",
                 fill=wv.hex_to_rgb(wv.INK_MUTED), font=ft)

    for i, (g, (xs, p50, p25, p75)) in enumerate(zip(groups, filt)):
        col = wv.hex_to_rgb(wv.PALETTE_CATEGORICAL[g.get("color_idx", i) % 8])
        line_pts = [(px(x), py(v)) for x, v in zip(xs, p50)]
        if len(line_pts) >= 2:
            dr.line(line_pts, fill=col, width=2)
        n_pts = len(xs)
        ly = mt + 8 + i * 30
        dr.line([(ml + pw + 14, ly), (ml + pw + 34, ly)], fill=col, width=3)
        dr.text((ml + pw + 38, ly - 6), g["label"], fill=wv.hex_to_rgb(wv.INK_SECONDARY), font=ft)
        dr.text((ml + pw + 38, ly + 8), f"(p50 line, p25-p75 band, {n_pts} pts)",
                 fill=wv.hex_to_rgb(wv.INK_MUTED), font=ft)

    dr.line([(ml, mt), (ml, mt + ph)], fill=wv.hex_to_rgb(wv.AXIS), width=2)
    dr.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=wv.hex_to_rgb(wv.AXIS), width=2)
    dr.text((ml, 14), full_title, fill=wv.hex_to_rgb(wv.INK_PRIMARY), font=ft)
    dr.text((ml, H - 18), xlabel, fill=wv.hex_to_rgb(wv.INK_MUTED), font=ft)
    base.save(out_path)


if __name__ == "__main__":
    # 極小合成資料自檢（不碰 MuJoCo／不算「CPU 重工作」，純畫圖）：兩組人工分位數帶，
    # 存到 /tmp scratch 供視覺核對用完即丟，不留在 repo 裡。
    xs = np.arange(0, 20)
    rng = np.random.default_rng(0)
    g1 = dict(label="fake completed", color_idx=0, xs=xs,
              p25=xs * 0.05, p50=xs * 0.08, p75=xs * 0.12)
    g2 = dict(label="fake stuck", color_idx=7, xs=xs,
              p25=xs * 0.08, p50=xs * 0.15, p75=xs * 0.25)
    out = "/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/f61430f0-e6b8-4dfe-93de-8859ae65c8e4/scratchpad/_smoke_band.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw_percentile_band([g1, g2], out, "smoke test", "chunk index j", "xy drift (m)")
    print("saved smoke plot:", out)
