#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
indoor_matplot_pretty.py
美观版：楼层底图(米范围) + 轨迹(米)，不做坐标映射，仅可选 y 翻转。
特点：
- 轨迹更顺滑（Chaikin 平滑），抗锯齿、圆角端点
- 霓虹描边/暗背景衬托，线条更“立”
- 三种风格：mono(纯色)、gradient(渐变)、multi(每条不同色)
- 轴/刻度/标题/图例全隐藏，导出零留白

依赖：numpy、matplotlib（无需 scipy/seaborn）
"""

import json, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------- IO ----------
def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_floor_info(path: Path):
    obj = load_json(path)
    if isinstance(obj, dict):
        return float(obj["map_info"]["width"]), float(obj["map_info"]["height"])
    if isinstance(obj, list) and obj:
        return float(obj[0]["map_info"]["width"]), float(obj[0]["map_info"]["height"])
    raise TypeError("floor_info.json 应为 dict 或非空 list")

def load_xy_from_txt(txt_path: Path) -> List[List[float]]:
    pts = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().split()
            if len(s) >= 4 and s[1] == "TYPE_WAYPOINT":
                try:
                    x, y = float(s[2]), float(s[3])
                    pts.append([x, y])
                except:
                    pass
    return pts

def find_txt_dir(floor_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit and explicit.exists() and explicit.is_dir():
        if any(p.suffix.lower()==".txt" for p in explicit.iterdir()):
            return explicit
        return explicit
    cand = floor_dir / "path_data_files"
    if cand.exists() and cand.is_dir() and any(p.suffix.lower()==".txt" for p in cand.iterdir()):
        return cand
    if any(p.suffix.lower()==".txt" for p in floor_dir.iterdir()):
        return floor_dir
    for sub in floor_dir.iterdir():
        if sub.is_dir() and any(p.suffix.lower()==".txt" for p in sub.iterdir()):
            return sub
    return None

# ---------- 美化工具 ----------
def color_for_name(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{h[4:10]}"

def chaikin_smooth(points: np.ndarray, n_iter: int = 2) -> np.ndarray:
    """无依赖平滑：Chaikin corner cutting。n_iter=0 则原样返回。"""
    if n_iter <= 0 or len(points) < 3:
        return points
    pts = points.astype(float)
    for _ in range(n_iter):
        Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        R = 0.25 * pts[:-1] + 0.75 * pts[1:]
        inter = np.empty((Q.shape[0] + R.shape[0], 2), dtype=float)
        inter[0::2] = Q
        inter[1::2] = R
        pts = np.vstack([pts[0], inter, pts[-1]])
    return pts

def decimate_minstep(points: np.ndarray, min_step: float = 0.0) -> np.ndarray:
    """按最小步长（米）抽稀，减少重合抖动。"""
    if min_step <= 0 or len(points) < 2:
        return points
    keep = [0]
    for i in range(1, len(points)):
        d2 = np.sum((points[i] - points[keep[-1]])**2)
        if d2 >= min_step * min_step:
            keep.append(i)
    return points[keep]

def add_gradient_line(ax, xs, ys, cmap="turbo", lw=5.0, alpha=1.0,
                      underglow_width=3.5, underglow_alpha=0.6):
    """渐变线：按照路径累计长度着色；先铺一层暗色“底光”再上色。"""
    from matplotlib.collections import LineCollection
    # underglow
    ax.plot(xs, ys, linewidth=lw + underglow_width,
            color=(0, 0, 0, underglow_alpha),
            solid_capstyle='round', solid_joinstyle='round', antialiased=True, zorder=2)
    # segments
    P = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([P[:-1], P[1:]], axis=1)
    d = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate(([0], np.cumsum(d)))
    t = (s[:-1] + s[1:]) * 0.5
    if t.ptp() == 0:
        c = np.zeros_like(t)
    else:
        c = (t - t.min()) / t.ptp()
    lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha,
                        capstyle='round', joinstyle='round', antialiased=True, zorder=3)
    lc.set_array(c)
    ax.add_collection(lc)
    return lc

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--txt-dir", type=Path, default=None)
    ap.add_argument("--y-flip", type=int, default=1, help="y -> H - y（默认 1）")
    ap.add_argument("--sample-every", type=int, default=1, help="抽稀步长（整数，默认 1）")
    ap.add_argument("--min-step", type=float, default=0.10, help="最小采样步长(米)；0 关闭（默认 0.10m）")
    ap.add_argument("--smooth", type=int, default=2, help="Chaikin 平滑迭代次数（默认 2）")

    # 视觉
    ap.add_argument("--style", choices=["mono", "gradient", "multi"], default="mono",
                    help="轨迹风格：mono=纯色；gradient=渐变；multi=每条不同色")
    ap.add_argument("--color", type=str, default="#00e5ff", help="mono 风格的颜色（默认霓虹青）")
    ap.add_argument("--line-width", type=float, default=6.0, help="线宽（默认 6）")
    ap.add_argument("--line-alpha", type=float, default=0.95, help="线条透明度（默认 0.95）")
    ap.add_argument("--underglow", type=float, default=4.5, help="底部描边宽度增量（像素，0 关闭，默认 4.5）")
    ap.add_argument("--underglow-alpha", type=float, default=0.55, help="底部描边透明度（默认 0.55）")
    ap.add_argument("--cmap", type=str, default="turbo", help="gradient 用的 colormap（默认 turbo）")

    # 画布
    ap.add_argument("--figsize", type=str, default="10,7.5")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--image-alpha", type=float, default=0.85)
    ap.add_argument("--darken", type=float, default=0.08, help="整体压暗底图（0~1，默认 0.08）")

    ap.add_argument("--out", type=Path, default=Path("pretty_overlay.png"))
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    floor_dir = args.floor_dir
    FLOORINFO = floor_dir / "floor_info.json"
    FLOORIMG  = floor_dir / "floor_image.png"

    if not FLOORINFO.exists():
        raise FileNotFoundError(f"缺少 floor_info.json：{FLOORINFO}")
    if not FLOORIMG.exists():
        raise FileNotFoundError(f"缺少 floor_image.png：{FLOORIMG}")

    # 地图范围（米）
    W, H = read_floor_info(FLOORINFO)

    # 读取底图
    img = plt.imread(str(FLOORIMG))

    # 画布
    fw, fh = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)

    # 铺底图（可选择整体稍压暗）
    ax.imshow(img, extent=[0, W, 0, H], origin="lower", alpha=args.image_alpha,
              interpolation="bilinear", zorder=0)
    if args.darken > 0:
        ax.imshow(np.zeros_like(img), extent=[0, W, 0, H], origin="lower",
                  alpha=args.darken, zorder=1)

    # 找 TXT
    txt_dir = find_txt_dir(floor_dir, None)
    if txt_dir:
        txts = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
        for txt in txts:
            pts = load_xy_from_txt(txt)
            if len(pts) < 2:
                continue
            if args.sample_every > 1:
                pts = pts[::args.sample_every]
            pts = np.asarray(pts, dtype=float)
            if args.y_flip:
                pts[:,1] = H - pts[:,1]
            pts = decimate_minstep(pts, args.min_step)
            pts = chaikin_smooth(pts, args.smooth)
            xs, ys = pts[:,0], pts[:,1]

            if args.style == "gradient":
                add_gradient_line(ax, xs, ys, cmap=args.cmap,
                                  lw=args.line_width, alpha=args.line_alpha,
                                  underglow_width=args.underglow, underglow_alpha=args.underglow_alpha)
            else:
                if args.underglow > 0:
                    ax.plot(xs, ys, linewidth=args.line_width + args.underglow,
                            color=(0,0,0,args.underglow_alpha),
                            solid_capstyle='round', solid_joinstyle='round',
                            antialiased=True, zorder=2)
                color = args.color if args.style == "mono" else color_for_name(txt.name)
                ax.plot(xs, ys, linewidth=args.line_width, alpha=args.line_alpha,
                        color=color, solid_capstyle='round', solid_joinstyle='round',
                        antialiased=True, zorder=3)

    # 收口：无坐标轴、等比例、零留白
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(0,0,1,1)
    plt.margins(0)

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0)
    print("✅ 已输出：", args.out.resolve())
    if args.preview:
        plt.show()

if __name__ == "__main__":
    main()
#
# python indoor_matplot_simple.py --floor-dir .\site1\B1 --style mono --color "#ffffff" --line-width 1.8 --underglow 0.8 --underglow-alpha 0.35 --smooth 2 --min-step 0.05 --image-alpha 0.82 --darken 0.12 --dpi 320 --out simple.png


# python indoor_matplot_simple.py --floor-dir .\site1\B1 --out pretty_mono.png
