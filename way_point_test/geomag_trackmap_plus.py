#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geomag_trackmap.py — Geomagnetic "track-style" map (Matplotlib, CRS.Simple)

Render colored trajectories on top of the floor plan. Color encodes magnetic
intensity (or component). Designed for paper-quality figures.

Usage examples:
  python geomag_trackmap.py --floor-dir .\\site1\\B1 --source cal --stat mag \
    --q 10,90 --lw 6 --alpha 0.95 --smooth 5 --break-dist-px 18 --out b1_track.png

  python geomag_trackmap.py --floor-dir .\\site1\\F1 --source uncal_debiased \
    --stat mag --vminmax 20,70 --lw 6 --alpha 0.95 --out f1_track_20_70.png
"""

import os, json, math, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

# ---------- Matplotlib aesthetics (English, paper style) ----------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------- I/O ----------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_floor_info(path: Path) -> Dict[str, Any]:
    obj = load_json(path)
    if isinstance(obj, dict):
        return {"map_w": float(obj["map_info"]["width"]),
                "map_h": float(obj["map_info"]["height"]),
                "raw": obj}
    if isinstance(obj, list) and obj:
        return {"map_w": float(obj[0]["map_info"]["width"]),
                "map_h": float(obj[0]["map_info"]["height"]),
                "raw": obj}
    raise TypeError("floor_info.json must be a dict or a non-empty list")

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

# ---------- affine ----------
def parse_affine_from_string(s: str):
    try:
        a,b,c,d,e,f = [float(x.strip()) for x in s.split(",")]
        return (a,b,c,d,e,f)
    except Exception:
        return None

def compose_affine(scale=(1.0,1.0), theta_deg=0.0, translate=(0.0,0.0)):
    sx, sy = scale
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    a = c*sx; b = -s*sy
    c2 = s*sx; d =  c*sy
    e, f = translate
    return (a,b,c2,d,e,f)

def try_affine_from_floorinfo(fi_raw: Any):
    def norm6(v):
        if isinstance(v, list) and len(v)==6:
            return tuple(float(x) for x in v)
        if isinstance(v, list) and len(v)==2 and isinstance(v[0], list) and len(v[0])==3:
            a,b,e = v[0]; c,d,f = v[1]
            return (float(a),float(b),float(c),float(d),float(e),float(f))
        return None
    raw = fi_raw
    t = (raw.get("transform") if isinstance(raw, dict) else None) or {}
    if isinstance(t, dict) and {"a","b","c","d","e","f"}.issubset(t.keys()):
        return (float(t["a"]), float(t["b"]), float(t["c"]), float(t["d"]), float(t["e"]), float(t["f"]))
    for k in ("affine","matrix"):
        if k in t:
            v = norm6(t[k])
            if v: return v
    if "scale" in t and "translate" in t:
        sx, sy = t.get("scale",[1,1]) or [1,1]
        tx, ty = t.get("translate",[0,0]) or [0,0]
        theta = float(t.get("theta_deg",0.0))
        return compose_affine((float(sx),float(sy)), theta, (float(tx),float(ty)))
    mi = raw.get("map_info") if isinstance(raw, dict) else None
    if isinstance(mi, dict):
        ox, oy = 0.0, 0.0
        if "origin" in mi and isinstance(mi["origin"], (list,tuple)) and len(mi["origin"])>=2:
            ox, oy = float(mi["origin"][0]), float(mi["origin"][1])
        theta = float(mi.get("theta_deg", 0.0))
        if "pixel_per_meter" in mi:
            ppm = float(mi["pixel_per_meter"]); return compose_affine((ppm,ppm), theta, (ox,oy))
        if "meters_per_pixel" in mi:
            mpp = float(mi["meters_per_pixel"]); ppm = 1.0/mpp if mpp!=0 else 1.0
            return compose_affine((ppm,ppm), theta, (ox,oy))
    return None

def apply_affine_xy(x, y, A):
    a,b,c,d,e,f = A
    return a*x + b*y + e, c*x + d*y + f

# ---------- TXT parsing ----------
def parse_headers(txt_path: Path) -> Dict[str,str]:
    meta = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"): break
            for tok in line[1:].strip().split("\t"):
                if ":" in tok:
                    k,v = tok.split(":",1)
                    meta[k.strip()] = v.strip()
    return meta

def parse_waypoints_and_mags(txt_path: Path, source="cal"):
    wps = []
    mags_cal = []
    mags_uncal_pairs = []  # (t, bux,buy,buz, bbx,bby,bbz)
    mags_uncal = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                t = int(parts[0])
            except Exception:
                continue
            typ = parts[1]

            if typ == "TYPE_WAYPOINT" and len(parts) >= 4:
                try:
                    x = float(parts[2]); y = float(parts[3])
                    wps.append((t,x,y))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD" and len(parts) >= 5:
                try:
                    mags_cal.append((t, float(parts[2]), float(parts[3]), float(parts[4])))
                except Exception:
                    pass
                continue

            if typ == "TYPE_MAGNETIC_FIELD_UNCALIBRATED":
                if len(parts) >= 8:
                    try:
                        bux, buy, buz = float(parts[2]), float(parts[3]), float(parts[4])
                        bbx, bby, bbz = float(parts[5]), float(parts[6]), float(parts[7])
                        mags_uncal_pairs.append((t, bux, buy, buz, bbx, bby, bbz))
                    except Exception:
                        pass
                elif len(parts) >= 5:
                    try:
                        mags_uncal.append((t, float(parts[2]), float(parts[3]), float(parts[4])))
                    except Exception:
                        pass
                continue

    wps.sort(key=lambda x: x[0])
    mags_cal.sort(key=lambda x: x[0])
    mags_uncal.sort(key=lambda x: x[0])
    mags_uncal_pairs.sort(key=lambda x: x[0])

    mags = []
    if source == "cal" and mags_cal:
        mags = mags_cal
    elif source == "uncal" and mags_uncal:
        mags = mags_uncal
    elif source == "uncal_debiased" and mags_uncal_pairs:
        for (t, bux,buy,buz, bbx,bby,bbz) in mags_uncal_pairs:
            mags.append((t, bux-bbx, buy-bby, buz-bbz))
    elif mags_cal:
        mags = mags_cal

    return wps, mags, parse_headers(txt_path)

def interpolate_pos_for_times(waypoints, times, mode="linear"):
    if not waypoints:
        return [None]*len(times)
    ts = np.array([w[0] for w in waypoints], dtype=np.int64)
    xs = np.array([w[1] for w in waypoints], dtype=float)
    ys = np.array([w[2] for w in waypoints], dtype=float)
    out = []
    j = 0
    for t in times:
        while j+1 < len(ts) and ts[j+1] <= t:
            j += 1
        if j < len(ts)-1:
            t0,t1 = ts[j], ts[j+1]
            if t0 <= t <= t1:
                if mode == "linear":
                    r = (t - t0)/max(1,(t1 - t0))
                    out.append((xs[j]*(1-r)+xs[j+1]*r, ys[j]*(1-r)+ys[j+1]*r))
                    continue
                elif mode == "hold":
                    out.append((xs[j], ys[j])); continue
        if t < ts[0]:
            out.append((xs[0], ys[0]) if mode=="hold" else None)
        elif t > ts[-1]:
            out.append((xs[-1], ys[-1]) if mode=="hold" else None)
        else:
            out.append((xs[j], ys[j]))
    return out

# ---------- helpers ----------
def robust_minmax(values, q_low=5.0, q_high=95.0):
    vs = np.asarray(values, dtype=float)
    vs = vs[np.isfinite(vs)]
    if vs.size == 0:
        return 0.0, 1.0
    try:
        lo = np.quantile(vs, q_low/100.0, method="linear")
        hi = np.quantile(vs, q_high/100.0, method="linear")
    except TypeError:
        lo = np.quantile(vs, q_low/100.0, interpolation="linear")
        hi = np.quantile(vs, q_high/100.0, interpolation="linear")
    except AttributeError:
        lo = np.percentile(vs, q_low, interpolation="linear")
        hi = np.percentile(vs, q_high, interpolation="linear")
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(vs)), float(np.nanmax(vs))
    if hi <= lo: hi = lo + 1e-6
    return float(lo), float(hi)

def movavg(arr, k):
    """Simple moving average with reflection padding."""
    if k <= 1: return np.asarray(arr, dtype=float)
    x = np.asarray(arr, dtype=float)
    if x.size < k: return x
    pad = k//2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(k)/k
    return np.convolve(xp, kernel, mode="valid")

def build_segments_with_breaks(xs, ys, vals, break_dist_px=20.0):
    """Split polyline by large jumps so lines don't cross empty areas."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if xs.size < 2: return [], []
    dx = np.diff(xs); dy = np.diff(ys)
    dist = np.hypot(dx, dy)
    # indices where we should break AFTER this index
    breaks = np.where(dist > break_dist_px)[0]
    starts = np.r_[0, breaks+1]
    ends = np.r_[breaks+1, xs.size-1]
    segs_all, v_all = [], []
    for s,e in zip(starts, ends):
        if e-s+1 < 2: continue
        X = xs[s:e+1]; Y = ys[s:e+1]; V = vals[s:e+1]
        segs = np.stack([np.stack([X[:-1], Y[:-1]], axis=1),
                         np.stack([X[1:],  Y[1:]],  axis=1)], axis=1)
        vline = 0.5*(V[:-1] + V[1:])
        if vline.size:
            segs_all.append(segs)
            v_all.append(vline)
    return segs_all, v_all

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--txt-dir", type=Path, default=None)
    ap.add_argument("--source", type=str, default="cal",
                    choices=["cal","uncal","uncal_debiased"])
    ap.add_argument("--stat", type=str, default="mag", choices=["mag","bx","by","bz"])
    ap.add_argument("--interp", type=str, default="linear", choices=["linear","hold","skip"])
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--affine", type=str, default=None, help="manual a,b,c,d,e,f (pixel = A * meter)")
    ap.add_argument("--y-flip", type=int, default=1, help="y -> H - y (origin=lower)")

    # aesthetics
    ap.add_argument("--lw", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--cmap", type=str, default="inferno")
    ap.add_argument("--q", type=str, default="10,90", help="quantile clip (e.g., '10,90')")
    ap.add_argument("--vminmax", type=str, default=None, help="absolute vmin,vmax (e.g., '20,70')")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window on values")
    ap.add_argument("--break-dist-px", type=float, default=18.0, help="break segments if gap > this (px)")
    ap.add_argument("--show-waypoints", type=int, default=0)
    ap.add_argument("--scatter-every", type=int, default=60)

    # background/output
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--image-alpha", type=float, default=0.75)
    ap.add_argument("--no-geojson", action="store_true")
    ap.add_argument("--figsize", type=str, default="8.0,6.2")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--out", type=Path, default=Path("geomag_track.png"))

    args = ap.parse_args()

    floor_dir = args.floor_dir
    FLOORINFO = floor_dir / "floor_info.json"
    GEOJSON   = floor_dir / "geojson_map.json"
    FLOORIMG  = floor_dir / "floor_image.png"
    if not FLOORINFO.exists():
        raise FileNotFoundError(f"Missing floor_info.json: {FLOORINFO}")

    # floor meta
    fi = read_floor_info(FLOORINFO)
    W, H, fi_raw = fi["map_w"], fi["map_h"], fi["raw"]

    # affine
    A = None
    if args.affine:
        A = parse_affine_from_string(args.affine)
        print("✔ Using manual affine:", A if A else "(parse failed)")
    if A is None:
        A = try_affine_from_floorinfo(fi_raw)
        print("✔ Affine from floor_info:", A if A else "⚠️ Not found; using raw meters + optional y-flip")

    # TXT
    txt_dir = find_txt_dir(floor_dir, args.txt_dir)
    if not txt_dir:
        raise FileNotFoundError("No directory containing .txt files was found.")
    txts = sorted([p for p in txt_dir.iterdir() if p.suffix.lower()==".txt"])
    if not txts:
        raise FileNotFoundError("No .txt files in TXT directory.")

    # collect
    all_segments = []
    all_v = []
    meta_any = {}
    total_mag = []
    total_wp  = 0

    for txt in txts:
        wps, mags, meta = parse_waypoints_and_mags(txt, source=args.source)
        if len(wps) < 2 or len(mags)==0:
            continue
        meta_any.update(meta)

        times = [t for (t, *_r) in mags]
        pos = interpolate_pos_for_times(wps, times, mode=args.interp)
        xs, ys, vs = [], [], []
        for p, (t,bx,by,bz) in zip(pos, mags):
            if p is None: continue
            x, y = p
            if A is not None: x,y = apply_affine_xy(x,y,A)
            if args.y_flip:  y = H - y
            if not (0 <= x <= W and 0 <= y <= H):  # clip to canvas
                continue
            if args.stat == "mag": v = math.sqrt(bx*bx + by*by + bz*bz)
            elif args.stat == "bx": v = bx
            elif args.stat == "by": v = by
            else: v = bz
            xs.append(x); ys.append(y); vs.append(v)

        if len(xs) < 2: continue

        # optional subsample + smoothing
        if args.subsample > 1:
            xs = xs[::args.subsample]; ys = ys[::args.subsample]; vs = vs[::args.subsample]
        if args.smooth > 1:
            vs = movavg(vs, args.smooth)

        segs_list, vline_list = build_segments_with_breaks(xs, ys, vs, break_dist_px=args.break_dist_px)
        if vline_list:
            all_segments.extend(segs_list)
            all_v.extend(vline_list)
            total_mag.extend(vs)
            total_wp += len(wps)

    if not all_v:
        raise RuntimeError("No drawable segments (check affine/coordinates/timestamps).")

    segs = np.concatenate(all_segments, axis=0)
    vline = np.concatenate(all_v, axis=0)

    # color scale
    if args.vminmax:
        vmin, vmax = [float(x.strip()) for x in args.vminmax.split(",")]
    else:
        ql, qh = [float(x.strip()) for x in args.q.split(",")]
        vmin, vmax = robust_minmax(vline, q_low=ql, q_high=qh)
    print(f"Color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    # figure
    fw, fh = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)

    # background image
    if (not args.no_image) and FLOORIMG.exists():
        img = plt.imread(str(FLOORIMG))
        ax.imshow(img, extent=[0, W, 0, H], origin="lower", alpha=args.image_alpha)

    # light geojson fill (optional)
    if (not args.no_geojson) and GEOJSON.exists():
        try:
            gj = load_json(GEOJSON)
            def draw_geom(geom):
                t = geom.get("type")
                if t == "Polygon":
                    for ring in geom["coordinates"][:1]:
                        pts = np.array(ring, dtype=float)
                        ax.add_patch(Polygon(pts, closed=True,
                                             facecolor=(0.75,0.88,0.95,0.20),
                                             edgecolor=(0.25,0.25,0.25,0.6),
                                             linewidth=0.8))
                elif t == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        if not poly: continue
                        pts = np.array(poly[0], dtype=float)
                        ax.add_patch(Polygon(pts, closed=True,
                                             facecolor=(0.75,0.88,0.95,0.20),
                                             edgecolor=(0.25,0.25,0.25,0.6),
                                             linewidth=0.8))
            if isinstance(gj, dict):
                if gj.get("type") == "FeatureCollection":
                    for feat in gj.get("features", []):
                        geom = feat.get("geometry", {})
                        if geom: draw_geom(geom)
                elif gj.get("type") in ("Polygon","MultiPolygon"):
                    draw_geom(gj)
        except Exception as e:
            print("⚠️ GeoJSON parse/draw failed:", e)

    # --- stroke/halo underlay for better contrast ---
    halo = LineCollection(segs, linewidths=args.lw*1.8, colors="white", alpha=0.85,
                          capstyle='round', joinstyle='round', zorder=1)
    ax.add_collection(halo)

    # --- colored line on top ---
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, cmap=args.cmap, norm=norm,
                        linewidths=args.lw, alpha=args.alpha,
                        capstyle='round', joinstyle='round', zorder=2)
    lc.set_array(vline)
    ax.add_collection(lc)

    # optional sparse waypoints
    if args.show_waypoints:
        pts = segs.reshape(-1,2)
        pts = pts[::max(1,int(args.scatter_every))]
        ax.scatter(pts[:,0], pts[:,1], s=8, c="black", alpha=0.35, zorder=3, linewidths=0, marker='o')

    # axes and labels
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")

    site = meta_any.get("SiteName", meta_any.get("SiteID", "Site"))
    floor = meta_any.get("FloorName", "Floor")
    # ensure English-only title: if contains non-ascii, drop it
    def ascii_only(s): return "".join(ch for ch in str(s) if ord(ch) < 128)
    title = f"{ascii_only(site)} — {ascii_only(floor)}   |   samples: {len(total_mag)}"
    ax.set_title(title)

    # colorbar with neat ticks
    cbar = fig.colorbar(lc, ax=ax, shrink=0.92, pad=0.01)
    label_unit = r"$|\mathbf{B}|$ ($\mu$T)" if args.stat=="mag" else f"{args.stat} ($\\mu$T)"
    cbar.set_label(label_unit)
    try:
        # 10/50/90% ticks (clipped to vmin/vmax)
        qs = np.quantile(vline, [0.1, 0.5, 0.9])
        ticks = np.clip(qs, vmin, vmax)
        cbar.set_ticks(ticks)
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print("✅ Saved:", args.out.resolve())

if __name__ == "__main__":
    main()

# python geomag_trackmap——plus.py --floor-dir .\site1\B1 --source cal --stat mag --q 10,90 --lw 6 --alpha 0.95 --smooth 5 --break-dist-px 18 --out b1_track.png