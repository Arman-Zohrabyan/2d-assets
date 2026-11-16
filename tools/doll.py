#!/usr/bin/env python3
"""Paper-doll pipeline: generated pixel-art strips -> aligned layers on one fixed frame grid.

Usage: python3 tools/doll.py source/doll/doll.json

Inputs (all side view, facing right):
  * base strips of a bald mannequin (one per animation; attack strips include their weapon);
  * "kit" strips: the same animation redrawn with one or two pieces of equipment, each piece in its
    own colour family, used only to cut those pieces out;
  * head sheets (heads and headgear), each scaled by its own bald head so every head has one size;
  * a gear sheet (swords and shields drawn upright) whose sprites are placed in the hands.

Every output layer is a strip of `frame` sized cells that lines up pixel for pixel with the body:
  body (head removed, weapon removed) <- base strip
  armour layers                      <- kit strip minus base, after registering each frame to the body
  head / headgear layers             <- rigid sprites, placed where the mannequin head is in each frame
  weapons in idle / walk / run       <- gear sprites at the annotated near hand (`hands` in the config)
  weapons in attacks                 <- cut out of the attack strip, or another sprite matched to its pose
  shields                            <- gear sprites held in guard before the chest; masks/near_arm/<anim>.png
                                        is the sword arm, drawn again over the shield; masks/far_arm/<anim>.png
                                        is the far arm, cut away under a shield
  pants, steel greaves               <- the body's own legs recoloured (`derived`), so they fit every frame
Draw order is the order of `slots` in the config (see "compose" in the written doll.json).
"""
from __future__ import annotations

import colorsys
import heapq
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA = 128
ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- basics

def load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def hsv_of(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees, saturation and value (0..1) for an (..., 3) uint8 array."""
    c = rgb.astype(float) / 255.0
    mx, mn = c.max(-1), c.min(-1)
    d = mx - mn
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60
    h = np.where(d == 0, 0, h)
    sat = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return h, sat, mx


def hsv_rule(rgb: np.ndarray, rule: dict) -> np.ndarray:
    h, s, v = hsv_of(rgb)
    lo, hi = rule["h"]
    hue = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
    ok = hue & (s >= rule["s"][0]) & (s <= rule["s"][1]) & (v >= rule["v"][0]) & (v <= rule["v"][1])
    if "grey_s" in rule:
        ok |= (s < rule["grey_s"]) & (v >= rule["v"][0]) & (v <= rule["v"][1])
    if "light" in rule:  # [s_max, v_min]: pale highlights of the same hue
        ok |= hue & (s <= rule["light"][0]) & (v >= rule["light"][1])
    return ok


def label(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """4-connected components. Returns labels (-1 = background) and component sizes."""
    lab = np.full(mask.shape, -1, dtype=np.int32)
    sizes: list[int] = []
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0] >= 0:
            continue
        n = len(sizes)
        lab[y0, x0] = n
        q, size = deque([(y0, x0)]), 0
        while q:
            y, x = q.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = n
                    q.append((ny, nx))
        sizes.append(size)
    return lab, sizes


def neighbours4(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1)
    return p[:-2, 1:-1].astype(np.int32) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def dilate(mask: np.ndarray) -> np.ndarray:
    return (neighbours4(mask) > 0) | mask


def dilate8(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1)
    out = mask.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def polygon_mask(poly: list, shape: tuple[int, int]) -> np.ndarray:
    """Pixels whose centre lies inside a polygon given in pixel-edge coordinates."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    px, py = xx + 0.5, yy + 0.5
    inside = np.zeros(shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if y1 == y2:
            continue
        cross = (y1 > py) != (y2 > py)
        inside ^= cross & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
    return inside


def geodesic(seed_dist: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Shortest 8-connected path length inside `allowed`, starting from finite values of `seed_dist`."""
    dist = np.where(np.isfinite(seed_dist) & allowed, seed_dist, np.inf)
    pq = [(float(dist[y, x]), int(y), int(x)) for y, x in zip(*np.nonzero(np.isfinite(dist)))]
    heapq.heapify(pq)
    h, w = allowed.shape
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (dy or dx) and 0 <= ny < h and 0 <= nx < w and allowed[ny, nx]:
                    nd = d + (1.4142 if dy and dx else 1.0)
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(pq, (nd, ny, nx))
    return dist


# ----------------------------------------------------------------------------- palette

def build_palette(arrays: list[np.ndarray], max_colors: int, fixed: list[str]) -> np.ndarray:
    fixed_rgb = np.array([hex_rgb(h) for h in fixed], dtype=np.uint8).reshape(-1, 3)
    pixels = np.concatenate([a[a[..., 3] >= ALPHA][:, :3] for a in arrays])
    rng = np.random.default_rng(0)
    if len(pixels) > 600_000:
        pixels = pixels[rng.choice(len(pixels), 600_000, replace=False)]
    side = int(np.ceil(np.sqrt(len(pixels))))
    canvas = np.zeros((side * side, 3), dtype=np.uint8)
    canvas[: len(pixels)] = pixels
    canvas[len(pixels):] = pixels[0]
    q = Image.fromarray(canvas.reshape(side, side, 3), "RGB").quantize(
        colors=max_colors - len(fixed_rgb), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    used = sorted(set(np.asarray(q).ravel().tolist()))
    auto = np.array(q.getpalette()[: 3 * 256], dtype=np.uint8).reshape(-1, 3)[used]
    return np.concatenate([fixed_rgb, auto])


def to_labels(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    pal_img = Image.new("P", (1, 1))
    flat = palette.ravel().tolist()
    pal_img.putpalette(flat + flat[:3] * (256 - len(palette)))
    idx = np.asarray(Image.fromarray(arr[..., :3], "RGB").quantize(palette=pal_img, dither=Image.Dither.NONE),
                     dtype=np.int32)
    idx = np.where(idx >= len(palette), 0, idx)
    return np.where(arr[..., 3] >= ALPHA, idx, -1)


class Colors:
    """Classifies palette indices into named colour families from the config."""

    def __init__(self, palette: np.ndarray, cfg: dict):
        self.palette = palette
        self.hsv = np.array([colorsys.rgb_to_hsv(*(c / 255.0)) for c in palette.astype(float)])
        self.rules = cfg["colors"]
        self.skin = hsv_rule(palette, cfg["skin_hsv"])
        self.linen = hsv_rule(palette, cfg["linen_hsv"]) & ~self.skin
        self.dark = self.hsv[:, 2] < 0.27
        self.outline = int(np.argmin(((palette.astype(int) - np.array(cfg["outline_rgb"])) ** 2).sum(1)))

    def _near(self, swatches: list[str], dist: float) -> np.ndarray:
        sw = np.array([hex_rgb(h) for h in swatches])
        d = np.sqrt(((self.palette[:, None, :].astype(int) - sw[None]) ** 2).sum(-1)).min(1)
        return d < dist

    def family(self, name: str) -> np.ndarray:
        """Boolean per palette index."""
        if name == "any":
            return ~(self.skin | self.linen | self.dark)
        return hsv_rule(self.palette, self.rules[name]) & ~self.skin & ~self.linen

    def value(self, img: np.ndarray) -> np.ndarray:
        return np.where(img >= 0, self.hsv[np.maximum(img, 0), 2], 0.0)

    def of(self, img: np.ndarray, table: np.ndarray) -> np.ndarray:
        return np.where(img >= 0, table[np.maximum(img, 0)], False)


# ----------------------------------------------------------------------------- frames in a raw strip

def split_runs(profile: np.ndarray, count: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, c in enumerate(profile):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(profile)])
    mass = [int(profile[a:b].sum()) for a, b in runs]
    runs = [r for r, m in zip(runs, mass) if m >= max(mass) * 0.02]
    while len(runs) > count:  # merge the closest neighbours
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        i = int(np.argmin(gaps))
        runs[i] = [runs[i][0], runs[i + 1][1]]
        del runs[i + 1]
    while len(runs) < count:  # touching figures: cut the widest run at its thinnest point
        i = int(np.argmax([b - a for a, b in runs]))
        a, b = runs[i]
        lo, hi = a + (b - a) // 4, b - (b - a) // 4
        cut = lo + int(np.argmin(profile[lo:hi]))
        runs[i:i + 1] = [[a, cut], [cut, b]]
    return [tuple(r) for r in runs]


def find_frames(mask: np.ndarray, count: int, rows: int) -> list[dict]:
    bands = [(0, mask.shape[0])] if rows == 1 else split_runs(mask.sum(1), rows)
    frames = []
    for band, (y0, y1) in enumerate(bands):
        sub = mask[y0:y1]
        for x0, x1 in split_runs(sub.sum(0), count // rows):
            ys = np.flatnonzero(sub[:, x0:x1].any(1))
            frames.append({"x0": x0, "x1": x1, "y0": y0 + int(ys[0]), "y1": y0 + int(ys[-1]) + 1, "band": band})
    return frames


def head_blob(skin: np.ndarray, f: dict) -> dict | None:
    """Largest skin component in the upper half of a frame (head + neck) in source pixels."""
    sub = skin[f["y0"]:f["y1"], f["x0"]:f["x1"]]
    lab, sizes = label(sub)
    best = None
    for i, size in enumerate(sizes):
        ys, xs = np.nonzero(lab == i)
        if ys.mean() > (f["y1"] - f["y0"]) * 0.5:
            continue
        if best is None or size > best["size"]:
            best = {"size": size, "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1),
                    "cx": f["x0"] + (xs.min() + xs.max() + 1) / 2, "y0": f["y0"] + int(ys.min())}
    return best


# ----------------------------------------------------------------------------- downsampling

def downsample(labels, f, scale, ax, bottom, cfg, n_colors, dark, thin=None):
    """Coverage decides opacity, weighted mode decides colour. (ax, bottom) land on (anchor_x, ground)."""
    w, h = cfg["frame"]
    ground, anchor_x = cfg["ground_y"], cfg["anchor_x"]
    out = np.full((h, w), -1, dtype=np.int32)
    weights = np.where(dark, 1.4, 1.0)
    if thin is not None:  # thin metal / wood (blades, bows): keep them light and unbroken
        weights = np.where(thin, 2.2, weights)
    for ty in range(h):
        sy0 = bottom - (ground + 1 - ty) / scale
        sy1 = bottom - (ground - ty) / scale
        ry0, ry1 = int(round(sy0)), max(int(round(sy1)), int(round(sy0)) + 1)
        ry0, ry1 = max(ry0, f["y0"]), min(ry1, f["y1"])
        if ry1 <= ry0:
            continue
        for tx in range(w):
            sx0 = ax + (tx - anchor_x) / scale
            sx1 = ax + (tx + 1 - anchor_x) / scale
            rx0, rx1 = int(round(sx0)), max(int(round(sx1)), int(round(sx0)) + 1)
            rx0, rx1 = max(rx0, f["x0"]), min(rx1, f["x1"])
            if rx1 <= rx0:
                continue
            block = labels[ry0:ry1, rx0:rx1].ravel()
            opaque = block[block >= 0]
            if len(opaque) == 0:
                continue
            if len(opaque) * 2 < len(block):
                if thin is None or thin[opaque].sum() * 4 < len(block):
                    continue
            out[ty, tx] = int(np.argmax(np.bincount(opaque, minlength=n_colors) * weights))
    return out


def close_outline(img: np.ndarray, outline: int) -> np.ndarray:
    op = img >= 0
    img = np.where(op & (neighbours4(op) == 0), -1, img)  # specks
    op = img >= 0
    holes = (~op) & (neighbours4(op) == 4)
    if holes.any():
        pad = np.pad(img, 1, constant_values=-1)
        img = np.where(holes, np.maximum(pad[1:-1, :-2], pad[:-2, 1:-1]), img)
    op = img >= 0
    p = np.pad(op, 1)
    edge = op & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return np.where(edge, outline, img)


def fill_holes(img: np.ndarray, passes: int = 3) -> np.ndarray:
    """Fill transparent pixels enclosed by at least 3 opaque neighbours (after cutting a weapon out)
    with the most common colour around them."""
    img = img.copy()
    for _ in range(passes):
        op = img >= 0
        holes = (~op) & (neighbours4(op) >= 3)
        if not holes.any():
            break
        pad = np.pad(img, 1, constant_values=-1)
        cand = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        for y, x in zip(*np.nonzero(holes)):
            vals = [int(v) for v in cand[:, y, x] if v >= 0]
            img[y, x] = max(set(vals), key=vals.count)
    return img


def fill_small_holes(img: np.ndarray, max_size: int = 4) -> np.ndarray:
    """Fill enclosed transparent pockets of up to `max_size` px (slits the layer below would show through)
    from their neighbours, working inwards."""
    img = img.copy()
    if not (img >= 0).any():
        return img
    lab, sizes = label(img < 0)
    for j, sz in enumerate(sizes):
        hole = lab == j
        if sz <= max_size and not (hole[0].any() or hole[-1].any() or hole[:, 0].any() or hole[:, -1].any()):
            img[hole] = -2
    while (img == -2).any():
        holes = img == -2
        pad = np.pad(img, 1, constant_values=-1)
        cand = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        best = np.where(cand >= 0, cand, -1).max(0)
        fillable = holes & (best >= 0)
        if not fillable.any():
            img[holes] = -1
            break
        img[fillable] = best[fillable]
    return img


def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.full_like(img, -1)
    h, w = img.shape
    ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
    yd, xd = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = img[yd, xd]
    return out


def register(moving: np.ndarray, fixed: np.ndarray, weight: np.ndarray | None = None, r: int = 4):
    """Integer shift (dx, dy) of `moving` that best matches `fixed` (same palette index)."""
    best, best_score = (0, 0), -1.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m = shift(moving, dx, dy)
            same = (m >= 0) & (m == fixed)
            score = float(same.sum() if weight is None else (same * weight).sum()) - 0.05 * (abs(dx) + abs(dy))
            if score > best_score:
                best, best_score = (dx, dy), score
    return best


# ----------------------------------------------------------------------------- strips -> frames

class Strip:
    """One raw strip cut into frames and normalised onto the output grid."""

    def __init__(self, path: Path, count: int, rows: int):
        self.arr = load(path)
        self.frames = find_frames(self.arr[..., 3] >= ALPHA, count, rows)

    def measure(self, skin_rule: dict):
        skin = hsv_rule(self.arr[..., :3], skin_rule) & (self.arr[..., 3] >= ALPHA)
        self.heads = [head_blob(skin, f) for f in self.frames]
        widths = [h["w"] for h in self.heads if h]
        self.head_w = float(np.median(widths)) if widths else None
        return self

    def anchors(self, mode: str, scale: float):
        """Source (ax, bottom) per frame. head: head centre fixed; feet: planted back foot fixed."""
        grounds = {}
        for f in self.frames:
            grounds[f["band"]] = max(grounds.get(f["band"], 0), f["y1"])
        out = []
        for i, f in enumerate(self.frames):
            lift = grounds[f["band"]] - f["y1"]
            bottom = f["y1"] + (lift if lift * scale >= 1.5 else 0)
            out.append([self._head_x(i, f), bottom])
        if mode == "feet":
            feet = [self._back_foot(f) for f in self.frames]
            offset = out[0][0] - feet[0]
            for i in range(len(out)):
                out[i][0] = feet[i] + offset
        return out

    def _head_x(self, i, f):
        h = self.heads[i]
        return h["cx"] if h else (f["x0"] + f["x1"]) / 2

    def _back_foot(self, f):
        a = self.arr[..., 3][f["y1"] - max(3, (f["y1"] - f["y0"]) // 25): f["y1"], f["x0"]:f["x1"]] >= ALPHA
        xs = np.flatnonzero(a.any(0))
        return f["x0"] + (xs[0] if len(xs) else 0)

    def render(self, labels, scale, anchors, cfg, n, dark, thin=None):
        return [downsample(labels, f, scale, ax, bottom, cfg, n, dark, thin)
                for f, (ax, bottom) in zip(self.frames, anchors)]


# ----------------------------------------------------------------------------- main build

def save_strip(frames: list[np.ndarray], palette: np.ndarray, path: Path, max_colors: int = 47) -> None:
    """Write frames side by side. At most `max_colors` opaque colours (+ transparent = 48 in a PNG palette)."""
    h, w = frames[0].shape
    rgba = np.zeros((h, w * len(frames), 4), dtype=np.uint8)
    idx, counts = np.unique(np.concatenate([f[f >= 0] for f in frames]), return_counts=True)
    if len(idx) > max_colors:  # merge the rarest colours into their nearest neighbour in this strip
        remap = {}
        keep = list(idx[np.argsort(-counts)][:max_colors])
        pal = palette.astype(int)
        for i in idx:
            if i not in keep:
                remap[int(i)] = int(keep[int(np.argmin(((pal[keep] - pal[i]) ** 2).sum(1)))])
        frames = [np.where(np.isin(f, list(remap)), np.vectorize(lambda v: remap.get(int(v), v))(f), f)
                  for f in frames]
    for i, f in enumerate(frames):
        op = f >= 0
        cell = rgba[:, i * w:(i + 1) * w]
        cell[op, :3] = palette[f[op]]
        cell[op, 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)


def paste(canvas: np.ndarray, sprite: np.ndarray, ox: int, oy: int) -> None:
    h, w = sprite.shape
    H, W = canvas.shape
    for y in range(h):
        cy = y + oy
        if not 0 <= cy < H:
            continue
        for x in range(w):
            cx = x + ox
            if 0 <= cx < W and sprite[y, x] >= 0:
                canvas[cy, cx] = sprite[y, x]


def crop(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    ys, xs = np.nonzero(img >= 0)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(xs.min()), int(ys.min())


# ----------------------------------------------------------------------------- held items (weapons, shields)

BROAD_SKIN = {"h": [8, 36], "s": [0.3, 0.7], "v": [0.5, 1]}  # hands including their shading
HIDE_BACK = 8  # columns behind a closed helmet whose hair it hides


def sheet_cells(arr: np.ndarray, per_row: list[int]) -> list[dict]:
    """Bounding boxes of the items on a sheet laid out in rows, `per_row[i]` items in row i."""
    a = arr[..., 3] >= ALPHA
    cells = []
    for (y0, y1), k in zip(split_runs(a.sum(1), len(per_row)), per_row):
        sub = a[y0:y1]
        for x0, x1 in split_runs(sub.sum(0), k):
            ys = np.flatnonzero(sub[:, x0:x1].any(1))
            cells.append({"x0": x0, "x1": x1, "y0": y0 + int(ys[0]), "y1": y0 + int(ys[-1]) + 1})
    return cells


def upright_points(rgba: np.ndarray) -> dict:
    """Crossguard, grip centre, tip and centre of an upright sword (blade up) in a high-res sprite."""
    op = rgba[..., 3] >= ALPHA
    rows = np.flatnonzero(op.any(1))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    widths = op.sum(1)
    lo = top + int((bottom - top) * 0.4)  # the crossguard is in the lower part
    peak = lo + int(np.argmax(widths[lo:bottom]))
    wide = np.flatnonzero(widths[lo:bottom] >= widths[peak] * 0.9) + lo
    wide = wide[np.abs(wide - peak) <= (bottom - top) * 0.05]  # one guard, several rows thick
    guard_y = float(wide.mean())

    def mid_x(y: float) -> float:
        xs = np.flatnonzero(op[min(int(round(y)), op.shape[0] - 1)])
        return float(xs.mean()) if len(xs) else op.shape[1] / 2

    grip_y = guard_y + 0.45 * (bottom - guard_y)
    return {"guard": (mid_x(guard_y), guard_y), "grip": (mid_x(grip_y), grip_y), "tip": (mid_x(top), float(top)),
            "centre": (op.shape[1] / 2, (top + bottom) / 2)}


def rotate_sprite(rgba: np.ndarray, pivot: tuple[float, float], angle: float):
    """Rotate a sprite that points UP so it points along `angle` (degrees; 0 = right, 90 = down).
    Returns the rotated image and where the pivot ended up in it."""
    h, w = rgba.shape[:2]
    r = int(np.hypot(h, w)) + 4
    canvas = np.zeros((2 * r, 2 * r, 4), dtype=np.uint8)
    ox, oy = r - int(round(pivot[0])), r - int(round(pivot[1]))
    canvas[oy:oy + h, ox:ox + w] = rgba
    im = Image.fromarray(canvas, "RGBA").rotate(-(angle + 90), resample=Image.Resampling.NEAREST, center=(r, r))
    return np.asarray(im), (float(r), float(r))


def resample(labels, scale, src_pivot, dst_pivot, shape, n, dark, thin):
    """Scale a label image by `scale` so that `src_pivot` lands on `dst_pivot` (continuous coordinates).
    Same rules as `downsample`: coverage decides opacity, a weighted mode decides colour."""
    H, W = shape
    out = np.full((H, W), -1, dtype=np.int32)
    ys, xs = np.nonzero(labels >= 0)
    if not len(ys):
        return out
    tx0 = max(0, int(np.floor(dst_pivot[0] + (xs.min() - src_pivot[0]) * scale)) - 1)
    tx1 = min(W, int(np.ceil(dst_pivot[0] + (xs.max() + 1 - src_pivot[0]) * scale)) + 1)
    ty0 = max(0, int(np.floor(dst_pivot[1] + (ys.min() - src_pivot[1]) * scale)) - 1)
    ty1 = min(H, int(np.ceil(dst_pivot[1] + (ys.max() + 1 - src_pivot[1]) * scale)) + 1)
    weights = np.where(thin, 2.2, np.where(dark, 1.4, 1.0))
    sh, sw = labels.shape
    for ty in range(ty0, ty1):
        sy0 = src_pivot[1] + (ty - dst_pivot[1]) / scale
        ry0 = int(round(sy0))
        ry1 = max(int(round(sy0 + 1 / scale)), ry0 + 1)
        ry0, ry1 = max(ry0, 0), min(ry1, sh)
        if ry1 <= ry0:
            continue
        for tx in range(tx0, tx1):
            sx0 = src_pivot[0] + (tx - dst_pivot[0]) / scale
            rx0 = int(round(sx0))
            rx1 = max(int(round(sx0 + 1 / scale)), rx0 + 1)
            rx0, rx1 = max(rx0, 0), min(rx1, sw)
            if rx1 <= rx0:
                continue
            block = labels[ry0:ry1, rx0:rx1].ravel()
            opaque = block[block >= 0]
            if not len(opaque):
                continue
            if len(opaque) * 2 < len(block) and thin[opaque].sum() * 4 < len(block):
                continue
            out[ty, tx] = int(np.argmax(np.bincount(opaque, minlength=n) * weights))
    return out


def sprite_mask(spr, pivot, angle, scale):
    """Binary native-scale mask of a rotated sheet sprite and where its pivot lands in it."""
    rot, rp = rotate_sprite(spr, pivot, angle)
    a = Image.fromarray(((rot[..., 3] >= ALPHA) * 255).astype(np.uint8))
    side = max(1, int(round(a.width * scale)))
    small = np.asarray(a.resize((side, side), Image.Resampling.BOX)) >= 100
    k = side / a.width
    return small, (rp[0] * k, rp[1] * k)


def match_pose(ref: np.ndarray, spr: np.ndarray, pivot, scale, skin=None, grip_off=0.0):
    """Angle and pivot position (continuous frame coordinates) at which the sheet sprite best covers the
    weapon pixels of `ref`: coarse search over angles and shifts, then a fine search on the angle.
    With `skin` (the body's hands) a pose whose grip, `grip_off` px behind the pivot, lies in a hand wins ties."""
    rm = ref >= 0
    if rm.sum() < 6:
        return None
    H, W = rm.shape
    ys, xs = np.nonzero(rm)
    rc = (xs.mean(), ys.mean())
    near_skin = dilate(dilate(skin)) if skin is not None else None

    def best_shift(ang):
        m, pv = sprite_mask(spr, pivot, ang, scale)
        my, mx = np.nonzero(m)
        if not len(my):
            return -1.0, None
        cy, cx = int(round(my.mean())), int(round(mx.mean()))
        back = -np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * grip_off
        best = (-1.0, None)
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                ty = my - cy + int(round(rc[1])) + dy
                tx = mx - cx + int(round(rc[0])) + dx
                ok = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
                hit = int(rm[ty[ok], tx[ok]].sum())
                score = hit / (len(my) + rm.sum() - hit)
                pos = (int(round(rc[0])) + dx - cx + pv[0], int(round(rc[1])) + dy - cy + pv[1])
                if near_skin is not None:
                    gx, gy = int(pos[0] + back[0]), int(pos[1] + back[1])
                    if 0 <= gx < W and 0 <= gy < H and near_skin[gy, gx]:
                        score += 0.08
                if score > best[0]:
                    best = (score, pos)
        return best

    scored = [(best_shift(a), a) for a in range(-180, 180, 6)]
    (iou, pos), ang = max(scored, key=lambda x: x[0][0])
    for a in range(ang - 4, ang + 5):
        (i2, p2) = best_shift(a)
        if i2 > iou:
            iou, pos, ang = i2, p2, a
    return pos, float(ang), iou


def seg_dist(xx, yy, p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t = np.clip(((xx - p[0]) * d[0] + (yy - p[1]) * d[1]) / max(float((d ** 2).sum()), 1e-6), 0, 1)
    return np.hypot(xx - (p[0] + t * d[0]), yy - (p[1] + t * d[1]))


def limb_path(limbs: np.ndarray, tips) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path (y, x) through the limb pixels from the pixel nearest one tip to the other."""
    ys, xs = np.nonzero(limbs)
    if len(ys) < 3 or tips is None:
        return None
    ends = [int(np.argmin((xs - tx) ** 2 + (ys - ty) ** 2)) for tx, ty in tips]
    start, goal = (int(ys[ends[0]]), int(xs[ends[0]])), (int(ys[ends[1]]), int(xs[ends[1]]))
    # through limb pixels, but able to cross a gap (the hand over the grip, the head over a carried bow)
    h, w = limbs.shape
    best = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, (y, x) = heapq.heappop(pq)
        if (y, x) == goal:
            break
        if d > best[(y, x)]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n_ = (y + dy, x + dx)
                if (dy or dx) and 0 <= n_[0] < h and 0 <= n_[1] < w:
                    nd = d + (1.4142 if dy and dx else 1.0) * (1.0 if limbs[n_] else 8.0)
                    if nd < best.get(n_, np.inf):
                        best[n_], prev[n_] = nd, (y, x)
                        heapq.heappush(pq, (nd, n_))
    if goal not in prev:
        return None
    path, p = [], goal
    while p is not None:
        path.append(p)
        p = prev[p]
    return path[::-1]


def lacquer_bow(palette, w, frames_by_anim, tips_by, string_i, col):
    """Another bow on the same animated limbs (so grip, flex and string stay exact): the limbs in red lacquer,
    gold tips that curl forward (a recurve), a gold grip wrap. Arrow and string keep their colours."""
    lq = w["lacquer"]
    r0 = len(palette)
    ramp = [hex_rgb(h) for h in lq["ramp"]]  # deep, main, light, shine
    gold = [hex_rgb(h) for h in lq["gold"]]  # dark, main, bright
    palette = np.concatenate([palette, np.array(ramp + gold, dtype=np.uint8)])
    red = lambda k: r0 + k
    gld = lambda k: r0 + len(ramp) + k
    out = {}
    for aid, frames in frames_by_anim.items():
        tips_a = tips_by.get((w["recolor_of"], aid))
        res = []
        for fi, f in enumerate(frames):
            g = f.copy()
            limbs = (f >= 0) & (f != string_i)
            path = limb_path(limbs, tips_a[fi] if tips_a else None)
            if not path or len(path) < 6:
                res.append(g)
                continue
            pm = np.zeros_like(limbs)
            for p in path:
                pm[p] = True
            limb = limbs & dilate8(pm)
            v = col.value(np.where(f < len(col.hsv), f, -1))
            g[limb] = np.where(v[limb] < 0.3, red(1), np.where(v[limb] < 0.66, red(2), red(3)))
            # the back edge of a 2-px limb in the deep tone, so the lacquer has a little depth
            g[limb & ~pm & (v < 0.3)] = red(0)
            n_p = len(path)
            paint = lambda p, c: g.__setitem__(p, c) if f[p] >= 0 else None  # only where the bow shows
            for k in (0, 1, n_p - 2, n_p - 1):  # gold nocks
                paint(path[k], gld(1))
            mid = n_p // 2
            for k in (mid - 1, mid, mid + 1):  # grip wrap (under the hand where the hand holds it)
                paint(path[k], gld(1))
            paint(path[mid - 2], red(0))
            paint(path[mid + 2], red(0))
            # recurve: each tip curls forward, away from the string (chord -> grip direction)
            (y1, x1), (y2, x2), (ym, xm) = path[0], path[-1], path[mid]
            fwd = np.array([xm - (x1 + x2) / 2, ym - (y1 + y2) / 2], dtype=float)
            fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 0.5 else np.array([1.0, 0.0])
            for tip, back in ((path[0], path[min(3, n_p - 1)]), (path[-1], path[max(n_p - 4, 0)])):
                d = np.array([tip[1] - back[1], tip[0] - back[0]], dtype=float)
                d = d / max(np.linalg.norm(d), 1e-6)
                s1 = np.rint(d * 0.75 + fwd * 0.75).astype(int)
                p1 = (tip[0] + s1[1], tip[1] + s1[0])
                p2 = (p1[0] + int(np.rint(fwd[1])), p1[1] + int(np.rint(fwd[0])))
                for p, c in ((p1, gld(2)), (p2, gld(0))):
                    if 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1] and g[p] < 0:
                        g[p] = c
            res.append(g)
        out[aid] = res
    return palette, out


# ----------------------------------------------------------------------------- main build

def main(config_path: str) -> None:
    cfg = json.loads((ROOT / config_path).read_text())
    out_dir = ROOT / cfg["out_dir"]
    fw, fh = cfg["frame"]
    ground = cfg["ground_y"]
    anims = [a for a in cfg["animations"] if (ROOT / a["src"]).exists()]
    for a in cfg["animations"]:
        if a not in anims:
            print(f"skip animation {a['id']}: {a['src']} missing")
    anim_by_id = {a["id"]: a for a in anims}
    skin_rgb = cfg["skin_hsv"]
    weapons = cfg["weapons"]

    # ---- load everything that exists
    base = {a["id"]: Strip(ROOT / a["src"], a["frames"], a.get("rows", 1)).measure(skin_rgb) for a in anims}
    kit_strips: dict[tuple[int, str], Strip] = {}
    for k, kit in enumerate(cfg["kits"]):
        for a in anims:
            if a["id"] not in kit.get("anims", [x["id"] for x in anims]):
                continue
            p = ROOT / kit["src"].format(anim=a["id"])
            if p.exists():
                kit_strips[(k, a["id"])] = Strip(p, a["frames"], a.get("rows", 1)).measure(skin_rgb)
            else:
                print(f"missing kit strip: {p.relative_to(ROOT)}")
    head_sheets = []
    for hc in (cfg["heads"] if isinstance(cfg["heads"], list) else [cfg["heads"]]):
        p = ROOT / hc["src"]
        if p.exists():
            head_sheets.append((hc, Strip(p, len(hc["cells"]), 1).measure(skin_rgb)))
        else:
            print(f"missing head sheet: {hc['src']}")
    gear = cfg.get("gear")
    gear_arr = load(ROOT / gear["src"]) if gear and (ROOT / gear["src"]).exists() else None
    # more sheets (e.g. polearms): every sprite gets its own scale from `length` (its height in output pixels)
    extra_gear = [(g, load(ROOT / g["src"])) for g in cfg.get("gear_sheets", []) if (ROOT / g["src"]).exists()]

    # palette: body/heads/weapons first, then a small ramp per equipment piece taken from its own pixels,
    # so rare dark hues (a green cape, a purple skirt) are not merged into the outline colour
    arrays = [s.arr for s in base.values()] + [st.arr for _, st in head_sheets]
    palette = build_palette(arrays, cfg["max_colors"], cfg["fixed_colors"])
    item_ramps = []
    ramp_of: dict[tuple[int, int], tuple[int, int]] = {}  # (kit, item) -> palette index range of its ramp
    base_n = len(palette)
    for k, kit in enumerate(cfg["kits"]):
        strips_k = [st for (kk, _), st in kit_strips.items() if kk == k]
        for ii, item in enumerate(kit["items"]):
            if item["color"] == "any" or not strips_k:
                continue
            rule = cfg["colors"][item["color"]]
            px = np.concatenate([st.arr[hsv_rule(st.arr[..., :3], rule) & (st.arr[..., 3] >= ALPHA)][:, :3] for st in strips_k])
            if len(px) < 50:
                continue
            ramp = build_palette([np.concatenate([px, np.full((len(px), 1), 255, np.uint8)], 1)[None]],
                                 cfg.get("item_colors", 6), [])
            start = base_n + sum(len(r) for r in item_ramps)
            ramp_of[(k, ii)] = (start, start + len(ramp))
            item_ramps.append(ramp)
    if gear_arr is not None:
        gp = gear_arr[gear_arr[..., 3] >= ALPHA][:, :3]
        item_ramps.append(build_palette([np.concatenate([gp, np.full((len(gp), 1), 255, np.uint8)], 1)[None]],
                                        gear.get("colors", 16), []))
    for g, arr in extra_gear:
        gp = arr[arr[..., 3] >= ALPHA][:, :3]
        item_ramps.append(build_palette([np.concatenate([gp, np.full((len(gp), 1), 255, np.uint8)], 1)[None]],
                                        g.get("colors", 12), []))
    if item_ramps:
        palette = np.concatenate([palette] + item_ramps)
    col = Colors(palette, cfg)
    n = len(palette)
    thin = col.family("steel") | col.family("gold") | col.family("wood")
    broad_skin = hsv_rule(palette, BROAD_SKIN)
    print(f"palette {n} colours; skin idx {np.flatnonzero(col.skin).tolist()} linen idx {np.flatnonzero(col.linen).tolist()}")

    # ---- body frames
    idle = base[anims[0]["id"]]
    ref_scale = cfg["body_height"] / float(np.median([f["y1"] - f["y0"] for f in idle.frames]))
    head_target = ref_scale * idle.head_w
    body: dict[str, list[np.ndarray]] = {}
    scales: dict[str, float] = {}

    def strip_scale(st: Strip, a: dict) -> float:
        """Head width is the ruler, unless the animation names a body height for one of its frames
        (GPT sometimes draws a figure with a bigger head; the rigid head sprites must not shrink the body)."""
        if "height" in a:
            f = st.frames[a.get("height_frame", 0)]
            return a["height"] / (f["y1"] - f["y0"])
        return head_target / st.head_w

    for a in anims:
        s = base[a["id"]]
        scales[a["id"]] = strip_scale(s, a)
        frames = s.render(to_labels(s.arr, palette), scales[a["id"]], s.anchors(a["anchor"], scales[a["id"]]),
                          cfg, n, col.dark, thin)
        body[a["id"]] = [close_outline(f, col.outline) for f in frames]
        print(f"body {a['id']:<16} scale {scales[a['id']]:.4f} heights "
              f"{[int((f >= 0).any(1).sum()) for f in body[a['id']]]}")

    # ---- head template from idle frame 0, then find the head in every frame
    f0 = body[anims[0]["id"]][0]
    skin0 = col.of(f0, col.skin)
    lab, sizes = label(skin0)
    comp = lab == int(np.argmax(sizes))
    tmask_full = dilate(dilate(comp) | comp) & ~col.of(f0, col.linen) & (f0 >= 0)  # head + highlights + outline ring
    ys, xs = np.nonzero(tmask_full)
    ty0, tx0 = ys.min(), xs.min()
    tmask = tmask_full[ty0:ys.max() + 1, tx0:xs.max() + 1]
    tpatch = np.where(tmask, f0[ty0:ys.max() + 1, tx0:xs.max() + 1], -1)
    th, tw = tpatch.shape
    pal_f = palette.astype(float)
    head_pos: dict[str, list[tuple[int, int]]] = {}
    head_box: dict[str, list[np.ndarray]] = {}  # where the mannequin head was, per frame
    for a in anims:
        positions, boxes = [], []
        for img in body[a["id"]]:
            best, best_cost = (tx0, ty0), 1e18
            rows = np.flatnonzero((img >= 0).any(1))
            top = rows[0] if len(rows) else 0
            for py in range(max(0, top - 4), min(fh - th, top + int(cfg["body_height"] * 0.55))):
                for px in range(0, fw - tw):
                    win = img[py:py + th, px:px + tw]
                    m = tmask & (win >= 0)
                    if m.sum() < tmask.sum() * 0.7:
                        continue
                    diff = pal_f[win[m]] - pal_f[tpatch[m]]
                    cost = float((diff ** 2).sum()) / m.sum() + 900.0 * (1 - m.sum() / tmask.sum())
                    if cost < best_cost:
                        best, best_cost = (px, py), cost
            positions.append(best)
            px, py = best
            win = img[py:py + th, px:px + tw]
            kill = tmask & (win >= 0) & ~col.of(win, col.linen)
            win[kill] = -1  # cut the mannequin head (and neck) out of the body
            box = np.zeros((fh, fw), dtype=bool)
            box[py:py + th, px:px + tw] = tmask
            box = dilate(box)
            boxes.append(box)
            # leftovers of the cut: tiny islands inside the head area
            lab, sizes = label(img >= 0)
            if sizes:
                keep = int(np.argmax(sizes))
                for i, sz in enumerate(sizes):
                    if i != keep and sz < 8 and (box & (lab == i)).sum() == sz:
                        img[lab == i] = -1
        head_pos[a["id"]] = positions
        head_box[a["id"]] = boxes

    # ---- weapons out of attack bodies
    yy_g, xx_g = np.mgrid[0:fh, 0:fw]
    layers: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    weapon_tips: dict[tuple[str, str], list] = {}  # (bow, anim) -> per frame ((x, y) top tip, (x, y) bottom tip)
    string_idx: dict[str, int] = {}

    def put(slot, item, anim, frames):
        layers.setdefault((slot, item), {})[anim] = frames

    for a in anims:
        if "weapon" not in a:
            continue
        rule = weapons[a["weapon"]]
        seed = np.zeros(n, dtype=bool)
        for fam in rule["core"]:
            seed |= col.family(fam)
        grow = np.zeros(n, dtype=bool)
        for fam in rule.get("grow", []):
            grow |= col.family(fam)
        grow |= col.dark
        w_frames, tips = [], []
        for fi, img in enumerate(body[a["id"]]):
            rows = np.flatnonzero((img >= 0).any(1))
            floor = rows[-1] - 4 if len(rows) else fh
            yy = np.arange(fh)[:, None] < floor
            core = col.of(img, seed) & yy
            if rule.get("thin_light"):  # bow string / fletching: light 1-px lines near the bow
                op = img >= 0
                p = np.pad(op, 1)
                lone = (~p[1:-1, :-2] & ~p[1:-1, 2:]) | (~p[:-2, 1:-1] & ~p[2:, 1:-1]) | \
                       (~p[:-2, :-2] & ~p[2:, 2:]) | (~p[:-2, 2:] & ~p[2:, :-2])
                light = col.of(img, col.linen) | ((col.value(img) > 0.75) & ~col.of(img, col.skin))
                cy, cx = np.nonzero(core)
                near_bow = np.zeros_like(op)
                if len(cx):
                    near_bow[max(cy.min() - 2, 0):cy.max() + 3, max(cx.min() - 16, 0):cx.max() + 3] = True
                core |= op & lone & light & yy & near_bow
            wm = core.copy()
            body_like = col.of(img, col.skin | col.linen)
            grow_light = grow & ~col.dark
            for _ in range(3):  # take grips / guards that touch the weapon
                wm |= dilate(wm) & col.of(img, grow_light) & ~wm & yy
            # then one ring of outline: a dark pixel joins unless it borders more body than weapon
            add = dilate(wm) & col.of(img, col.dark) & ~wm & yy
            wm |= add & (neighbours4(wm) >= neighbours4(body_like & ~wm))
            if rule.get("thin_light"):
                # bow limbs are thin outline-coloured curves: follow them through the air, never along the body
                op = img >= 0
                for _ in range(16):
                    add = dilate(wm) & ~wm & op & col.of(img, grow | seed) & (neighbours4(body_like) == 0) & yy
                    if not add.any():
                        break
                    wm |= add
            # hands stay with the body (bows: wood is hand-coloured in the broad sense, use the strict rule)
            wm &= ~col.of(img, col.skin if rule.get("thin_light") else broad_skin)
            # a weapon part must contain weapon colour; outline chains along the body stay with the body
            lab, sizes = label(wm)
            keep = np.zeros_like(wm)
            for i, sz in enumerate(sizes):
                m = lab == i
                if int((m & core).sum()) >= min(2, sz):
                    keep |= m
            wm = keep
            nock = rule.get("string", {}).get("nock", {}).get(a["id"], {}).get(str(fi))
            if rule.get("thin_light"):
                # a bow is its two limbs (the front-most big parts, facing right) plus, while drawn, the arrow
                # from the drawing hand forward; GPT's string is redrawn below, brown leg shading is dropped
                lab2, sizes2 = label(wm)
                xr = [(int(np.nonzero(lab2 == i)[1].min()), int(np.nonzero(lab2 == i)[1].max())) for i in range(len(sizes2))]
                cand = sorted([i for i in range(len(sizes2)) if sizes2[i] >= 12] or list(range(len(sizes2))),
                              key=lambda i: -xr[i][1])
                limbs = cand[:1] + [i for i in cand[1:2] if xr[i][1] >= xr[cand[0]][0] - 2] if cand else []
                keep = np.isin(lab2, limbs)
                if nock and keep.any():
                    front = int(np.nonzero(keep)[1].max()) + 8
                    arrow_band = (np.abs(yy_g - nock[1]) <= 2) & (xx_g >= nock[0] - 1) & (xx_g <= front)
                    for i in range(len(sizes2)):
                        if i not in limbs and ((lab2 == i) & arrow_band).sum() * 2 >= sizes2[i]:
                            keep |= lab2 == i
                wm = keep & ~(col.of(img, col.linen) & ~np.isin(lab2, limbs))  # old string bits go
                ys2, xs2 = np.nonzero(np.isin(lab2, limbs) & wm)
                tips.append(((int(round(xs2[ys2 == ys2.min()].mean())), int(ys2.min())),
                             (int(round(xs2[ys2 == ys2.max()].mean())), int(ys2.max()))) if len(ys2) else None)
            else:
                # bits cut off by the weapon: weapon-coloured / outline bits next to it join it, body bits stay
                rest = (img >= 0) & ~wm
                lab, sizes = label(rest)
                near_w = dilate(dilate(wm))
                if sizes:
                    main_part = int(np.argmax(sizes))
                    for i, sz in enumerate(sizes):
                        m = lab == i
                        if i == main_part or sz >= 25 or not (m & near_w).any():
                            continue
                        if (m & body_like).sum() * 2 < sz:
                            wm |= m
            w_frames.append(np.where(wm, img, -1))
            took_outline = wm & col.of(img, col.dark)
            img[wm] = -1
            img[:] = fill_small_holes(fill_holes(img), 8)
            # where the weapon took the body's outline with it, the cloth edge gets an outline back
            op = img >= 0
            edge = op & (neighbours4(op) < 4) & dilate(took_outline) & col.of(img, col.linen)
            img[edge] = col.outline
        string = rule.get("string")
        if string:  # 1-px bow string, too thin to survive downsampling: tips -> nock (drawing hand) or straight
            idx = int(np.argmin(((palette.astype(int) - np.array(hex_rgb(string["color"]))) ** 2).sum(1)))
            string_idx[a["weapon"]] = idx
            limb_idx = None
            if rule.get("limb_outline"):  # the limbs' outline in dark wood, so the bow reads as a bow
                limb_idx = int(np.argmin(((palette.astype(int) - np.array(hex_rgb(rule["limb_outline"]))) ** 2).sum(1)))
            nocks = string.get("nock", {}).get(a["id"], {})
            for fi, wf in enumerate(w_frames):
                if tips[fi] is None:
                    continue
                if limb_idx is not None:
                    wf[col.of(wf, col.dark)] = limb_idx
                top, bot = tips[fi]
                pts = [top, tuple(int(round(v)) for v in nocks[str(fi)]), bot] if str(fi) in nocks else [top, bot]
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    steps = max(abs(x1 - x0), abs(y1 - y0))
                    for t in range(1, steps):
                        x, y = round(x0 + (x1 - x0) * t / steps), round(y0 + (y1 - y0) * t / steps)
                        if wf[y, x] < 0:
                            wf[y, x] = idx
        put("weapon", a["weapon"], a["id"], w_frames)
        if tips:
            weapon_tips[(a["weapon"], a["id"])] = tips

    # ---- hands: near-arm polyline + far hand per frame (annotated in the config, output pixel coordinates)
    hands = {aid: [{"shoulder": (v[0], v[1]), "elbow": (v[2], v[3]), "hand": (v[4], v[5]), "far": (v[6], v[7])}
                   for v in rows]
             for aid, rows in cfg.get("hands", {}).items() if aid in body}

    def fist(img, hand):
        """The near hand's skin pixels around an annotated hand point."""
        return col.of(img, broad_skin) & (np.hypot(xx_g - hand[0], yy_g - hand[1]) <= 2.6)

    for aid, hs in hands.items():  # annotations are by eye: snap each hand onto the skin blob it points at
        for img, h in zip(body[aid], hs):
            for key in ("hand", "far"):
                skin = col.of(img, broad_skin) & (np.hypot(xx_g - h[key][0], yy_g - h[key][1]) <= 3.5)
                lab, sizes = label(skin)
                if sizes:
                    i = int(np.argmax(sizes))
                    if sizes[i] >= 3:
                        ys, xs = np.nonzero(lab == i)
                        h[key] = (float(xs.mean()), float(ys.mean()))

    arm_masks: dict[str, list[np.ndarray]] = {}
    for aid, hs in hands.items():
        out = []
        for img, h in zip(body[aid], hs):
            d = np.minimum(seg_dist(xx_g, yy_g, h["shoulder"], h["elbow"]), seg_dist(xx_g, yy_g, h["elbow"], h["hand"]))
            out.append(((d <= cfg.get("arm_radius", 2.0)) & dilate(img >= 0)) | dilate(fist(img, h["hand"])))
        arm_masks[aid] = out

    # far arm where it shows outside the torso (annotated polygons): with a shield it is behind the shield, so
    # it is cut from the body and the arm-bound layers (masks/far_arm); the near arm is never cut
    far_cut: dict[str, list[np.ndarray]] = {}
    for aid, hs in hands.items():
        polys = cfg.get("far_arm_cut", {}).get(aid, {})
        out = []
        for fi in range(len(hs)):
            m = np.zeros((fh, fw), dtype=bool)
            for poly in polys.get(str(fi), []):
                m |= polygon_mask(poly, (fh, fw))
            out.append(m & ~arm_masks[aid][fi])
        far_cut[aid] = out

    def arrow_like(comp: np.ndarray) -> bool:
        """A copy of the arrow: at most 2 rows tall and at least 5 px long."""
        ys, xs = np.nonzero(comp)
        return ys.max() - ys.min() <= 1 and xs.max() - xs.min() >= 4

    def thin_part(comp: np.ndarray) -> bool:
        """A line-like piece (arrow shaft, blade): few pixels for its length."""
        ys, xs = np.nonzero(comp)
        length = np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        return len(ys) <= 2.5 * length

    # ---- kit items: register each kit frame on the body frame, keep what changed
    pal_i = palette.astype(int)
    # shoe colours: the only non-outline colours at the mannequin's feet that are not linen or skin
    shoe = np.zeros(n, dtype=bool)
    for aid in [x["id"] for x in anims if not x.get("weapon")]:
        for img in body[aid]:
            feet = img[ground - 3:ground + 1]
            for v in np.unique(feet[feet >= 0]):
                if not (col.dark[v] or col.linen[v] or col.skin[v] or broad_skin[v]):
                    shoe[v] = True
    print(f"shoe colours {np.flatnonzero(shoe).tolist()}")
    for (k, anim_id), s in sorted(kit_strips.items()):
        kit = cfg["kits"][k]
        a = anim_by_id[anim_id]
        scale = strip_scale(s, a)
        frames = s.render(to_labels(s.arr, palette), scale, s.anchors(a["anchor"], scale), cfg, n, col.dark, thin)
        ref_frames = [close_outline(f, col.outline) for f in base[anim_id].render(
            to_labels(base[anim_id].arr, palette), scales[anim_id],
            base[anim_id].anchors(a["anchor"], scales[anim_id]), cfg, n, col.dark, thin)]
        per_item = {i: [] for i in range(len(kit["items"]))}
        shifts = []
        bow_attack = bool(a.get("weapon")) and bool(weapons[a["weapon"]].get("thin_light"))
        w_fams = np.zeros(n, dtype=bool)  # the attack's weapon, redrawn by the kit, must not leak into armour
        if a.get("weapon"):
            for fam in weapons[a["weapon"]]["core"] + weapons[a["weapon"]].get("grow", []):
                w_fams |= col.family(fam)
        for fi, (img, ref) in enumerate(zip(frames, ref_frames)):
            # armour sits on the body: the weaponless body (dilated 2 px) bounds where it may appear,
            # so a sword the kit drew a little elsewhere cannot leak into a steel piece
            on_body = dilate(dilate(body[anim_id][fi] >= 0))
            body_like = col.of(ref, col.skin | col.linen)
            dx, dy = register(img, ref, weight=body_like.astype(float) + 0.2)
            img = shift(img, dx, dy)
            shifts.append((dx, dy))
            op = img >= 0
            changed = op & ((ref < 0) | (np.sqrt(((pal_i[np.maximum(img, 0)] - pal_i[np.maximum(ref, 0)]) ** 2).sum(-1)) > 40))
            claimed = np.zeros_like(op)
            masks = []
            inside = body[anim_id][fi] >= 0
            near_weapon = np.zeros_like(op)
            w_zone = np.zeros_like(op)
            if a.get("weapon"):  # weapon parts sticking out of the body
                w_layer = layers[("weapon", a["weapon"])][anim_id][fi] >= 0
                near_weapon = dilate(dilate(w_layer)) & col.of(img, w_fams) & ~inside
                w_zone = dilate(dilate(dilate(w_layer)))
            # never part of equipment: boots, the body's own hands, skin around the neck
            body_hands = np.zeros_like(op)
            if anim_id in hands:
                hh = hands[anim_id][fi]
                body_hands = (fist(body[anim_id][fi], hh["hand"]) | fist(body[anim_id][fi], hh["far"])) & inside
            forbid_other = body_hands | \
                (col.of(img, broad_skin) & dilate(dilate(head_box[anim_id][fi])))
            changed &= ~forbid_other
            for ii, item in enumerate(kit["items"]):  # each pixel goes to the first item whose colour matches
                own = np.zeros(n, dtype=bool)
                if (k, ii) in ramp_of:
                    own[ramp_of[(k, ii)][0]:ramp_of[(k, ii)][1]] = True
                m = changed & col.of(img, col.family(item["color"])) & ~claimed & ~near_weapon
                if not item.get("keeps_shoe_colours"):  # greaves sit on the boots: they use their band instead
                    m &= ~col.of(img, shoe & ~own)
                if item.get("on_body"):
                    # in attacks the kit's own weapon swings close to the body: allow only a 1-px rim there
                    m &= dilate(inside) if a.get("weapon") else on_body
                if item["slot"] in cfg.get("forearm_slots", ["bracers"]) and anim_id in hands:
                    h = hands[anim_id][fi]
                    fore = np.minimum(seg_dist(xx_g, yy_g, h["elbow"], h["hand"]),
                                      seg_dist(xx_g, yy_g, h["shoulder"], h["far"]))
                    m &= fore <= 4.0
                if "band" in item:  # fraction of body height above the ground: [lo, hi]
                    lo, hi = item["band"]
                    hgt = (ground - yy_g) / cfg["body_height"]
                    m &= (hgt >= lo) & (hgt <= hi)
                masks.append(m)
                claimed |= m
            # dark shading and outlines that changed: flood them into the touching piece
            body_like = col.of(img, col.skin | col.linen)
            for _ in range(40):
                free = changed & col.of(img, col.dark) & ~claimed
                counts = np.stack([neighbours4(m) for m in masks])
                best = counts.argmax(0)
                grab = free & (counts.max(0) > 0) & (counts.max(0) >= neighbours4(body_like))
                if not grab.any():
                    break
                for i in range(len(masks)):
                    add = grab & (best == i)
                    if kit["items"][i].get("on_body"):
                        add &= on_body
                    masks[i] = masks[i] | add
                    claimed |= add
            for i, m in enumerate(masks):
                lab, sizes = label(m)
                min_size = max(3, 0.08 * sum(sizes))
                for j, sz in enumerate(sizes):
                    comp = lab == j
                    # specks; armour mostly in the air (the kit's weapon drawn elsewhere); the kit's copy of the
                    # weapon itself (a part lying mostly along the attack's weapon, e.g. an arrow in a sleeve)
                    if sz < min_size or (kit["items"][i].get("on_body") and a.get("weapon")
                                         and (comp & inside).sum() * 2 < sz) \
                            or (bow_attack and (comp & w_zone).sum() * 2 > sz and arrow_like(comp)):
                        m[comp] = False
                item_img = fill_small_holes(np.where(m, img, -1))  # slits the body would show through
                per_item[i].append(item_img)
        for i, item in enumerate(kit["items"]):
            put(item["slot"], item["id"], anim_id, per_item[i])
        print(f"kit {k} {anim_id:<16} shifts {shifts} pixels {[int((f >= 0).sum()) for f in per_item[0]]}"
              f"{' / ' + str([int((f >= 0).sum()) for f in per_item[1]]) if len(per_item) > 1 else ''}")

    # ---- heads and headgear: every sheet is scaled by its own bald head, so all heads share one size
    def place_on(moving, fixed, init, scorer, r=4):
        """Offset (x, y) of `moving` in `fixed` coordinates: start at `init`, refine within r px."""
        pad = r + 2 + max(moving.shape)
        H, W = fixed.shape[0] + 2 * pad, fixed.shape[1] + 2 * pad
        fc = np.full((H, W), -1, dtype=np.int32)
        mc = np.full((H, W), -1, dtype=np.int32)
        paste(fc, fixed, pad, pad)
        paste(mc, moving, pad + init[0], pad + init[1])
        dx, dy = scorer(mc, fc, r)
        return init[0] + dx, init[1] + dy

    skin_w = lambda img: col.of(img, col.skin).astype(float) + 0.1
    by_skin = lambda m, f, r: register(m, f, weight=skin_w(f), r=r)
    face_rule = {"h": [8, 36], "s": [0.25, 0.75], "v": [0.45, 1]}

    def face_front(cell: np.ndarray, frac: float = 0.22) -> np.ndarray:
        """Per row, the front-most skin pixel of the face (-1 where the row has no face front)."""
        skin = hsv_rule(cell[..., :3], face_rule) & (cell[..., 3] >= ALPHA)
        prof = np.array([np.flatnonzero(r).max() if r.any() else -1 for r in skin], dtype=float)
        if (prof >= 0).any():
            prof[prof < prof.max() - cell.shape[1] * frac] = -1
        return prof

    def neck_front(cell: np.ndarray):
        skin = hsv_rule(cell[..., :3], face_rule) & (cell[..., 3] >= ALPHA)
        h = cell.shape[0]
        vals = [np.flatnonzero(skin[y]).max() for y in range(int(h * 0.9), h) if skin[y].any()]
        return float(np.median(vals)) if vals else None

    def cell_match(bald_cell: np.ndarray, cell: np.ndarray) -> tuple[float, int]:
        """(dx, dy) in sheet pixels: point (x, y) of the bald cell is (x - dx, y + dy) in `cell`.
        GPT keeps every neck stub on one line, so dy starts from the height difference; the same face
        is drawn on every head, so its front profile (forehead, nose, lips) gives dx."""
        dy0 = cell.shape[0] - bald_cell.shape[0]
        pb, pc = face_front(bald_cell), face_front(cell)
        best = None
        for dy in range(dy0 - 25, dy0 + 26):
            ys = np.arange(len(pb))
            yc = ys + dy
            ok = (yc >= 0) & (yc < len(pc))
            ys, yc = ys[ok], yc[ok]
            v = (pb[ys] >= 0) & (pc[yc] >= 0)
            if v.sum() < 30:
                continue
            d = pb[ys[v]] - pc[yc[v]]
            med = float(np.median(d))
            score = float(np.mean(np.abs(d - med))) - v.sum() * 0.002
            if best is None or score < best[0]:
                best = (score, dy, med)
        if best is not None and best[0] < 3:
            return best[2], best[1], True
        nb, nc = neck_front(bald_cell), neck_front(cell)  # closed helmet: only the neck shows
        return (nb - nc if nb is not None and nc is not None else 0.0), dy0, False

    def hide_mask(spr: np.ndarray, above: int = 6, closed: bool = False) -> np.ndarray:
        """What a headgear hides of the head under it: hair poking out ABOVE it. In each of its columns
        (and one either side) the `above` rows over its top pixel; nothing under the headgear is cut,
        so it can never open a see-through gap. Returned with `above` extra rows on top."""
        op = spr >= 0
        h, w = op.shape
        out = np.zeros((h + above, w + 2), dtype=bool)
        if closed:  # a closed helmet: no hair anywhere above its lower edge (ponytails, buns, fringes)
            rows = np.flatnonzero(op.any(1))
            out[:rows[-1] + above + 1, :] = True
        else:
            for x in range(w):
                ys = np.flatnonzero(op[:, x])
                if len(ys):
                    for xx in (x, x + 1, x + 2):  # padded by one column on each side
                        out[:ys[0] + above, xx] = True
        back = np.zeros((h + above, HIDE_BACK), dtype=bool)  # room behind the head (facing right)
        if closed:
            back[:rows[-1] + above + 1, :] = True
        return np.concatenate([back, out], 1)

    def register_mask(moving: np.ndarray, fixed: np.ndarray, r: int = 5) -> tuple[int, int]:
        """Integer shift of `moving` (same size canvas) that best overlaps the opaque shape of `fixed`."""
        best, best_s = (0, 0), -1.0
        fo = fixed >= 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                mo = shift(moving, dx, dy) >= 0
                sc = (mo & fo).sum() / max((mo | fo).sum(), 1) - 0.002 * (abs(dx) + abs(dy))
                if sc > best_s:
                    best, best_s = (dx, dy), sc
        return best

    def headgear_alone(name, scale, kx, dx, dy, bw, bh, by_name, sheet_labels, work, bald_origin):
        """A headgear cell minus the bald head it was drawn on, both downsampled at `scale` and stretched `kx` times
        across (the sheet cells are stretched before downsampling, so no pixel is doubled). Returns the sprite and its
        offset from the normal bald sprite (both cells are anchored at their centre x and bottom)."""
        cols = np.minimum((np.arange(int(sheet_labels.shape[1] * kx)) / kx).astype(int), sheet_labels.shape[1] - 1)
        wide = sheet_labels[:, cols]
        stretch = lambda f: {**f, "x0": int(round(f["x0"] * kx)), "x1": int(round(f["x1"] * kx))}

        def down(f):
            return crop(downsample(wide, f, scale, (f["x0"] + f["x1"]) / 2, f["y1"], {**cfg, **work}, n, col.dark))
        fb, fn = stretch(by_name["bald"]), stretch(by_name[name])
        sb, bcx, bcy = down(fb)
        sn, ncx, ncy = down(fn)
        rel = (int(round((dx * kx + (fn["x1"] - fn["x0"]) / 2 - bw * kx / 2) * scale + ncx - bcx)),
               int(round(((fn["y1"] - fn["y0"]) - dy - bh) * scale + ncy - bcy)))
        pad = max(sn.shape) + 8
        hcv = np.full((sb.shape[0] + 2 * pad, sb.shape[1] + 2 * pad), -1, dtype=np.int32)
        pcv = hcv.copy()
        paste(hcv, sn, pad + rel[0], pad + rel[1])
        paste(pcv, sb, pad, pad)
        dist = np.sqrt(((pal_i[np.maximum(hcv, 0)] - pal_i[np.maximum(pcv, 0)]) ** 2).sum(-1))
        keep = (hcv >= 0) & ((pcv < 0) | (dist > 40)) & ~col.of(hcv, col.skin)
        lab, sizes = label(keep)
        if sizes:
            keep = np.isin(lab, [i for i, sz in enumerate(sizes) if sz >= 0.2 * max(sizes)])
        spr, cx, cy = crop(fill_holes(np.where(keep, hcv, -1), passes=2))
        return spr, (cx - pad + bcx - bald_origin[0], cy - pad + bcy - bald_origin[1])

    first_bald = None  # (scale, cell height, cell width) of the first sheet's bald head
    first_bald_sprite = None
    hides: dict[str, list] = {}
    for hc, sheet in head_sheets:
        cells = hc["cells"]
        by_name = dict(zip(cells, sheet.frames))
        bald = sheet.heads[cells.index("bald")]
        bf = by_name["bald"]
        bh, bw = bf["y1"] - bf["y0"], bf["x1"] - bf["x0"]
        if first_bald is None:
            s_head = th / bald["h"] if bald else ref_scale
            first_bald = (s_head, bh, bw)
        else:  # the same bald head drawn on another sheet: match its outline size, not its skin shading
            s_head = first_bald[0] * (first_bald[1] / bh + first_bald[2] / bw) / 2
        work = {"frame": [64, 64], "ground_y": 60, "anchor_x": 32}
        sheet_labels = to_labels(sheet.arr, palette)
        sprites, origin, hi = {}, {}, {}
        for name, f in by_name.items():
            ax = (f["x0"] + f["x1"]) / 2
            spr, cx, cy = crop(downsample(sheet_labels, f, s_head, ax, f["y1"], {**cfg, **work}, n, col.dark))
            sprites[name], origin[name] = spr, (cx, cy)
            hi[name] = sheet.arr[f["y0"]:f["y1"], f["x0"]:f["x1"]]
        # bald head -> mannequin head template (both cut at the neck: align bottom and face side); the bald
        # head of every later sheet goes onto the first sheet's bald head by shape, so all sheets agree
        b = sprites["bald"]
        if first_bald_sprite is None:
            offsets = {"bald": place_on(b, tpatch, (tw - b.shape[1], th - b.shape[0]), by_skin)}
            first_bald_sprite = (b, offsets["bald"])
        else:
            fb, foff = first_bald_sprite
            init = (fb.shape[1] - b.shape[1], fb.shape[0] - b.shape[0])
            offsets = {"bald": place_on(b, fb, init, lambda m, f, r: register_mask(m, f, r), r=5)}
            offsets["bald"] = (offsets["bald"][0] + foff[0], offsets["bald"][1] + foff[1])
        for name in cells:
            if name == "bald" or (name.startswith("~") and name != hc.get("headgear_on")):
                continue
            f = by_name[name]
            dx, dy, by_face = cell_match(hi["bald"], hi[name])
            # the matched points coincide: shift between the two downsampled canvases, then the crops
            sx = (dx + (f["x1"] - f["x0"]) / 2 - bw / 2) * s_head + origin[name][0] - origin["bald"][0]
            sy = ((f["y1"] - f["y0"]) - dy - bh) * s_head + origin[name][1] - origin["bald"][1]
            rel = (int(round(sx)), int(round(sy)))
            if not name.lstrip("~").startswith("head:") and hc.get("headgear_on"):
                # headgear: keep only what it adds to the head it was drawn on, fill pinholes
                parent = sprites[hc["headgear_on"]]
                prel = (0, 0) if hc["headgear_on"] == "bald" else \
                    (offsets[hc["headgear_on"]][0] - offsets["bald"][0], offsets[hc["headgear_on"]][1] - offsets["bald"][1])
                spr = sprites[name]
                pad = max(spr.shape) + 8
                hcv = np.full((parent.shape[0] + 2 * pad, parent.shape[1] + 2 * pad), -1, dtype=np.int32)
                pcv = hcv.copy()
                paste(hcv, spr, pad + rel[0] - prel[0], pad + rel[1] - prel[1])
                paste(pcv, parent, pad, pad)
                dist = np.sqrt(((pal_i[np.maximum(hcv, 0)] - pal_i[np.maximum(pcv, 0)]) ** 2).sum(-1))
                keep = (hcv >= 0) & ((pcv < 0) | (dist > 40)) & ~col.of(hcv, col.skin)
                lab, sizes = label(keep)
                if sizes:
                    big = max(sizes)
                    keep = np.isin(lab, [i for i, sz in enumerate(sizes) if sz >= 0.2 * big])
                spr, cx, cy = crop(fill_holes(np.where(keep, hcv, -1), passes=2))
                sprites[name] = spr
                rel = (cx - pad + prel[0], cy - pad + prel[1])
                if not by_face:  # a closed helmet shows no face to match: centre it on the skull
                    rel = (int(round((b.shape[1] - spr.shape[1]) / 2)), rel[1])
                fit = cfg.get("headgear_fit", {}).get(name.partition(":")[2])
                if fit and hc["headgear_on"] == "bald":
                    # drawn tight on the bald skull, it is smaller than a head with hair: cut it again from the
                    # sheet wider (kx) and a little taller (ky), centred on the head, then nudged (dx, dy)
                    big, rel = headgear_alone(name, s_head * fit.get("ky", 1.0), fit.get("kx", 1.0) / fit.get("ky", 1.0),
                                              dx, dy, bw, bh, by_name, sheet_labels, work, origin["bald"])
                    rel = (rel[0] + fit.get("dx", 0), rel[1] + fit.get("dy", 0))
                    if fit.get("nasal"):
                        # below the rim nothing may hang in front of the face (it read as stray pixels); the nose
                        # guard is a light steel strip on the face's own front edge instead
                        big = big.copy()
                        rim = int(np.argmax((big >= 0).sum(1)))  # widest row: the band
                        steel = int(np.argmin(((palette[:n].astype(int) - np.array(hex_rgb(fit["nasal"]))) ** 2).sum(1)))
                        for y in range(rim + 2, big.shape[0]):
                            yb = y + rel[1]
                            xs_b = np.flatnonzero(b[yb] >= 0) if 0 <= yb < b.shape[0] else []
                            if not len(xs_b):
                                continue
                            front = int(xs_b.max()) - rel[0]  # the face's front edge, in sprite columns
                            big[y, max(front - 2, 0):] = -1
                            if y < rim + 2 + fit.get("nasal_len", 2) and 0 <= front < big.shape[1]:
                                big[y, front] = steel
                    if fit.get("outline"):
                        # downsampling thins the dome's outline away on its lit slope: wherever a light dome pixel
                        # touches the background above the rim, the outline goes back on, outside it
                        rim = int(np.argmax((big >= 0).sum(1)))
                        big = np.pad(big, 1, constant_values=-1)
                        op = big >= 0
                        lit = op & ~col.of(big, col.dark)
                        ring = ~op & (neighbours4(lit) > 0) & (np.arange(big.shape[0])[:, None] < rim + 1)
                        big[ring] = col.outline
                        big, cx_, cy_ = crop(big)
                        rel = (rel[0] - 1 + cx_, rel[1] - 1 + cy_)
                    print(f"headgear {name}: fitted {fit} {spr.shape} -> {big.shape} at {rel}")
                    sprites[name] = spr = big
            offsets[name] = (rel[0] + offsets["bald"][0], rel[1] + offsets["bald"][1])
            fit = cfg.get("headgear_fit", {}).get(name.partition(":")[2]) if name.startswith("headgear:") else None
            if fit and (fit.get("erase") or fit.get("paint")):
                # hand touch-ups of the rigid sprite, given as pixels of the first animation's first frame
                px0, py0 = head_pos[anims[0]["id"]][0]
                ox0, oy0 = px0 + offsets[name][0], py0 + offsets[name][1]
                spr = sprites[name] = sprites[name].copy()
                for x, y in fit.get("erase", []):
                    if 0 <= y - oy0 < spr.shape[0] and 0 <= x - ox0 < spr.shape[1]:
                        spr[y - oy0, x - ox0] = -1
                for x, y, hx in fit.get("paint", []):
                    if 0 <= y - oy0 < spr.shape[0] and 0 <= x - ox0 < spr.shape[1]:
                        spr[y - oy0, x - ox0] = int(np.argmin(((palette[:n].astype(int) - np.array(hex_rgb(hx))) ** 2).sum(1)))
        sizes_txt = {k: v.shape for k, v in sprites.items() if (v >= 0).any() and not k.startswith("~")}
        print(f"{hc['src']}: sprites (h, w) {sizes_txt}; scale {s_head:.4f}; offsets "
              f"{ {k: v for k, v in offsets.items() if not k.startswith('~')} }")
        for name in cells:
            if name == "bald" or name.startswith("~"):
                continue
            kind, _, item = name.partition(":")
            ox, oy = offsets[name]
            hm = hide_mask(sprites[name], closed=item in cfg.get("closed_headgear", [])) if kind == "headgear" else None
            for a in anims:
                frames, hframes = [], []
                for (px, py) in head_pos[a["id"]]:
                    canvas = np.full((fh, fw), -1, dtype=np.int32)
                    paste(canvas, sprites[name], px + ox, py + oy)
                    frames.append(canvas)
                    if hm is not None:
                        hcanvas = np.full((fh, fw), -1, dtype=np.int32)
                        paste(hcanvas, np.where(hm, 0, -1), px + ox - 1 - HIDE_BACK, py + oy - 6)
                        hframes.append(hcanvas >= 0)
                put(kind, item, a["id"], frames)
                if hm is not None:
                    hides.setdefault(item, {})[a["id"]] = hframes

    # torso armour in looping animations: GPT's redraws hide different parts of it behind a swinging arm, so
    # it changes shape from frame to frame. Complete each frame from the fullest frame, moved with the shoulder,
    # clipped to the body and kept behind the near arm.
    for (sl, item), per_anim in layers.items():
        if sl not in cfg.get("torso_slots", ["chest"]):
            continue
        for aid, frames in per_anim.items():
            if aid not in hands or not anim_by_id[aid].get("loop", True):
                continue
            r = int(np.argmax([(f >= 0).sum() for f in frames]))
            sr = hands[aid][r]["shoulder"]
            for fi, f in enumerate(frames):
                sh = hands[aid][fi]["shoulder"]
                ref = shift(frames[r], int(round(sh[0] - sr[0])), int(round(sh[1] - sr[1])))
                fill = (ref >= 0) & (f < 0) & (body[aid][fi] >= 0) & ~arm_masks[aid][fi] \
                    & ~col.of(body[aid][fi], broad_skin)
                f[fill] = ref[fill]

    for (sl, item), per_anim in layers.items():
        if sl not in cfg.get("forearm_slots", ["bracers"]):
            continue
        for aid, frames in per_anim.items():
            if aid not in hands or not anim_by_id[aid].get("loop", True):
                continue
            counts = [int((f >= 0).sum()) for f in frames]
            med = float(np.median(counts))
            m_ = len(frames)
            for fi in range(m_):
                if counts[fi] >= 0.4 * med:
                    continue
                nb = max([(fi - 1) % m_, (fi + 1) % m_], key=lambda j: counts[j])
                if counts[nb] < 0.4 * med:
                    continue
                d = np.array(hands[aid][fi]["hand"]) - np.array(hands[aid][nb]["hand"])
                moved = shift(frames[nb], int(round(d[0])), int(round(d[1])))
                frames[fi] = np.where((moved >= 0) & (body[aid][fi] >= 0), moved, frames[fi])

    # cloaks: (1) pinholes between the cape and the body are closed with the cape's colour; (2) where GPT drew the
    # far arm over the cape, the cape is missing; under a shield that arm is cut away, so the cape is continued under
    # it row by row (only where the cape already lies on the same row) in an `under` strip drawn before the body
    under: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for (sl, item), per_anim in layers.items():
        if sl != "cloak":
            continue
        for aid, frames in per_anim.items():
            und = []
            wst = cfg.get("waist", {}).get(aid, fh)
            for fi, f in enumerate(frames):
                bodyop = body[aid][fi] >= 0
                cl = f >= 0
                lab, sizes = label(~(bodyop | cl | head_box[aid][fi]))  # the head closes pockets at the neck
                border = set(lab[0].tolist()) | set(lab[-1].tolist()) | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
                w_y = wst[fi] if isinstance(wst, list) else wst
                # the cape's shadow tone: what shows through a gap between an arm and the torso, where the cape
                # hangs behind the body (GPT drew it, but 1-3 px pieces are dropped as specks when the kit is cut)
                vals = [v for v in np.unique(f[cl]) if not col.dark[v]]
                shade = min(vals, key=lambda v: col.hsv[v, 2]) if vals else None
                for i, sz in enumerate(sizes):
                    if i in border or shade is None:
                        continue
                    m = lab == i
                    ys_m, xs_m = np.nonzero(m)
                    touching = (m & (neighbours4(cl) > 0)).any() or (m & dilate8(dilate8(cl))).any()
                    if sz > 12 or (ys_m.mean() >= w_y + 8 and not (sz <= 3 and touching)):
                        continue  # between the legs is real background (unless a speck right at the cape)
                    rows = slice(max(ys_m.min() - 1, 0), ys_m.max() + 2)
                    xs_cl = np.nonzero(cl[rows])[1]
                    if len(xs_cl) and ((xs_cl >= xs_m.min() - 7) & (xs_cl <= xs_m.max() + 7)).any():
                        f[m] = shade
                u = np.full((fh, fw), -1, dtype=np.int32)
                if aid in far_cut:
                    cut = far_cut[aid][fi] & bodyop
                    for y in np.flatnonzero(cut.any(1)):
                        xs_c = np.flatnonzero(cut[y])
                        xs_cl = np.flatnonzero(cl[y])
                        for x in xs_c:
                            left = xs_cl[xs_cl < x]
                            if len(left):
                                src = [v for v in f[y, left[::-1]] if not col.dark[v]] or [f[y, left[-1]]]
                                u[y, x] = src[0]
                und.append(u)
            if any((u >= 0).any() for u in und):
                under.setdefault(("cloak", item), {})[aid] = und

    sprites_hi: dict[str, np.ndarray] = {}
    if gear_arr is not None:
        for name, c in zip(gear["cells"], sheet_cells(gear_arr, gear["rows"])):
            rgba = gear_arr[c["y0"]:c["y1"], c["x0"]:c["x1"]].copy()
            rgba[rgba[..., 3] < ALPHA] = 0
            sprites_hi[name] = rgba
        # a curved blade drawn bending the other way: mirror it (once) before it is posed
        for nm_ in {w.get(k) for w in weapons.values() if w.get("mirror") for k in ("carry", "attack_sprite")} - {None}:
            if nm_ in sprites_hi:
                sprites_hi[nm_] = np.ascontiguousarray(sprites_hi[nm_][:, ::-1])
        ref = sprites_hi[gear["scale_from"]["cell"]]
        s_gear = gear["scale_from"]["length"] / ref.shape[0]
        shields = [nm for nm in gear["cells"] if nm.startswith("shield:")]
        s_shield = gear["shield_height"] / max(sprites_hi[nm].shape[0] for nm in shields) if shields else s_gear
        print(f"gear: scale {s_gear:.4f}, shields {s_shield:.4f}; "
              f"{ {k: v.shape[:2] for k, v in sprites_hi.items()} }")
    sprite_scale: dict[str, float] = {}
    for g, arr in extra_gear:
        for name, c in zip(g["cells"], sheet_cells(arr, g["rows"])):
            rgba = arr[c["y0"]:c["y1"], c["x0"]:c["x1"]].copy()
            rgba[rgba[..., 3] < ALPHA] = 0
            sprites_hi[name] = rgba
            sprite_scale[name] = g["length"] / rgba.shape[0]
        print(f"{g['src']}: { {k: (sprites_hi[k].shape[:2], round(sprite_scale[k], 4)) for k in g['cells']} }")

    def place(name, pivot_key, angle, target, scale):
        """Render a sheet sprite at native scale: pivot -> target (continuous coords), blade along `angle`."""
        spr = sprites_hi[name]
        if isinstance(pivot_key, float):  # a point on the shaft, as a share of the height from the top (polearms)
            op_ = spr[..., 3] >= ALPHA
            rows_ = np.flatnonzero(op_.any(1))
            y_ = rows_[0] + pivot_key * (rows_[-1] + 1 - rows_[0])
            xs_ = np.flatnonzero(op_[min(int(y_), op_.shape[0] - 1)])
            pv = (float(xs_.mean()) if len(xs_) else spr.shape[1] / 2, float(y_))
        else:
            pv = upright_points(spr)[pivot_key]
        if angle is None:
            rot, rp = spr, pv
        else:
            rot, rp = rotate_sprite(spr, pv, angle)
        target = (np.floor(target[0]) + 0.5, np.floor(target[1]) + 0.5)
        return resample(to_labels(rot, palette), scale, rp, target, (fh, fw), n, col.dark, thin)

    def blade_angle(name, pivot_key, angle, target, scale):
        """Lower the blade toward the horizontal if its tip would go into the ground."""
        pts = upright_points(sprites_hi[name])
        length = np.hypot(pts["tip"][0] - pts[pivot_key][0], pts["tip"][1] - pts[pivot_key][1]) * scale
        limit = ground - 2.0
        if target[1] + length * np.sin(np.radians(angle)) > limit:
            angle = float(np.degrees(np.arcsin(np.clip((limit - target[1]) / length, -1, 1))))
        return angle

    carry_angle = cfg.get("carry_angle", {})
    for wid, w in weapons.items():
        # carried in the near hand while idle / walking / running
        if "carry" in w and sprites_hi:
            for aid, hs in hands.items():
                if anim_by_id[aid].get("weapon"):
                    continue
                frames = []
                for img, h in zip(body[aid], hs):
                    target = (h["hand"][0] + 0.5, h["hand"][1] + 0.5)
                    sc = sprite_scale.get(w["carry"], s_gear)
                    if "grip_frac" in w:  # a polearm: held on its shaft, head up and forward; the head is in front of it
                        ang = carry_angle[w["kind"]][aid]
                        f = place(w["carry"], float(w["grip_frac"]), ang, target, sc)
                        px_, py_ = head_pos[aid][len(frames)]
                        hb_ = np.zeros((fh, fw), dtype=bool)
                        hb_[py_:py_ + th, px_:px_ + tw] = tmask
                        f[hb_] = -1
                    else:
                        ang = blade_angle(w["carry"], "grip", carry_angle[w["kind"]][aid], target, sc)
                        f = place(w["carry"], "grip", ang, target, sc)
                    f[fist(img, h["hand"])] = -1  # fingers wrap the grip
                    frames.append(f)
                put("weapon", wid, aid, frames)
        if "carry_from" in w:  # a weapon cut out of an attack frame, moved to the near hand
            cf = w["carry_from"]
            src = layers[("weapon", wid)][cf["anim"]][cf["frame"]]
            for aid, hs in hands.items():
                if anim_by_id[aid].get("weapon"):
                    continue
                frames, moved_tips = [], []
                src_tips = weapon_tips.get((wid, cf["anim"]), [None] * (cf["frame"] + 1))[cf["frame"]]
                for fi, h in enumerate(hs):
                    ddx, ddy = int(round(h["hand"][0] - cf["grip"][0])), int(round(h["hand"][1] - cf["grip"][1]))
                    f = shift(src, ddx, ddy)
                    moved_tips.append(tuple((x + ddx, y + ddy) for x, y in src_tips) if src_tips else None)
                    px, py = head_pos[aid][fi]
                    hb = np.zeros((fh, fw), dtype=bool)
                    hb[py:py + th, px:px + tw] = tmask
                    f[hb] = -1  # the head is in front of a bow held at the side
                    frames.append(f)
                put("weapon", wid, aid, frames)
                weapon_tips[(wid, aid)] = moved_tips
        # its own sprite in the attack of the weapon it replaces, posed like that weapon frame by frame
        if "attack_sprite" in w and sprites_hi:
            for a in anims:
                if a.get("weapon") != w["pose_from"]:
                    continue
                frames, poses = [], []
                ref_spr = sprites_hi[w["pose_sprite"]]
                rp = upright_points(ref_spr)
                own = upright_points(sprites_hi[w["attack_sprite"]])
                for fi, ref in enumerate(layers[("weapon", a["weapon"])][a["id"]]):
                    hands_px = col.of(body[a["id"]][fi], broad_skin)
                    pose = match_pose(ref, ref_spr, rp["guard"], s_gear, skin=hands_px,
                                      grip_off=np.hypot(rp["grip"][0] - rp["guard"][0], rp["grip"][1] - rp["guard"][1]) * s_gear)
                    if pose is None:
                        frames.append(np.full((fh, fw), -1, dtype=np.int32))
                        continue
                    (gx, gy), ang, iou = pose
                    # its grip goes into the fist(s) holding the original weapon: skin blobs touching it
                    lab_h, sizes_h = label(hands_px & dilate(dilate(ref >= 0)))
                    blobs = sorted([i for i, sz in enumerate(sizes_h) if sz >= 3], key=lambda i: -sizes_h[i])
                    blobs = blobs[:2] if w["hands"] == 2 else blobs[:1]
                    if blobs:
                        hy, hx = np.nonzero(np.isin(lab_h, blobs))
                        gx, gy = float(hx.mean() + 0.5), float(hy.mean() + 0.5)
                        if "grip_frac" in w:  # a long head must not go into the ground
                            head_len = float(w["grip_frac"]) * sprites_hi[w["attack_sprite"]].shape[0] \
                                * sprite_scale.get(w["attack_sprite"], s_gear)
                            if gy + head_len * np.sin(np.radians(ang)) > ground - 2:
                                ang = float(np.degrees(np.arcsin(np.clip((ground - 2 - gy) / head_len, -1, 1))))
                        poses.append(("grip", round(gx, 1), round(gy, 1), round(ang), round(iou, 2)))
                        f = place(w["attack_sprite"], float(w["grip_frac"]) if "grip_frac" in w else "grip", ang,
                                  (gx, gy), sprite_scale.get(w["attack_sprite"], s_gear))
                    else:
                        poses.append(("guard", round(gx, 1), round(gy, 1), round(ang), round(iou, 2)))
                        f = place(w["attack_sprite"], float(w["grip_frac"]) if "grip_frac" in w else "guard", ang,
                                  (gx, gy), sprite_scale.get(w["attack_sprite"], s_gear))
                    # where the original weapon was hidden by a hand or the head, this one is too
                    f[(ref < 0) & (col.of(body[a["id"]][fi], broad_skin) | head_box[a["id"]][fi])] = -1
                    frames.append(f)
                put("weapon", wid, a["id"], frames)
                print(f"weapon {wid} in {a['id']}: poses from {w['pose_from']} {poses}")

    # shields: held in guard at the front of the chest (near shoulder + a per-animation offset), so a shield does not
    # swing with GPT's far hand from front to back. Seen from the side its face is turned toward the enemy: the
    # face-on sheet sprite is squashed across (`shield.squash`), so it does not hide half of the body. The far arm,
    # which would show outside the torso, is cut from the layers below (masks/far_arm); the cut edge of the back
    # gets an outline that travels with the shield.
    sh_cfg = cfg.get("shield", {})
    for nm in [x for x in (gear["cells"] if gear_arr is not None else []) if x.startswith("shield:")]:
        sid = nm.partition(":")[2]
        src = nm
        if sh_cfg.get("squash", 1.0) != 1.0:
            hi = sprites_hi[nm]
            w_new = max(1, int(round(hi.shape[1] * sh_cfg["squash"])))
            src = nm + "@side"
            sprites_hi[src] = np.asarray(Image.fromarray(hi, "RGBA").resize((w_new, hi.shape[0]), Image.Resampling.NEAREST))
        for aid, hs in hands.items():
            gx, gy = sh_cfg.get("guard", {}).get(aid, [6, 7])
            # the shield arm is alive: it follows the torso's bob (taken from the head, the shoulder annotations miss
            # it), swings a share of the far hand's swing (clamped, so it never goes behind the back) and tilts with it
            k_x, max_x, tilt = sh_cfg.get("sway", {}).get(aid, [0.0, 0.0, 0.0])
            far_x = np.array([h["far"][0] for h in hs], float)
            head_y = np.array([head_pos[aid][i][1] for i in range(len(hs))], float)
            sh_y = np.array([h["shoulder"][1] for h in hs], float)
            raw_dx = np.clip(k_x * (far_x - far_x.mean()), -max_x, max_x)
            if anim_by_id[aid].get("loop", True):  # GPT's hand jumps between some frames: smooth around the cycle
                raw_dx = 0.5 * raw_dx + 0.25 * (np.roll(raw_dx, 1) + np.roll(raw_dx, -1))
            frames = []
            for fi, h in enumerate(hs):
                dx = float(raw_dx[fi])
                c = (h["shoulder"][0] + gx + dx, sh_y.mean() + gy + head_y[fi] - head_y.mean())
                ang = -90.0 - tilt * dx if tilt else None
                f = place(src, "centre", ang, (c[0] + 0.5, c[1] + 0.5), s_shield)
                img = body[aid][fi]
                op = img >= 0
                cut = far_cut[aid][fi] & op
                edge = op & ~cut & ~col.of(img, col.dark) & (neighbours4(cut) > 0)
                f[edge & (f < 0)] = col.outline
                frames.append(f)
            put("shield", sid, aid, frames)

    # recoloured variants (same shape and animation, another material)
    for wid, w in weapons.items():
        if "recolor_of" not in w:
            continue
        if "lacquer" in w:  # a bow of its own look on the same limbs: see lacquer_bow()
            palette, frames_by_anim = lacquer_bow(palette, w, layers[("weapon", w["recolor_of"])],
                                                  weapon_tips, string_idx.get(w["recolor_of"], -2), col)
            for aid, frames in frames_by_anim.items():
                put("weapon", wid, aid, frames)
            continue
        used = sorted({int(v) for frames in layers[("weapon", w["recolor_of"])].values() for f in frames for v in np.unique(f) if v >= 0})
        hsv = col.hsv
        mapping = np.arange(len(palette))
        extra = []
        for i in used:
            h, s, v = hsv[i]
            wood_like = col.family("wood")[i] or col.family("leather")[i] or (i == int(np.argmin(
                ((palette[:len(col.palette)].astype(int) - np.array(hex_rgb(w.get("limb_outline", "#3d2414")))) ** 2).sum(1))))
            if wood_like and not broad_skin[i] and not col.skin[i] and not col.linen[i]:
                rgb = colorsys.hsv_to_rgb(w["hue"] / 360, min(1.0, s * w.get("sat", 1.0) + 0.05),
                                          min(1.0, max(v * w.get("val", 1.0), w.get("val_floor", 0.0))))
                extra.append(np.round(np.array(rgb) * 255).astype(np.uint8))
                mapping[i] = len(palette) + len(extra) - 1
        if extra:
            palette = np.concatenate([palette, np.array(extra)])
        for aid, frames in layers[("weapon", w["recolor_of"])].items():
            put("weapon", wid, aid, [np.where(f >= 0, mapping[np.maximum(f, 0)], -1) for f in frames])

    # ---- derived layers: drawn from the mannequin's own legs, so they sit on them in every frame
    # (GPT redraws legs a little elsewhere in every kit frame, so leg armour cut from a kit never fits).
    # pants: the cloth below the waist, recoloured by lightness; shins: the leg from the shoe up `length` px.
    hue = col.hsv[:, 0] * 360
    shoe_tab = (hue >= 8) & (hue <= 30) & (col.hsv[:, 1] >= 0.38) & (col.hsv[:, 2] <= 0.42)
    hand_tab = broad_skin & (col.hsv[:, 2] >= 0.62)  # brown leg shading is darker than any skin
    dark_tab = col.hsv[:, 2] < 0.3

    def legs_of(aid: str, fi: int) -> tuple[np.ndarray, np.ndarray]:
        """Cloth pixels of the legs below the waist, and the shoes."""
        img = body[aid][fi]
        op = img >= 0
        lab, sizes = label(col.of(img, shoe_tab))
        shoes = np.isin(lab, [i for i, sz in enumerate(sizes)
                              if sz >= 6 and np.nonzero(lab == i)[0].max() >= ground - 14])  # not the belt
        lab, sizes = label(col.of(img, hand_tab))
        hands_px = np.isin(lab, [i for i, sz in enumerate(sizes) if sz >= 4])
        w = cfg.get("waist", {}).get(aid, fh)
        w = w[fi] if isinstance(w, list) else w
        legs = op & ~shoes & ~col.of(img, dark_tab) & ~hands_px & (yy_g >= w)
        for poly in cfg.get("leg_exclude", {}).get(aid, {}).get(str(fi), []):  # an arm hanging below the waist
            legs &= ~polygon_mask(poly, (fh, fw))
        if aid in hands:
            legs &= ~arm_masks[aid][fi] & ~far_cut[aid][fi]
        return legs, shoes

    # arms for bracers: near arm (shoulder, elbow, hand) and the far arm where it shows. From `hands` (far shoulder next
    # to the near one, far elbow on the shoulder-hand line) or from `arms` (both arms annotated; far may be null)
    def arm_list(aid: str, fi: int) -> list[tuple]:
        out = []
        if aid in hands:
            h = hands[aid][fi]
            S, E, H = np.array(h["shoulder"]), np.array(h["elbow"]), np.array(h["hand"])
            out.append((S, E, H, "near"))
            Sf, Hf = S + [1.0, 0.0], np.array(h["far"])
            lu, lf = np.hypot(*(E - S)), np.hypot(*(H - E))
            Ef = Sf + (Hf - Sf) * lu / max(lu + lf, 1e-6)
            out.append((Sf, Ef, Hf, "far"))
        elif aid in cfg.get("arms", {}):
            v = cfg["arms"][aid][fi]
            out.append((np.array(v[0:2], float), np.array(v[2:4], float), np.array(v[4:6], float), "near"))
            if v[6]:
                w = v[6]
                out.append((np.array(w[0:2], float), np.array(w[2:4], float), np.array(w[4:6], float), "near"))
        return out

    def arm_param(S, E, H):
        """Distance of every pixel from the polyline hand-elbow-shoulder, and its position along it from the hand."""
        px, py = xx_g + 0.5, yy_g + 0.5
        res = []
        for P0, P1, base in ((H, E, 0.0), (E, S, float(np.hypot(*(E - H))))):
            d = P1 - P0
            t = np.clip(((px - P0[0]) * d[0] + (py - P0[1]) * d[1]) / max(float((d ** 2).sum()), 1e-6), 0, 1)
            res.append((np.hypot(px - (P0[0] + t * d[0]), py - (P0[1] + t * d[1])), base + t * float(np.hypot(*d))))
        (d1, s1), (d2, s2) = res
        return np.minimum(d1, d2), np.where(d1 <= d2, s1, s2)

    derived_range: dict[tuple[str, str], tuple[int, int]] = {}
    for dv in cfg.get("derived", []):
        ramp = np.array([hex_rgb(h) for h in dv["ramp"]], dtype=np.uint8)
        r0 = len(palette)
        palette = np.concatenate([palette, ramp])
        derived_range[(dv["slot"], dv["id"])] = (r0, r0 + len(ramp))
        for a in anims:
            frames = []
            for fi in range(a["frames"]):
                legs, shoes = legs_of(a["id"], fi)
                v = col.value(body[a["id"]][fi])
                f = np.full((fh, fw), -1, dtype=np.int32)
                if dv["from"] == "legs":
                    tone = np.select([v >= 0.84, v >= 0.76, v >= 0.62], [0, 1, 2], 3)
                    f[legs] = r0 + tone[legs]
                elif dv["from"] == "arms":  # sleeve from the wrist up a share of the arm (`coverage`)
                    img = body[a["id"]][fi]
                    op = img >= 0
                    lab, sizes = label(col.of(img, hand_tab))
                    skin_px = np.isin(lab, [i for i, sz in enumerate(sizes) if sz >= 3])
                    sleeve_ok = op & ~col.of(img, dark_tab) & ~skin_px
                    taken = np.zeros((fh, fw), dtype=bool)
                    for S, E, H, kind in arm_list(a["id"], fi):
                        dist_l, s_along = arm_param(S, E, H)
                        length = float(np.hypot(*(E - H)) + np.hypot(*(S - E)))
                        wrist = 1.5
                        cov = dv["coverage"] * (length - wrist)
                        ok = sleeve_ok & (dist_l <= 2.4) & ~taken
                        if kind == "far":
                            # the far arm is behind the torso; it shows only where it swings out behind the back,
                            # which is the annotated far-arm region (the one a shield cuts away)
                            vis = far_cut[a["id"]][fi] if a["id"] in far_cut else np.zeros((fh, fw), dtype=bool)
                            if not vis.any():
                                continue
                            ok = sleeve_ok & vis & ~taken
                            if a["id"] in arm_masks:
                                ok &= ~arm_masks[a["id"]][fi]
                        # the arm's own sleeve: flood from the wrist, so a torso next to the arm is not taken
                        region = ok & (np.hypot(xx_g + 0.5 - H[0], yy_g + 0.5 - H[1]) <= 3.6)
                        for _ in range(40):
                            grown = region | (dilate(region) & ok)
                            if (grown == region).all():
                                break
                            region = grown
                        sp = s_along - wrist
                        arm = region & (sp <= cov)
                        tone = np.where(v >= 0.84, 1, np.where(v >= 0.7, 2, 3))
                        f[arm] = r0 + tone[arm]
                        f[arm & (sp > cov - 1.5)] = r0  # light rim at the top
                        f[arm & (sp <= 1.0)] = r0 + 1  # cuff at the wrist
                        if dv["coverage"] >= 0.75:  # elbow cop
                            f[arm & (np.hypot(xx_g + 0.5 - E[0], yy_g + 0.5 - E[1]) <= 1.3)] = r0
                        if dv["coverage"] < 1.0:
                            f[region & ~arm & (sp > cov) & (sp <= cov + 1.0)] = col.outline  # the upper edge
                        taken |= region
                else:  # shins: geodesic distance up the leg from the shoe (any leg angle, a kicked-up foot too)
                    # where the back shin passes behind the front leg, the front leg above its own greave is closed
                    open_legs = legs.copy()
                    for poly in cfg.get("shin_exclude", {}).get(a["id"], {}).get(str(fi), []):
                        open_legs &= ~polygon_mask(poly, (fh, fw))
                    near1 = open_legs & dilate8(shoes)
                    near2 = open_legs & dilate8(dilate8(shoes)) & ~near1
                    dist = geodesic(np.where(near1, 1.0, np.where(near2, 2.0, np.inf)), open_legs)
                    length = dv.get("length", 8)
                    shin = open_legs & (dist <= length)
                    tone = np.where(v >= 0.84, 1, np.where(v >= 0.7, 2, 3))
                    f[shin] = r0 + tone[shin]
                    # the plate's upper edge: leg cloth right above it (at the knee, or where a closed front leg starts)
                    edge = legs & ~shin & (neighbours4(shin) > 0) & ((dist > length) | ~open_legs)
                    below_edge = np.zeros_like(edge)
                    below_edge[1:] = edge[:-1]
                    top = shin & ((dist > length - 1.5) | below_edge)
                    ankle = shin & (dist <= 1.0)
                    f[top] = r0  # light rim at the knee
                    f[ankle] = r0 + 1
                    f[edge] = col.outline
                    body_part = shin & ~top & ~ankle
                    for y in range(fh):  # light from the upper left: one lit column per plate row
                        xs = np.flatnonzero(body_part[y])
                        for run in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1) if len(xs) else []:
                            if len(run) >= 3:
                                f[y, run[1]] = r0
                frames.append(f)
            put(dv["slot"], dv["id"], a["id"], frames)
        print(f"derived {dv['slot']}:{dv['id']} from the body's {dv['from']}")

    # ---- materials: one shape, many materials. The piece's own pixels (its colour family as drawn, or its derived
    # ramp) are recoloured by lightness into the material's 5-tone ramp; outline and other details stay
    mat_cfg = cfg.get("materials", {})
    mat_slots = cfg.get("material_slots", {})
    kit_family: dict[tuple[str, str], np.ndarray] = {}
    for k, kit in enumerate(cfg["kits"]):
        for ii, it in enumerate(kit["items"]):
            fam = np.zeros(len(palette), dtype=bool)
            fam[:n] = col.family(it["color"])
            if (k, ii) in ramp_of:
                fam[ramp_of[(k, ii)][0]:ramp_of[(k, ii)][1]] = True
            kit_family[(it["slot"], it["id"])] = fam
    value_of = palette.astype(float).max(1) / 255.0
    tone_layers: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for (sl, item), per_anim in list(layers.items()):
        if sl not in mat_slots:
            continue
        fam = np.zeros(len(palette), dtype=bool)
        if (sl, item) in derived_range:
            fam[derived_range[(sl, item)][0]:derived_range[(sl, item)][1]] = True
        elif (sl, item) in kit_family:
            fam[:len(kit_family[(sl, item)])] = kit_family[(sl, item)]
        elif sl == "headgear":
            fam[:n] = col.family(cfg.get("headgear_color", "steel"))
        idx = np.concatenate([f[f >= 0] for frames in per_anim.values() for f in frames])
        idx = idx[fam[idx]]
        if not len(idx):
            continue
        q = np.quantile(value_of[idx], [0.12, 0.35, 0.65, 0.88])  # tone edges, dark to light
        tone_of = np.full(len(palette), -1)
        for i in np.unique(idx):
            tone_of[i] = 4 - int(np.searchsorted(q, value_of[i]))  # 0 = highlight, 2 = the material colour, 4 = deepest
        tone_layers[(sl, item)] = {aid: [np.where(f >= 0, tone_of[np.maximum(f, 0)], -1) for f in frames]
                                   for aid, frames in per_anim.items()}
    print(f"materials: tone maps for {len(tone_layers)} pieces")

    # ---- write strips, masks + doll.json
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in anims:
        save_strip(body[a["id"]], palette, out_dir / "body" / f"{a['id']}.png")
    mask_paths = {}
    white = np.array([[255, 255, 255]], dtype=np.uint8)
    for aid, ms in arm_masks.items():
        rel = f"masks/near_arm/{aid}.png"
        save_strip([np.where(m, 0, -1) for m in ms], white, out_dir / rel)
        mask_paths[aid] = rel
    far_paths = {}
    for aid, ms in far_cut.items():
        rel = f"masks/far_arm/{aid}.png"
        save_strip([np.where(m, 0, -1) for m in ms], white, out_dir / rel)
        far_paths[aid] = rel
    hide_paths: dict[str, dict[str, str]] = {}
    for item, per_anim in hides.items():
        for aid, ms in per_anim.items():
            rel = f"masks/headgear/{item}/{aid}.png"
            save_strip([np.where(m, 0, -1) for m in ms], white, out_dir / rel)
            hide_paths.setdefault(item, {})[aid] = rel
    labels_cfg = cfg["labels"]
    order = list(labels_cfg)
    attack_of = {weapons[a["weapon"]]["kind"]: a["id"] for a in anims if a.get("weapon")}
    slots_out = []
    for slot in cfg["slots"]:
        items = []
        if slot["id"] == "body":
            items.append({"id": "base", "label": "Манекен",
                          "strips": {a["id"]: f"body/{a['id']}.png" for a in anims}})
        found = [(item, per_anim) for (sl, item), per_anim in layers.items() if sl == slot["id"]]
        found.sort(key=lambda x: order.index(f"{slot['id']}:{x[0]}") if f"{slot['id']}:{x[0]}" in order else 999)
        for item, per_anim in found:
            strips = {}
            for anim_id in [a["id"] for a in anims if a["id"] in per_anim]:
                rel = f"layers/{slot['id']}/{item}/{anim_id}.png"
                save_strip(per_anim[anim_id], palette, out_dir / rel)
                strips[anim_id] = rel
            entry = {"id": item, "label": labels_cfg.get(f"{slot['id']}:{item}", item), "strips": strips}
            if slot["id"] == "weapon":
                w = weapons[item]
                entry.update({"kind": w["kind"], "hands": w["hands"], "attack": w.get("attack") or attack_of.get(w["kind"])})
            if item in hide_paths and slot["id"] == "headgear":
                entry["hides"] = hide_paths[item]  # head pixels to cut before drawing this headgear
            if (slot["id"], item) in tone_layers:
                # tone map: the piece's own pixels, 0 = highlight .. 2 = the material colour .. 4 = deepest
                # (grey 0, 60, 120, 180, 240); everything else transparent. Any colour recolours the piece.
                entry["material"] = cfg.get("item_material", {}).get(f"{slot['id']}:{item}")
                entry["tones"] = {}
                greys = np.array([[t * 60] * 3 for t in range(5)], dtype=np.uint8)
                for anim_id, frames_t in tone_layers[(slot["id"], item)].items():
                    rel = f"layers/{slot['id']}/{item}/{anim_id}_tones.png"
                    save_strip(frames_t, greys, out_dir / rel)
                    entry["tones"][anim_id] = rel
            if (slot["id"], item) in under:  # drawn before the body: shows only where a shield cut the far arm
                entry["under"] = {}
                for anim_id, frames_u in under[(slot["id"], item)].items():
                    rel = f"layers/{slot['id']}/{item}/{anim_id}_under.png"
                    save_strip(frames_u, palette, out_dir / rel)
                    entry["under"][anim_id] = rel
            items.append(entry)
        slots_out.append({**slot, "items": items, **({"materials": mat_slots[slot["id"]]} if slot["id"] in mat_slots else {})})
    (out_dir / "doll.json").write_text(json.dumps({
        "name": cfg.get("name", "doll"),
        "frame_size": [fw, fh],
        "pivot": {"x": cfg["anchor_x"], "y": cfg["ground_y"] + 1},
        "facing": "right",
        "view": "side",
        "compose": "First draw the `under` strips of the worn items, then the slots in order. When a shield is worn: cut masks.far_arm out of every slot listed in "
                   "the shield slot's `cuts` (the far arm is behind the shield), and right after the shield draw "
                   "every earlier slot again clipped to masks.near_arm (the sword arm is in front of the shield). "
                   "When headgear is worn, cut its `hides` mask out of the head layer (hair that would poke "
                   "through).",
        "animations": [{**{k: a[k] for k in ("id", "label", "frames", "fps", "loop") if k in a},
                        **({"weapon": weapons[a["weapon"]]["kind"]} if a.get("weapon") else {})} for a in anims],
        "masks": {"near_arm": mask_paths, "far_arm": far_paths},
        "materials": {**{k: {"label": v["label"], "ramp": v["ramp"]} for k, v in mat_cfg.items()},
                      **{k: {"label": v} for k, v in cfg.get("material_labels_extra", {}).items()}},
        "recolor": "A piece with `tones` can take any material colour: for each frame, pixels of its tone map "
                   "(grey 0/60/120/180/240 = tone 0..4, highlight to deepest) take tone k of a 5-colour ramp; "
                   "tone 2 is the material colour itself. `materials` lists ready ramps.",
        "slots": slots_out,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_dir}: {sum(len(v) for v in layers.values())} layer strips + body + "
          f"{len(mask_paths) + len(far_paths)} arm masks")


if __name__ == "__main__":
    main(sys.argv[1])
