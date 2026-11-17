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
    cols = count // rows
    cell_w = mask.shape[1] / cols
    for band, (y0, y1) in enumerate(bands):
        sub = mask[y0:y1]
        for col_i, (x0, x1) in enumerate(split_runs(sub.sum(0), cols)):
            ys = np.flatnonzero(sub[:, x0:x1].any(1))
            frames.append({"x0": x0, "x1": x1, "y0": y0 + int(ys[0]), "y1": y0 + int(ys[-1]) + 1, "band": band,
                           "cell_x0": col_i * cell_w})
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


def close_gaps(img: np.ndarray, cover: np.ndarray, cut: np.ndarray, max_size: int = 24) -> np.ndarray:
    """Background pockets enclosed by the body and a layer drawn over it (a weapon posed a little off GPT's, whose cut
    left a slit in the body): filled from the body's own neighbouring colours, outside in. Only pockets mostly inside
    `cut` (where GPT's weapon was) count: a real gap between an arm and the body stays."""
    out = img.copy()
    lab, sizes = label(~((img >= 0) | (cover >= 0)))
    edge = set(lab[0].tolist()) | set(lab[-1].tolist()) | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    for i, sz in enumerate(sizes):
        todo = lab == i
        if i in edge or sz > max_size or (todo & cut).sum() * 2 < sz:
            continue
        for _ in range(max_size):
            if not todo.any():
                break
            for y, x in zip(*np.nonzero(todo)):
                nb = [out[yy, xx] for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                      if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1] and out[yy, xx] >= 0 and not todo[yy, xx]]
                if nb:
                    out[y, x] = max(set(nb), key=nb.count)
            todo &= out < 0
    return out


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

    def __init__(self, path: Path, count: int, rows: int, first: int = 0, take: int | None = None):
        """`first` / `take`: an animation that is only a part of the sheet (e.g. one row of a 2-row sheet that holds
        two animations): the whole sheet is cut into `count` frames, then frames first .. first + take are kept."""
        self.arr = load(path)
        self.frames = find_frames(self.arr[..., 3] >= ALPHA, count, rows)[first:first + (take or count)]

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
        elif mode == "cell":  # where GPT put the figure in its cell (a fall: neither the head nor a foot stays put)
            rel = out[0][0] - self.frames[0]["cell_x0"]
            for i, f in enumerate(self.frames):
                out[i][0] = f["cell_x0"] + rel
        return out

    def _head_x(self, i, f):
        h = self.heads[i]
        return h["cx"] if h else (f["x0"] + f["x1"]) / 2

    def _back_foot(self, f):
        a = self.arr[..., 3][f["y1"] - max(3, (f["y1"] - f["y0"]) // 25): f["y1"], f["x0"]:f["x1"]] >= ALPHA
        xs = np.flatnonzero(a.any(0))
        return f["x0"] + (xs[0] if len(xs) else 0)

    def render(self, labels, scale, anchors, cfg, n, dark, thin=None, frame_scale=None):
        """`frame_scale` {frame: factor}: GPT drew that frame's figure smaller or bigger than the others."""
        fs = frame_scale or {}
        return [downsample(labels, f, scale * fs.get(str(i), 1.0), ax, bottom, cfg, n, dark, thin)
                for i, (f, (ax, bottom)) in enumerate(zip(self.frames, anchors))]


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


def rot_labels(lab: np.ndarray, angle: float, pivot: tuple[float, float], k: int = 4):
    """Turn a label image (-1 = empty) by `angle` degrees clockwise (y down) about `pivot` (continuous coordinates):
    nearest rotation of a k times enlarged copy, then a majority vote per k x k block (pixel art keeps its clusters).
    Returns the cropped image and where the pivot is in it."""
    h, w = lab.shape
    q = angle / 90.0
    if abs(q - round(q)) < 1e-6:  # quarter turns are exact
        q = int(round(q)) % 4
        px, py = pivot
        return np.rot90(lab, -q).copy(), [(px, py), (h - py, px), (w - px, h - py), (py, w - px)][q]
    r = int(np.ceil(np.hypot(h, w))) + 3
    side = 2 * r
    canvas = np.zeros((side, side), np.int32)  # label + 1, 0 = empty
    ox, oy = r - int(np.floor(pivot[0])), r - int(np.floor(pivot[1]))
    canvas[oy:oy + h, ox:ox + w] = lab + 1
    piv = (ox + pivot[0], oy + pivot[1])
    big = np.kron(canvas, np.ones((k, k), np.int32))
    im = Image.fromarray(big).rotate(-angle, resample=Image.Resampling.NEAREST, center=(piv[0] * k, piv[1] * k))
    big = np.asarray(im).reshape(side, k, side, k).transpose(0, 2, 1, 3).reshape(side, side, k * k)
    out = np.full((side, side), -1, np.int32)
    for y, x in zip(*np.nonzero((big > 0).sum(2) * 2 >= k * k)):
        v = big[y, x]
        out[y, x] = int(np.bincount(v[v > 0]).argmax()) - 1
    ys, xs = np.nonzero(out >= 0)
    if not len(ys):
        return out[:1, :1], (0.0, 0.0)
    y0, x0 = int(ys.min()), int(xs.min())
    return out[y0:ys.max() + 1, x0:xs.max() + 1], (piv[0] - x0, piv[1] - y0)


def paste_turned(canvas: np.ndarray, spr: np.ndarray, x: int, y: int, pivot: tuple[float, float], angle: float,
                 centre: tuple[float, float] | None = None) -> None:
    """Paste a label sprite whose top-left would be (x, y) upright, turned by `angle` about `pivot` (sprite coordinates);
    `centre`: where the pivot goes in the canvas (default: where it is upright)."""
    if not angle:
        paste(canvas, spr, x, y)
        return
    cx, cy = centre if centre is not None else (x + pivot[0], y + pivot[1])
    rot, rp = rot_labels(spr, angle, pivot)
    paste(canvas, rot, int(round(cx - rp[0])), int(round(cy - rp[1])))


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
    def open_strip(path: Path, a: dict) -> Strip:
        """`sheet` {name, frames, rows, first}: the animation is frames first .. first + frames of a sheet shared with
        another animation (kits are named after the sheet then)."""
        sh = a.get("sheet")
        if sh:
            return Strip(path, sh["frames"], sh.get("rows", 1), sh["first"], a["frames"]).measure(skin_rgb)
        return Strip(path, a["frames"], a.get("rows", 1)).measure(skin_rgb)

    base = {a["id"]: open_strip(ROOT / a["src"], a) for a in anims}
    kit_strips: dict[tuple[int, str], Strip] = {}
    for k, kit in enumerate(cfg["kits"]):
        for a in anims:
            if a["id"] not in kit.get("anims", [x["id"] for x in anims]):
                continue
            p = ROOT / kit["src"].format(anim=a.get("sheet", {}).get("name", a["id"]))
            if p.exists():
                kit_strips[(k, a["id"])] = open_strip(p, a)
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
    # `palette_anims`: the animations whose sheets make the palette (later animations take its colours, so adding one
    # does not recolour the others)
    pal_anims = set(cfg.get("palette_anims", [a["id"] for a in anims]))
    arrays = [s.arr for aid, s in base.items() if aid in pal_anims] + [st.arr for _, st in head_sheets]
    palette = build_palette(arrays, cfg["max_colors"], cfg["fixed_colors"])
    item_ramps = []
    ramp_of: dict[tuple[int, int], tuple[int, int]] = {}  # (kit, item) -> palette index range of its ramp
    base_n = len(palette)
    for k, kit in enumerate(cfg["kits"]):
        strips_k = [st for (kk, aid), st in kit_strips.items() if kk == k and aid in pal_anims]
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
        # `scale_as`: the other animation of the same sheet (GPT draws one sheet at one scale)
        scales[a["id"]] = scales[a["scale_as"]] if a.get("scale_as") in scales else strip_scale(s, a)
        frames = s.render(to_labels(s.arr, palette), scales[a["id"]], s.anchors(a["anchor"], scales[a["id"]]),
                          cfg, n, col.dark, thin, a.get("frame_scale"))
        body[a["id"]] = [close_outline(f, col.outline) for f in frames]
        print(f"body {a['id']:<16} scale {scales[a['id']]:.4f} heights "
              f"{[int((f >= 0).any(1).sum()) for f in body[a['id']]]}")

    # ---- head template from idle frame 0, then find the head in every frame
    f0 = body[anims[0]["id"]][0]
    head_sil = None  # front / back views: (widest row, top row, centre x) of the mannequin head's silhouette
    if cfg.get("head_rows"):
        # front / back views: the skull is drawn in several shading bands (the back of a head has no single skin
        # tone), so the head is its whole silhouette: everything above the collar (the first row with shirt cloth)
        op0 = f0 >= 0
        rows0 = np.flatnonzero(op0.any(1))
        top0 = int(rows0[0])
        linen0 = col.of(f0, col.linen)
        collar = next((y for y in range(top0, top0 + cfg["head_rows"]) if linen0[y].sum() >= 2), top0 + cfg["head_rows"])
        comp = op0 & ~linen0 & (np.arange(fh)[:, None] < collar)
        lab, sizes = label(comp)
        comp = lab == int(np.argmax(sizes))
        widths = [np.ptp(np.flatnonzero(r)) + 1 if r.any() else 0 for r in comp]
        wide_y = int(np.argmax(widths))
        head_sil = (float(widths[wide_y]), top0, float(np.flatnonzero(comp[wide_y]).mean()) + 0.5)
        tmask_full = comp | (dilate(comp) & ~linen0 & op0)
    else:
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
    head_mask: dict[str, list[np.ndarray]] = {}  # the template's own pixels there (turned with the head)
    # `head_turn` {anim: {frame: degrees clockwise}}: the head lies turned (a fall); the template is turned the same way
    # and searched over the whole frame, and every head / headgear sprite is drawn turned about the template's centre
    head_turn = {a["id"]: [float(cfg.get("head_turn", {}).get(a["id"], {}).get(str(i), 0)) for i in range(len(body[a["id"]]))]
                 for a in anims}
    head_centre: dict[str, list[tuple[float, float]]] = {}
    for a in anims:
        positions, boxes, masks, centres = [], [], [], []
        for fi, img in enumerate(body[a["id"]]):
            ang = head_turn[a["id"]][fi]
            if ang:
                patch_, (pcx, pcy) = rot_labels(tpatch, ang, (tw / 2, th / 2))
                mask_ = patch_ >= 0
            else:
                patch_, mask_, (pcx, pcy) = tpatch, tmask, (tw / 2, th / 2)
            ph, pw = patch_.shape
            best, best_cost = (tx0, ty0), 1e18
            rows = np.flatnonzero((img >= 0).any(1))
            top = rows[0] if len(rows) else 0
            # `head_window` {anim: [dx, dy_min, dy_max]}: search near the idle head only (an attack in place whose blade
            # crosses the face would otherwise match the head elsewhere)
            hw = cfg.get("head_window", {}).get(a["id"])
            if ang:
                ys_rng, xs_rng = range(0, fh - ph), range(0, fw - pw)
            else:
                ys_rng = range(max(0, ty0 + hw[1]), ty0 + hw[2] + 1) if hw else \
                    range(max(0, top - 4), min(fh - th, top + int(cfg["body_height"] * 0.55)))
                xs_rng = range(tx0 - hw[0], tx0 + hw[0] + 1) if hw else range(0, fw - tw)
            for py in ys_rng:
                for px in xs_rng:
                    win = img[py:py + ph, px:px + pw]
                    m = mask_ & (win >= 0)
                    if m.sum() < mask_.sum() * 0.7:
                        continue
                    diff = pal_f[win[m]] - pal_f[patch_[m]]
                    cost = float((diff ** 2).sum()) / m.sum() + 900.0 * (1 - m.sum() / mask_.sum())
                    if cost < best_cost:
                        best, best_cost = (px, py), cost
            # `head_at` {anim: {frame: [x, y]}}: where the search took raised hands for the head (the corner of the
            # upright template; with a turn, of the upright template with the same centre)
            at = cfg.get("head_at", {}).get(a["id"], {}).get(str(fi))
            if at:
                best = (int(round(at[0] + tw / 2 - pcx)), int(round(at[1] + th / 2 - pcy)))
            px, py = best
            centres.append((px + pcx, py + pcy))
            positions.append((int(round(px + pcx - tw / 2)), int(round(py + pcy - th / 2))))
            lab_m = np.full((fh, fw), -1, dtype=np.int32)
            paste(lab_m, np.where(mask_, 0, -1), px, py)
            full = lab_m >= 0
            kill = full & (img >= 0) & ~col.of(img, col.linen)
            img[kill] = -1  # cut the mannequin head (and neck) out of the body
            masks.append(full)
            box = dilate(full)
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
        head_mask[a["id"]] = masks
        head_centre[a["id"]] = centres
        turned = {i: x for i, x in enumerate(head_turn[a["id"]]) if x}
        print(f"head {a['id']:<16} template {tw}x{th} from ({tx0}, {ty0}); found at {positions}"
              + (f"; turned {turned}" if turned else ""))

    # ---- weapons out of attack bodies
    yy_g, xx_g = np.mgrid[0:fh, 0:fw]

    def height_along(aid: str, fi: int) -> np.ndarray:
        """Every pixel's height above the soles, measured up the body: above the ground for a standing figure, along the
        body turned with the head for one that falls or lies (`head_turn`)."""
        ang = head_turn[aid][fi]
        if not ang:
            return (ground - yy_g).astype(float)
        tr = np.radians(ang)
        up = xx_g * np.sin(tr) - yy_g * np.cos(tr)
        return up - up[body[aid][fi] >= 0].min()

    # a waist line for animations without `waist`: the idle's distance from the head centre down to the waist, along
    # the body (turned with the head in a fall)
    _w0 = cfg.get("waist", {}).get(anims[0]["id"])
    waist_k = float(np.mean([(_w0[i] if isinstance(_w0, list) else _w0) - head_centre[anims[0]["id"]][i][1]
                             for i in range(len(body[anims[0]["id"]]))])) if _w0 is not None else None

    def below_waist(aid: str, fi: int) -> np.ndarray:
        w = cfg.get("waist", {}).get(aid)
        if w is not None or waist_k is None:
            w = fh if w is None else (w[fi] if isinstance(w, list) else w)
            return yy_g >= w
        cx, cy = head_centre[aid][fi]
        tr = np.radians(head_turn[aid][fi])
        return (xx_g + 0.5 - cx) * -np.sin(tr) + (yy_g + 0.5 - cy) * np.cos(tr) >= waist_k
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
            # `weapon_cut` {anim: {frame: [[x0, y0, x1, y1], ...]}}: a blade so thin that downsampling left only its
            # outline, which the colour rules leave with the body: weapon and outline colours in these boxes are weapon
            for x0, y0, x1, y1 in cfg.get("weapon_cut", {}).get(a["id"], {}).get(str(fi), []):
                box_ = np.zeros_like(wm)
                box_[y0:y1, x0:x1] = True
                wm |= box_ & (img >= 0) & col.of(img, seed | grow)
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

    # every arm in front of the body: the near arm, plus both arms where `arms` annotates them (front and back
    # views show both arms whole); legs and torso fills keep out of these
    all_arms: dict[str, list[np.ndarray]] = {aid: [m.copy() for m in ms] for aid, ms in arm_masks.items()}
    for aid, rows in cfg.get("arms", {}).items():
        if aid not in all_arms:  # only with `hands` too (front / back views); side attacks keep `leg_exclude`
            continue
        out = all_arms[aid]
        for fi, v in enumerate(rows):
            for arm in (v[0:6], v[6] if len(v) > 6 else None):
                if not arm:
                    continue
                S, E, H = (arm[0], arm[1]), (arm[2], arm[3]), (arm[4], arm[5])
                d = np.minimum(seg_dist(xx_g, yy_g, S, E), seg_dist(xx_g, yy_g, E, H))
                out[fi] |= (d <= cfg.get("arm_radius", 2.0)) & dilate(body[aid][fi] >= 0)

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
        # the kit's other row of the same sheet sets its scale too (`scale_as`), as for the base
        sa = a.get("scale_as")
        scale = strip_scale(kit_strips[(k, sa)], anim_by_id[sa]) if sa and (k, sa) in kit_strips else strip_scale(s, a)
        frames = s.render(to_labels(s.arr, palette), scale, s.anchors(a["anchor"], scale), cfg, n, col.dark, thin,
                          a.get("frame_scale"))
        ref_frames = [close_outline(f, col.outline) for f in base[anim_id].render(
            to_labels(base[anim_id].arr, palette), scales[anim_id],
            base[anim_id].anchors(a["anchor"], scales[anim_id]), cfg, n, col.dark, thin, a.get("frame_scale"))]
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
                    hgt = height_along(anim_id, fi) / cfg["body_height"]
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

    def neck_match(bald_cell: np.ndarray, cell: np.ndarray) -> tuple[float, int, bool]:
        """Like cell_match for front / back views, where no face profile shows: the neck stubs end on one line
        and the neck is the same neck, so the cells are matched by the neck's bottom and its centre."""
        def neck_x(c: np.ndarray) -> float:
            a = c[..., 3] >= ALPHA
            xs = np.nonzero(a[int(a.shape[0] * 0.92):])[1]
            return float(xs.mean()) if len(xs) else a.shape[1] / 2
        return neck_x(bald_cell) - neck_x(cell), cell.shape[0] - bald_cell.shape[0], True

    def hide_mask(spr: np.ndarray, above: int = 6, closed: bool = False) -> np.ndarray:
        """What a headgear hides of the head under it: hair poking out ABOVE it. In each of its columns
        (and one either side) the `above` rows over its top pixel; nothing under the headgear is cut,
        so it can never open a see-through gap. Returned with `above` extra rows on top."""
        op = spr >= 0
        h, w = op.shape
        out = np.zeros((h + above, w + 2), dtype=bool)
        if closed:  # a closed helmet: no hair anywhere above its lower edge (ponytails, buns, fringes); a single
            # dangling outline pixel under the rim is not the edge (it would cut a hole into the neck)
            rows = np.flatnonzero(op.sum(1) >= 3)
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
            if head_sil and cfg.get("head_height"):  # the same head size as the other views (bald sprite height)
                s_head = cfg["head_height"] / bh
            elif head_sil:  # front / back views: the bald head's widest row matches the mannequin head's
                cell_a = sheet.arr[bf["y0"]:bf["y1"], bf["x0"]:bf["x1"], 3] >= ALPHA
                s_head = head_sil[0] / max(np.ptp(np.flatnonzero(r)) + 1 for r in cell_a if r.any())
            else:
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
            if head_sil:  # centred on the mannequin head, the neck ends where the mannequin's did
                init = (int(round(head_sil[2] - tx0 - b.shape[1] / 2)), th - b.shape[0])
                # `head_nudge` [dx, dy]: a bigger head than the mannequin's (sized like the other views) can read as
                # a hunch in a three-quarter back view; set it down onto the shoulders by eye
                init = (init[0] + cfg.get("head_nudge", [0, 0])[0], init[1] + cfg.get("head_nudge", [0, 0])[1])
                offsets = {"bald": init}  # pure geometry: a colour fit would move with every palette change
            else:
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
            dx, dy, by_face = neck_match(hi["bald"], hi[name]) if head_sil else cell_match(hi["bald"], hi[name])
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
                if fit and hc["headgear_on"] == "bald" and any(k in fit for k in ("kx", "ky", "dx", "dy", "nasal", "outline")):
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
            # hand touch-ups: `headgear_fit` for headgear, `head_fit` for heads (a face redrawn pixel by pixel where
            # downsampling lost it, e.g. the eyes of a face turned to the camera)
            fit = cfg.get("headgear_fit" if name.startswith("headgear:") else "head_fit", {}).get(name.partition(":")[2])
            if fit and (fit.get("erase") or fit.get("paint") or fit.get("patch")):
                # hand touch-ups of the rigid sprite, given as pixels of the first animation's first frame
                px0, py0 = head_pos[anims[0]["id"]][0]
                ox0, oy0 = px0 + offsets[name][0], py0 + offsets[name][1]
                spr = sprites[name] = sprites[name].copy()
                if fit.get("patch"):  # a block of pixel art: one character per pixel, " " keeps, "." erases
                    pt = fit["patch"]
                    # `at` is in sprite pixels (from its top-left) with "in": "sprite", else in idle frame 0 pixels
                    bx, by = (ox0, oy0) if pt.get("in") == "sprite" else (0, 0)
                    for dy_, row in enumerate(pt["rows"]):
                        for dx_, ch in enumerate(row):
                            x, y = bx + pt["at"][0] + dx_, by + pt["at"][1] + dy_
                            if ch == " " or not (0 <= y - oy0 < spr.shape[0] and 0 <= x - ox0 < spr.shape[1]):
                                continue
                            spr[y - oy0, x - ox0] = -1 if ch == "." else int(np.argmin(
                                ((palette[:n].astype(int) - np.array(hex_rgb(pt["colors"][ch]))) ** 2).sum(1)))
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
                for fi_h, (px, py) in enumerate(head_pos[a["id"]]):
                    canvas = np.full((fh, fw), -1, dtype=np.int32)
                    ang = head_turn[a["id"]][fi_h]
                    pivot = (tw / 2 - ox, th / 2 - oy)  # the template's centre, in sprite coordinates
                    paste_turned(canvas, sprites[name], px + ox, py + oy, pivot, ang, head_centre[a["id"]][fi_h])
                    if kind == "head" and cfg.get("neck_seam") and not ang:
                        # the neck stub's bottom outline lies across the collar when the neck faces the camera
                        # (isometric): drop it where the body continues below
                        rows_h = np.flatnonzero((canvas >= 0).any(1))
                        if len(rows_h):
                            yb = rows_h[-1]
                            seam = (canvas[yb] >= 0) & col.dark[np.maximum(canvas[yb], 0)] & (body[a["id"]][fi_h][yb] >= 0)
                            canvas[yb][seam] = -1
                        # where the sprite's neck is shorter than the mannequin's (cut away with its head), 1-4 px of
                        # background show between chin and collar: the head layer closes them from its own pixels
                        comb = np.where(canvas >= 0, canvas, body[a["id"]][fi_h])
                        filled = fill_small_holes(comb, 4)
                        gap = (comb < 0) & (filled >= 0) & dilate(canvas >= 0)
                        canvas[gap] = filled[gap]
                    frames.append(canvas)
                    if hm is not None:
                        hcanvas = np.full((fh, fw), -1, dtype=np.int32)
                        paste_turned(hcanvas, np.where(hm, 0, -1), px + ox - 1 - HIDE_BACK, py + oy - 6,
                                     (pivot[0] + 1 + HIDE_BACK, pivot[1] + 6), ang, head_centre[a["id"]][fi_h])
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
                fill = (ref >= 0) & (f < 0) & (body[aid][fi] >= 0) & ~all_arms[aid][fi] \
                    & ~col.of(body[aid][fi], broad_skin)
                f[fill] = ref[fill]

    # `fill` {slot: "all" | "low"}: in any animation (attacks too) a piece GPT hid behind a weapon, a raised arm or a
    # drawn bow is completed from another frame of it, moved with the head, clipped to the body, behind the arms.
    # "all": every frame from the fullest one (torso armour); "low": only frames that lost over 40 % of the piece,
    # from the frame closest to the median (pauldrons, skirts, boots move with the limbs)
    for (sl, item), per_anim in layers.items():
        mode = cfg.get("fill", {}).get(sl)
        if not mode:
            continue
        for aid, frames in per_anim.items():
            if aid not in head_pos:
                continue
            counts = [int((f >= 0).sum()) for f in frames]
            # a frame where the body lies turned (a fall) is neither a source nor a target: a shift cannot turn a piece
            ok = [i for i in range(len(frames)) if not head_turn[aid][i]]
            if not ok:
                continue
            med = float(np.median([counts[i] for i in ok]))
            if mode == "all":
                r, targets = max(ok, key=lambda i: counts[i]), ok
            else:
                r = min(ok, key=lambda i: abs(counts[i] - med))
                targets = [i for i in ok if counts[i] < 0.6 * med]
            rx, ry = head_pos[aid][r]
            for fi in targets:
                hx, hy = head_pos[aid][fi]
                ref = shift(frames[r], hx - rx, hy - ry)
                op = body[aid][fi] >= 0
                arms_ = all_arms[aid][fi] if aid in all_arms else np.zeros_like(op)
                fill = (ref >= 0) & (frames[fi] < 0) & op & ~arms_ & ~col.of(body[aid][fi], broad_skin)
                frames[fi][fill] = ref[fill]

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
            low = float(np.degrees(np.arcsin(np.clip((limit - target[1]) / length, -1, 1))))
            angle = low if np.cos(np.radians(angle)) >= 0 else 180.0 - low  # a blade pointing left stays left
        return angle

    def posed_frames(w: dict, own_name: str, ref_name: str, a: dict, ref_frames: list[np.ndarray]):
        # a parry holds the blade in front of the face: the head cut took GPT's blade there with it, so the head does
        # not hide the posed sprite (slot order puts the weapon behind the head where it should be, e.g. from behind)
        hide_head = not a.get("role", "").startswith("block")
        """A sheet sprite posed like a weapon cut out of GPT's frames: angle and place by matching the reference sprite's
        silhouette to it, the grip into the fist(s) that held it; hidden where a hand or the head hid the original."""
        frames, poses = [], []
        ref_spr = sprites_hi[ref_name]
        rp = upright_points(ref_spr)
        for fi, ref in enumerate(ref_frames):
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
                    head_len = float(w["grip_frac"]) * sprites_hi[own_name].shape[0] * sprite_scale.get(own_name, s_gear)
                    if gy + head_len * np.sin(np.radians(ang)) > ground - 2:
                        ang = float(np.degrees(np.arcsin(np.clip((ground - 2 - gy) / head_len, -1, 1))))
                # a blade's silhouette is nearly the same both ways along its axis: from the fist, the blade goes where
                # GPT's weapon is
                piv_ = float(w["grip_frac"]) if "grip_frac" in w else "grip"
                sc_ = sprite_scale.get(own_name, s_gear)
                f = place(own_name, piv_, ang, (gx, gy), sc_)
                # (a polearm is held off-centre, its long butt would always win: swords only)
                f_back = place(own_name, piv_, ang + 180.0, (gx, gy), sc_) if "grip_frac" not in w else f
                if ((f_back >= 0) & (ref >= 0)).sum() > ((f >= 0) & (ref >= 0)).sum():
                    f, ang = f_back, (ang + 180.0 + 180.0) % 360.0 - 180.0
                poses.append(("grip", round(gx, 1), round(gy, 1), round(ang), round(iou, 2)))
            else:
                poses.append(("guard", round(gx, 1), round(gy, 1), round(ang), round(iou, 2)))
                f = place(own_name, float(w["grip_frac"]) if "grip_frac" in w else "guard", ang, (gx, gy),
                          sprite_scale.get(own_name, s_gear))
            # where the original weapon was hidden by a hand or the head, this one is too
            f[(ref < 0) & (col.of(body[a["id"]][fi], broad_skin) | (head_box[a["id"]][fi] if hide_head else False))] = -1
            frames.append(f)
        return frames, poses

    carry_angle = cfg.get("carry_angle", {})

    def carry_ang(kind: str, aid: str, fi: int):
        """`carry_angle[kind][anim]`: one angle, or one per frame (a fall: the weapon goes down with the hand)."""
        v = carry_angle[kind][aid]
        return v[fi] if isinstance(v, list) else v

    for wid, w in weapons.items():
        # carried in the near hand while idle / walking / running
        if "carry" in w and w["carry"] in sprites_hi:  # (a sheet not generated yet: the weapon is left out)
            for aid, hs in hands.items():
                if anim_by_id[aid].get("weapon"):
                    continue
                frames = []
                for img, h in zip(body[aid], hs):
                    target = (h["hand"][0] + 0.5, h["hand"][1] + 0.5)
                    sc = sprite_scale.get(w["carry"], s_gear)
                    if "grip_frac" in w:  # a polearm: held on its shaft, head up and forward; the head is in front of it
                        ang = carry_ang(w["kind"], aid, len(frames))
                        f = place(w["carry"], float(w["grip_frac"]), ang, target, sc)
                        f[head_mask[aid][len(frames)]] = -1
                    else:
                        ang = blade_angle(w["carry"], "grip", carry_ang(w["kind"], aid, len(frames)), target, sc)
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
                turns = cf.get("turn", {}).get(aid)  # degrees clockwise per frame (a fall: the bow lies down)
                for fi, h in enumerate(hs):
                    if turns and turns[fi]:
                        canvas_t = np.full((fh, fw), -1, dtype=np.int32)
                        gx_, gy_ = cf["grip"]
                        paste_turned(canvas_t, src, 0, 0, (gx_, gy_), turns[fi], (h["hand"][0], h["hand"][1]))
                        frames.append(canvas_t)
                        moved_tips.append(None)
                        continue
                    ddx, ddy = int(round(h["hand"][0] - cf["grip"][0])), int(round(h["hand"][1] - cf["grip"][1]))
                    f = shift(src, ddx, ddy)
                    moved_tips.append(tuple((x + ddx, y + ddy) for x, y in src_tips) if src_tips else None)
                    f[head_mask[aid][fi]] = -1  # the head is in front of a bow held at the side
                    frames.append(f)
                put("weapon", wid, aid, frames)
                weapon_tips[(wid, aid)] = moved_tips
        # its own sprite in the attack of the weapon it replaces, posed like that weapon frame by frame
        if "attack_sprite" in w and sprites_hi:
            for a in anims:
                if a.get("weapon") != w["pose_from"]:
                    continue
                frames, poses = posed_frames(w, w["attack_sprite"], w["pose_sprite"], a,
                                             layers[("weapon", a["weapon"])][a["id"]])
                put("weapon", wid, a["id"], frames)
                print(f"weapon {wid} in {a['id']}: poses from {w['pose_from']} {poses}")
        # `resprite` on an animation: GPT drew its weapon too thin or too dark to read at game size, so the weapon's own
        # sheet sprite takes its place, posed like GPT's weapon frame by frame (before other weapons pose from it)
        if w.get("carry") in sprites_hi and "recolor_of" not in w:
            for a in anims:
                if a.get("weapon") != wid or not a.get("resprite") or a["id"] not in layers.get(("weapon", wid), {}):
                    continue
                gpt = layers[("weapon", wid)][a["id"]]
                frames, poses = posed_frames(w, w["carry"], w["carry"], a, gpt)
                for fi, f in enumerate(frames):  # where GPT's blade was cut from the body and this one does not lie
                    body[a["id"]][fi] = close_gaps(body[a["id"]][fi], f, dilate(gpt[fi] >= 0))
                put("weapon", wid, a["id"], frames)
                print(f"weapon {wid} in {a['id']}: own sprite posed {poses}")

    # shields: held in guard at the front of the chest (near shoulder + a per-animation offset), so a shield does not
    # swing with GPT's far hand from front to back. Seen from the side its face is turned toward the enemy: the
    # face-on sheet sprite is squashed across (`shield.squash`), so it does not hide half of the body. The far arm,
    # which would show outside the torso, is cut from the layers below (masks/far_arm); the cut edge of the back
    # gets an outline that travels with the shield.
    sh_cfg = cfg.get("shield", {})
    # isometric views: the shield is strapped to the hand without the weapon (`hands` far point) plus a per-animation
    # offset, seen at an angle (`squash`); from behind (`back`) it is its back side, drawn behind the body (slot order)
    # `hand_anims`: the side view straps it to the hand in these animations only (a fall)
    hand_anims = set(hands) if sh_cfg.get("mode") == "hand" else set(sh_cfg.get("hand_anims", []))
    if hand_anims and gear_arr is not None:
        for nm in [x for x in gear["cells"] if x.lstrip("~").startswith("shield:")]:
            sid = nm.lstrip("~").partition(":")[2]
            src = f"sprite:shield_back_{sid}" if sh_cfg.get("back") else nm
            if src not in sprites_hi:
                continue
            hi = sprites_hi[src]
            sq = sh_cfg.get("squash", 1.0)
            if sq != 1.0:
                sprites_hi[src + "@iso"] = np.asarray(Image.fromarray(hi, "RGBA").resize(
                    (max(1, int(round(hi.shape[1] * sq))), hi.shape[0]), Image.Resampling.NEAREST))
                src = src + "@iso"
            sc = s_shield if not sh_cfg.get("back") else sh_cfg.get("height", gear["shield_height"]) / hi.shape[0]
            for aid, hs in hands.items():
                if aid not in hand_anims:
                    continue
                off = sh_cfg.get("offset", {}).get(aid, [0, -3])
                # one [dx, dy] for the animation, or one per frame
                offs = off if isinstance(off[0], list) else [off] * len(hs)
                fx = np.array([h["far"][0] for h in hs], float)
                fy = np.array([h["far"][1] for h in hs], float)
                if aid in sh_cfg.get("follow", []):  # the hand goes far (a fall): the shield goes with it
                    cx, cy = fx, fy
                else:
                    # the arm swings the shield only a little (a share of the hand's swing, clamped) and the body's
                    # bob carries it: following GPT's hand fully made it jump across the body
                    k, mx = sh_cfg.get("sway", [0.3, 2.0])
                    hy = np.array([head_pos[aid][i][1] for i in range(len(hs))], float)
                    cx = fx.mean() + np.clip(k * (fx - fx.mean()), -mx, mx)
                    cy = fy.mean() + (hy - hy.mean()) + np.clip(k * (fy - fy.mean()), -mx, mx)
                # `angle` {anim: per frame}: null = upright, else where the shield's top points (0 = right, 90 = down)
                angs = sh_cfg.get("angle", {}).get(aid, [None] * len(hs))
                frames = [place(src, "centre", angs[i], (cx[i] + offs[i][0] + 0.5, cy[i] + offs[i][1] + 0.5), sc)
                          for i in range(len(hs))]
                put("shield", sid, aid, frames)
        if sh_cfg.get("mode") == "hand":
            sh_cfg = {}
    for nm in [x for x in (gear["cells"] if gear_arr is not None and sh_cfg else []) if x.startswith("shield:")]:
        sid = nm.partition(":")[2]
        src = nm
        if sh_cfg.get("squash", 1.0) != 1.0:
            hi = sprites_hi[nm]
            w_new = max(1, int(round(hi.shape[1] * sh_cfg["squash"])))
            src = nm + "@side"
            sprites_hi[src] = np.asarray(Image.fromarray(hi, "RGBA").resize((w_new, hi.shape[0]), Image.Resampling.NEAREST))
        for aid, hs in hands.items():
            if aid in hand_anims:
                continue
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
            if aid in layers.get(("weapon", wid), {}):  # its own sprite already (carried): only the rest is recoloured
                continue
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
        # shoe-brown, or a colour that stands at the mannequin's feet (front / back views: a palette can merge the
        # shoe browns away from the brown rule); small blobs count there, a foot seen end-on is only a few pixels
        tab = shoe_tab | (shoe if cfg.get("view") == "isometric" else np.zeros_like(shoe))
        min_sz = 4 if cfg.get("view") == "isometric" else 6
        lab, sizes = label(col.of(img, tab))
        up_ = height_along(aid, fi)  # a lying body: its belt is near the ground too, but not near the soles
        shoes = np.isin(lab, [i for i, sz in enumerate(sizes)
                              if sz >= min_sz and up_[lab == i].min() <= 14])  # not the belt
        if not shoes.any() and cfg.get("view") == "isometric":
            # a foot seen from behind can be two pixels: the bottom rows of the figure stand for the shoes
            yb = int(np.flatnonzero(op.any(1))[-1])
            shoes = op & (yy_g >= yb - 2) & ~col.of(img, col.linen)
        lab, sizes = label(col.of(img, hand_tab))
        hands_px = np.isin(lab, [i for i, sz in enumerate(sizes) if sz >= 4])
        legs = op & ~shoes & ~col.of(img, dark_tab) & ~hands_px & below_waist(aid, fi)
        for poly in cfg.get("leg_exclude", {}).get(aid, {}).get(str(fi), []):  # an arm hanging below the waist
            legs &= ~polygon_mask(poly, (fh, fw))
        if aid in hands:
            legs &= ~all_arms[aid][fi] & ~far_cut[aid][fi]
        return legs, shoes

    # arms for bracers: near arm (shoulder, elbow, hand) and the far arm where it shows. From `arms` (both arms
    # annotated, both whole; the second may be null) or from `hands` (far shoulder next to the near one, far elbow on
    # the shoulder-hand line, the far arm only where it swings out behind the back)
    def arm_list(aid: str, fi: int) -> list[tuple]:
        out = []
        if aid in hands and aid not in cfg.get("arms", {}):
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

    # rigid pieces: where GPT redraws a piece a little elsewhere in every frame (a front-view walk: registration of a
    # symmetric body has nothing to lock on), one good frame of it is used for the whole animation, moved with the
    # head (the torso bobs with it), clipped to the body (1 px rim) and kept behind the forearms. `rigid`:
    # {anim: [reference frame, dx, dy(, "complete")(, "behind")]} on the kit item
    for kit in cfg["kits"]:
        for it in kit["items"]:
            for aid, (ref_i, rdx, rdy, *opt) in it.get("rigid", {}).items():
                frames = layers.get((it["slot"], it["id"]), {}).get(aid)
                if not frames:
                    continue
                ref = shift(frames[ref_i], rdx, rdy)
                rx, ry = head_pos[aid][ref_i]
                # "complete": where an arm hid the piece in the reference frame, the other frames (moved to it) fill
                # it in (3/4 views, where the near arm crosses the torso)
                for fj in sorted(range(len(frames)), key=lambda j: -int((frames[j] >= 0).sum())):
                    if fj != ref_i and "complete" in opt:
                        hx, hy = head_pos[aid][fj]
                        g = shift(frames[fj], rx - hx + rdx, ry - hy + rdy)
                        ref = np.where((ref < 0) & (g >= 0), g, ref)
                for fi in range(len(frames)):
                    hx, hy = head_pos[aid][fi]
                    f = shift(ref, hx - rx, hy - ry)
                    op = body[aid][fi] >= 0
                    keep = (f >= 0) & dilate(op)
                    if "behind" in opt:  # a cape seen from the front: on the body only over the shoulders (the drape)
                        top_b = int(np.flatnonzero(op.any(1))[0])
                        keep = (f >= 0) & (~op | (yy_g < top_b + 4))
                    # a forearm swung in front of the torso hides the piece (the upper arm lies on the shoulder,
                    # which the piece covers)
                    for S_, E_, H_, _ in arm_list(aid, fi):
                        fore = seg_dist(xx_g, yy_g, E_, H_) <= 1.8
                        keep &= ~(fore & op) & ~fist(body[aid][fi], H_)
                    frames[fi] = np.where(keep, f, -1)

    # recoloured items: the same shape in other cloth colours (`recolors`: {slot, id, of, ramp light -> dark}). The source
    # item's lit pixels take the ramp by lightness (pixel-weighted quantiles); outline and dark detail stay
    for rc in cfg.get("recolors", []):
        src = layers.get((rc["slot"], rc["of"]))
        if not src:
            continue
        ramp = np.array([hex_rgb(h) for h in rc["ramp"]], dtype=np.uint8)
        r0 = len(palette)
        palette = np.concatenate([palette, ramp])
        val = palette.astype(float).max(1) / 255.0
        # the outline (navy, or nearly black) stays; the item's own deep shades are recoloured too
        hue_ = col.hsv[:, 0] * 360
        keep_ = col.dark & (((hue_ >= 190) & (hue_ <= 280)) | (col.hsv[:, 2] < 0.16))
        lit_vals = np.concatenate([val[f[(f >= 0) & ~keep_[np.maximum(f, 0)]]] for fr in src.values() for f in fr])
        q = np.quantile(lit_vals, np.linspace(0, 1, len(ramp) + 1)[1:-1]) if len(lit_vals) else []
        mapping = np.arange(len(palette))
        for i in range(n):
            if not keep_[i]:
                mapping[i] = r0 + len(ramp) - 1 - int(np.searchsorted(q, val[i]))
        rec = lambda fr: [np.where(f >= 0, mapping[np.maximum(f, 0)], -1) for f in fr]
        only = rc.get("anims")  # e.g. the side view: newer animations recolour, older ones have their own kit
        for aid, fr in src.items():
            if only is None or aid in only:
                put(rc["slot"], rc["id"], aid, rec(fr))
        if (rc["slot"], rc["of"]) in under:
            under.setdefault((rc["slot"], rc["id"]), {}).update(
                {aid: rec(fr) for aid, fr in under[(rc["slot"], rc["of"])].items() if only is None or aid in only})
        print(f"recolor {rc['slot']}:{rc['of']} -> {rc['id']}")

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
        # tone edges from the palette animations' pixels (a new animation does not move them), dark to light
        idx_q = np.concatenate([f[f >= 0] for aid, frames in per_anim.items() if aid in pal_anims for f in frames] or [idx])
        idx_q = idx_q[fam[idx_q]] if len(idx_q) else idx
        q = np.quantile(value_of[idx_q if len(idx_q) else idx], [0.12, 0.35, 0.65, 0.88])
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
    # an animation's `role`: none = the attack of its weapon's kind; "block" = that kind's parry, "block_shield" = the
    # parry with a shield (any one-handed weapon), "hurt", "death"
    attack_of = {weapons[a["weapon"]]["kind"]: a["id"] for a in anims if a.get("weapon") and not a.get("role")}
    block_of = {weapons[a["weapon"]]["kind"]: a["id"] for a in anims if a.get("weapon") and a.get("role") == "block"}
    slots_out = []
    for slot in cfg["slots"]:
        items = []
        if slot["id"] == "body":
            items.append({"id": "base", "label": labels_cfg.get("body:base", "Mannequin"),
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
                entry.update({"kind": w["kind"], "hands": w["hands"], "attack": w.get("attack") or attack_of.get(w["kind"]),
                              "block": w.get("block", block_of.get(w["kind"]))})
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
        **({"label": cfg["label"]} if "label" in cfg else {}),
        "frame_size": [fw, fh],
        "pivot": {"x": cfg["anchor_x"], "y": cfg["ground_y"] + 1},
        "facing": cfg.get("facing", "right"),
        "view": cfg.get("view", "side"),
        "compose": "First draw the `under` strips of the worn items, then the slots in order. When a shield is worn: cut masks.far_arm out of every slot listed in "
                   "the shield slot's `cuts` (the far arm is behind the shield), and right after the shield draw "
                   "every earlier slot again clipped to masks.near_arm (the sword arm is in front of the shield). "
                   "When headgear is worn, cut its `hides` mask out of the head layer (hair that would poke "
                   "through).",
        "animations": [{**{k: a[k] for k in ("id", "label", "frames", "fps", "loop", "role") if k in a},
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
