#!/usr/bin/env python3
"""Rough arm annotations for front / back / three-quarter views, from the built body strips.

Finds the hands (skin blobs below the shoulders), then follows each arm up to the shoulder row: per row, the run of
non-outline pixels around the arm's centre line, as long as it is arm-wide (a wider run is the torso the arm lies
on, the line then goes straight up). Prints `hands` (near arm + far hand) and `arms` (both arms) for the config.
Check the result on a grid (grid_view.py --overlay arms) and fix single points by hand.

Usage: python3 tools/auto_arms.py assets/doll_s --near left [--anims idle,walk] [--shoulder-drop 3]
  --near   which hand holds the weapon: left / right side of the image (front view: the figure's right hand is on
           the left of the image), or near = the hand lower in the image
"""
from __future__ import annotations

import argparse
import colorsys
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def frames_of(path: Path, fw: int) -> list[np.ndarray]:
    arr = np.asarray(Image.open(path).convert("RGBA"))
    return [arr[:, i * fw:(i + 1) * fw] for i in range(arr.shape[1] // fw)]


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        comp, q = [], deque([(y, x)])
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(comp)
    return out


def annotate(img: np.ndarray, near: str, shoulder_drop: int, max_run: int) -> tuple[list, list]:
    rgb, op = img[..., :3].astype(float) / 255, img[..., 3] >= 128
    hsv = np.array([[colorsys.rgb_to_hsv(*p) for p in row] for row in rgb])
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    dark = op & (val < 0.3)
    skin = op & (hue >= 12) & (hue <= 31) & (sat >= 0.35) & (sat <= 0.75) & (val >= 0.5)  # linen is <= 0.3
    rows = np.flatnonzero(op.any(1))
    top, bottom = int(rows[0]), int(rows[-1])
    sh_y = top + shoulder_drop
    hands = []
    for comp in components(skin):
        ys, xs = np.array(comp).T
        # hands hang between the shoulders and the knees; skin-hued shading lower down is the legs
        if len(comp) >= 3 and sh_y + 6 < ys.mean() < top + 0.6 * (bottom - top):
            hands.append((float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, len(comp)))
    hands = sorted(hands, key=lambda h: -h[2])[:2]
    hands.sort(key=lambda h: h[0])
    arms = []
    for hx, hy, _ in hands:
        x = hx
        line = {}
        for y in range(int(hy) - 2, sh_y - 1, -1):
            xi = int(round(x - 0.5))
            ok = op[y] & ~dark[y]
            cand = [c for c in range(max(xi - 2, 0), min(xi + 3, img.shape[1])) if ok[c]]
            if not cand:
                line[y] = x
                continue
            c0 = min(cand, key=lambda c: abs(c - xi))
            l, r = c0, c0
            while l - 1 >= 0 and ok[l - 1]:
                l -= 1
            while r + 1 < img.shape[1] and ok[r + 1]:
                r += 1
            if r - l + 1 <= max_run:
                x = (l + r + 1) / 2
            line[y] = x
        sx = line.get(sh_y, x)
        ey = int(round((sh_y + hy) / 2))
        ex = line.get(ey, (sx + hx) / 2)
        arms.append([round(sx * 2) / 2, float(sh_y) + 0.5, round(ex * 2) / 2, ey + 0.5, round(hx * 2) / 2, round(hy * 2) / 2])
    if not arms:
        return [], []
    if near == "lower":
        ni = int(np.argmax([a[5] for a in arms]))
    else:
        ni = 0 if near == "left" else len(arms) - 1
    n_arm = arms[ni]
    far = arms[1 - ni] if len(arms) == 2 else None
    hands_row = n_arm + ([far[4], far[5]] if far else [n_arm[4], n_arm[5]])
    arms_row = n_arm + [far]
    return hands_row, arms_row


def annotate_bent(img: np.ndarray, head_cx: float, half_w: float, near: str, shoulder_drop: int,
                  head_bottom: int | None = None) -> tuple[list, list]:
    """Arms bent at the elbow (a run): shoulders at the head's centre +- `half_w` (measured in the walk), hands are the
    skin blobs on each side, the elbow is where that side of the silhouette sticks out furthest between them. A hand
    that is hidden ends the arm at its elbow."""
    rgb, op = img[..., :3].astype(float) / 255, img[..., 3] >= 128
    hsv = np.array([[colorsys.rgb_to_hsv(*p) for p in row] for row in rgb])
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    skin = op & (hue >= 12) & (hue <= 31) & (sat >= 0.35) & (sat <= 0.75) & (val >= 0.5)
    rows = np.flatnonzero(op.any(1))
    top, bottom = int(rows[0]), int(rows[-1])
    # the shoulders hang from the collar (the bottom of the head sprite), not from the top of the figure: a raised
    # hand is the top of the figure then
    sh_y = head_bottom + 3 if head_bottom is not None else top + shoulder_drop
    blobs = []
    for comp in components(skin):
        ys, xs = np.array(comp).T
        neck = abs(xs.mean() + 0.5 - head_cx) < 4 and sh_y - 5 <= ys.mean() < sh_y + 2  # left of the mannequin's neck
        if len(comp) >= 2 and ys.mean() < top + 0.62 * (bottom - top) and not neck:  # raised hands count too
            blobs.append((float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, len(comp)))
    arms = []
    for side in (-1, 1):  # left of the image, right of the image
        S = (head_cx + side * half_w, sh_y + 0.5)
        mine = [b for b in blobs if (b[0] - head_cx) * side > -1.5]
        H = max(mine, key=lambda b: b[2])[:2] if mine else None
        y_end = int(H[1]) if H else int(sh_y + 0.3 * (bottom - top))
        best = None
        y_a, y_b = sorted((int(sh_y) + 2, y_end))  # a raised hand: the elbow is between it and the shoulder too
        for y in range(y_a, max(y_b, y_a + 1)):
            xs = np.flatnonzero(op[y])
            if not len(xs):
                continue
            x_out = xs.min() if side < 0 else xs.max()
            if best is None or (x_out - best[0]) * side > 0:
                best = (float(x_out), y)
        E = (best[0] - side * 1.5 + 0.5, best[1] + 0.5) if best else (S[0], S[1] + 5)
        if H is None:
            H = (E[0], E[1] + 3)
        arms.append([round(S[0] * 2) / 2, S[1], round(E[0] * 2) / 2, E[1], round(H[0] * 2) / 2, round(H[1] * 2) / 2])
    left, right = arms
    n_arm, f_arm = (left, right) if near == "left" else (right, left)
    return n_arm + f_arm[4:6], n_arm + [f_arm]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doll", help="built doll folder, e.g. assets/doll_s")
    ap.add_argument("--near", choices=["left", "right", "lower"], required=True)
    ap.add_argument("--anims", default="idle,walk")
    ap.add_argument("--shoulder-drop", type=int, default=3, help="shoulder row below the top of the headless body")
    ap.add_argument("--max-run", type=int, default=6, help="widest run that is still an arm")
    ap.add_argument("--bent", action="store_true", help="arms bent at the elbow (a run): see annotate_bent")
    a = ap.parse_args()
    doll = Path(a.doll)
    meta = json.loads((doll / "doll.json").read_text())
    fw = meta["frame_size"][0]
    hands, arms = {}, {}
    for anim in a.anims.split(","):
        hands[anim], arms[anim] = [], []
        heads = frames_of(doll / "layers" / "head" / "a" / f"{anim}.png", fw) if a.bent else None
        walk = frames_of(doll / "body" / "walk.png", fw) if a.bent else None
        if a.bent:  # shoulder half-width: half the torso width a little below the collar, in the walk
            ws = []
            for img in walk:
                op = img[..., 3] >= 128
                top = int(np.flatnonzero(op.any(1))[0])
                xs = np.flatnonzero(op[top + 2])
                ws.append((xs.max() - xs.min()) / 2 - 1.5 if len(xs) else 6)
            half_w = float(np.median(ws))
        for fi, img in enumerate(frames_of(doll / "body" / f"{anim}.png", fw)):
            if a.bent:
                hy, hx = np.nonzero(heads[fi][..., 3] >= 128)
                h, r = annotate_bent(img, float(hx.mean()) + 0.5, half_w, a.near, a.shoulder_drop, int(hy.max()))
            else:
                h, r = annotate(img, a.near, a.shoulder_drop, a.max_run)
            hands[anim].append(h)
            arms[anim].append(r)
    print(json.dumps({"hands": hands, "arms": arms}))


if __name__ == "__main__":
    main()
