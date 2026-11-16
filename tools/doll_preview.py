#!/usr/bin/env python3
"""Contact sheets for the paper doll: one PNG per animation, rows = outfits, columns = frames.

Usage:
  python3 tools/doll_preview.py [assets/doll] [scale=3]
      writes <doll>/_preview/<anim>.png. Rows: body only, each item alone on the body, then full outfits.
  python3 tools/doll_preview.py assets/doll SCALE --outfit '{"head":"a","weapon":"sword1h"}' --anims idle,walk --out x.png
      one sheet for one outfit (rows = animations).

Composition follows doll.json "compose": slots in order; when a shield is worn, right after the shield every
earlier slot is drawn again clipped to the near-arm mask, so the sword arm stays in front of the shield.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import colorsys

import numpy as np
from PIL import Image, ImageChops, ImageDraw

OUTFITS = {  # `shape@material` picks a material version
    "knight": {"head": "a", "headgear": "knight", "chest": "plate", "shoulders": "bands", "greaves": "plate",
               "bracers": "full", "cloak": "red", "skirt": "blue", "shield": "heater", "weapon": "sword1h"},
    "paladin": {"head": "b", "headgear": "knight@gold", "chest": "plate@gold", "shoulders": "layered",
                "greaves": "plate@gold", "bracers": "full@gold", "cloak": "blue", "skirt": "red", "shield": "heater",
                "weapon": "sword1h"},
    "archer": {"head": "b", "headgear": "archer", "pants": "green", "chest": "riveted", "greaves": "boots",
               "bracers": "forearm@leather", "cloak": "green", "weapon": "bow"},
    "merc": {"head": "b", "pants": "leather", "chest": "mail", "shoulders": "layered@steel", "greaves": "plate",
             "bracers": "elbow", "weapon": "sword2h"},
    "halberdier": {"head": "a", "headgear": "nasal@copper", "pants": "blue", "chest": "mail@copper",
                   "shoulders": "round@copper", "greaves": "cuffed", "bracers": "forearm@copper", "weapon": "halberd"},
    "guard": {"head": "c", "headgear": "nasal", "pants": "blue", "chest": "brigandine", "shoulders": "round",
              "greaves": "cuffed", "bracers": "forearm", "cloak": "blue", "shield": "round", "weapon": "falchion"},
    "hunter": {"head": "c", "headgear": "archer@silver", "pants": "green", "chest": "riveted", "greaves": "boots",
               "bracers": "wrist@silver", "cloak": "green", "weapon": "bow_dark"},
}


def _shift_hue(h: float, toward: float, deg: float) -> float:
    d = (toward - h + 540) % 360 - 180
    return (h + max(-deg, min(deg, d))) % 360


def ramp_from(hex_colour: str) -> list[tuple[int, int, int]]:
    """5 tones from one material colour (tone 2): lighter and a little warmer up, darker and a little cooler down.
    The viewer (server/public/app.js, rampFrom) uses the same formula."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h *= 360
    spec = [(60, 8, 0.6, lambda l_: l_ + (1 - l_) * 0.6), (60, 4, 0.85, lambda l_: l_ + (1 - l_) * 0.3),
            (0, 0, 1.0, lambda l_: l_), (240, 6, 1.05, lambda l_: l_ * 0.68), (240, 12, 1.1, lambda l_: l_ * 0.42)]
    out = []
    for toward, deg, ks, fl in spec:
        hh = _shift_hue(h, toward, deg) if deg else h
        rr, gg, bb = colorsys.hls_to_rgb(hh / 360, min(1.0, max(0.0, fl(l))), min(1.0, s * ks))
        out.append((round(rr * 255), round(gg * 255), round(bb * 255)))
    return out


def recolour(layer: Image.Image, tones: Image.Image, ramp) -> Image.Image:
    """The piece's own pixels (tone map: grey 0..240 in steps of 60) take the ramp's colours."""
    a = np.array(layer.convert("RGBA"))
    t = np.array(tones.convert("RGBA"))
    m = t[..., 3] > 0
    idx = np.clip(np.round(t[..., 0] / 60).astype(int), 0, 4)
    rgb = np.array(ramp, dtype=np.uint8)
    a[m, :3] = rgb[idx[m]]
    return Image.fromarray(a, "RGBA")


class Doll:
    def __init__(self, root: Path):
        self.root = root
        self.data = json.loads((root / "doll.json").read_text())
        self.fw, self.fh = self.data["frame_size"]
        self.slots = self.data["slots"]
        self.masks = self.data.get("masks", {}).get("near_arm", {})
        self.far = self.data.get("masks", {}).get("far_arm", {})
        self._strips: dict[str, Image.Image] = {}

    def strip(self, rel: str) -> Image.Image:
        if rel not in self._strips:
            self._strips[rel] = Image.open(self.root / rel).convert("RGBA")
        return self._strips[rel]

    def cell(self, rel: str, frame: int) -> Image.Image:
        return self.strip(rel).crop((frame * self.fw, 0, (frame + 1) * self.fw, self.fh))

    def item(self, slot: dict, outfit: dict):
        """The worn item; `shape@material` (a ready material, e.g. plate@gold) or `shape@#rrggbb` (any colour)
        recolours the piece's own pixels through its tone map."""
        want = "base" if slot["id"] == "body" else outfit.get(slot["id"])
        want, _, mat = (want or "").partition("@")
        it = next((i for i in slot["items"] if i["id"] == want), None)
        if it and mat and it.get("tones"):
            ramp = ramp_from(mat) if mat.startswith("#") else \
                [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in self.data["materials"][mat]["ramp"]]
            it = {**it, "_ramp": ramp}
        return it

    def layer(self, it: dict, anim: str, frame: int) -> Image.Image:
        img = self.cell(it["strips"][anim], frame)
        if it.get("_ramp") and anim in it.get("tones", {}):
            img = recolour(img, self.cell(it["tones"][anim], frame), it["_ramp"])
        return img

    def allowed(self, slot: dict, outfit: dict) -> bool:
        req = slot.get("requires")
        if not req:
            return True
        other = next((s for s in self.slots if s["id"] == req["slot"]), None)
        it = self.item(other, outfit) if other else None
        return bool(it) and it.get("hands") == req["hands"]

    def compose(self, anim: str, outfit: dict, frame: int) -> Image.Image:
        canvas = Image.new("RGBA", (self.fw, self.fh), (0, 0, 0, 0))
        under = Image.new("RGBA", (self.fw, self.fh), (0, 0, 0, 0))  # everything before the shield
        hg_slot = next((s for s in self.slots if s["id"] == "headgear"), None)
        headgear = self.item(hg_slot, outfit) if hg_slot else None
        sh_slot = next((s for s in self.slots if s["id"] == "shield"), None)
        shield_on = bool(sh_slot and self.item(sh_slot, outfit) and self.allowed(sh_slot, outfit)
                         and anim in self.item(sh_slot, outfit)["strips"])
        far = self.far.get(anim) if shield_on else None
        cuts = set(sh_slot.get("cuts", [])) if sh_slot else set()
        for slot in self.slots:  # `under` strips first (a cape continued under the far arm, seen when it is cut)
            it = self.item(slot, outfit)
            if it and self.allowed(slot, outfit) and anim in it.get("under", {}):
                canvas.alpha_composite(self.cell(it["under"][anim], frame))
        for slot in self.slots:
            it = self.item(slot, outfit)
            if it and not self.allowed(slot, outfit):
                it = None
            if it and anim in it["strips"]:
                layer = self.layer(it, anim, frame)
                if slot["id"] == "head" and headgear and anim in headgear.get("hides", {}):
                    hide = self.cell(headgear["hides"][anim], frame).getchannel("A")
                    layer.putalpha(ImageChops.subtract(layer.getchannel("A"), hide))
                if far and slot["id"] in cuts:  # the far arm is behind the shield
                    layer.putalpha(ImageChops.subtract(layer.getchannel("A"), self.cell(far, frame).getchannel("A")))
                canvas.alpha_composite(layer)
                if slot["id"] == "shield" and anim in self.masks:
                    mask = self.cell(self.masks[anim], frame).getchannel("A")
                    arm = under.copy()
                    arm.putalpha(ImageChops.multiply(arm.getchannel("A"), mask))
                    canvas.alpha_composite(arm)
                elif slot["id"] != "shield":
                    under.alpha_composite(layer)
        return canvas


def sheet(doll: Doll, rows: list[tuple[str, dict, str]], scale: int, n_frames, label_w: int = 150) -> Image.Image:
    fw, fh = doll.fw, doll.fh
    n = max(n_frames(anim) for _, _, anim in rows)
    img = Image.new("RGB", (label_w + n * fw * scale, len(rows) * fh * scale), (120, 132, 148))
    draw = ImageDraw.Draw(img)
    for r, (name, outfit, anim) in enumerate(rows):
        draw.text((6, r * fh * scale + 6), name, fill=(255, 255, 255))
        for f in range(n_frames(anim)):
            c = doll.compose(anim, outfit, f).resize((fw * scale, fh * scale), Image.Resampling.NEAREST)
            img.paste(c, (label_w + f * fw * scale, r * fh * scale), c)
        draw.line([(0, (r + 1) * fh * scale - 1), (img.width, (r + 1) * fh * scale - 1)], fill=(90, 100, 115))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doll", nargs="?", default="assets/doll")
    ap.add_argument("scale", nargs="?", type=int, default=3)
    ap.add_argument("--outfit", help="JSON {slot: item}; also accepts a preset name: " + ", ".join(OUTFITS))
    ap.add_argument("--anims", help="comma-separated animation ids (default: all)")
    ap.add_argument("--out", help="output PNG for --outfit")
    args = ap.parse_args()
    doll = Doll(Path(args.doll))
    frames = {a["id"]: a["frames"] for a in doll.data["animations"]}
    anims = args.anims.split(",") if args.anims else list(frames)

    if args.outfit:
        outfit = OUTFITS.get(args.outfit) or json.loads(args.outfit)
        out = Path(args.out or f"{args.doll}/_preview/outfit.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet(doll, [(a, outfit, a) for a in anims], args.scale, frames.get).save(out)
        print(f"wrote {out}")
        return

    one_hand = next((i["id"] for s in doll.slots if s["id"] == "weapon" for i in s["items"] if i.get("hands") == 1), None)
    base_rows = [("body", {"head": "a"})]
    for slot in doll.slots:
        if slot["id"] in ("body", "head"):
            continue
        for item in slot["items"]:
            outfit = {"head": "a", slot["id"]: item["id"]}
            if slot["id"] == "shield" and one_hand:
                outfit["weapon"] = one_hand
            base_rows.append((f"{slot['id']}:{item['id']}", outfit))
    base_rows += list(OUTFITS.items())
    out = Path(args.doll) / "_preview"
    out.mkdir(exist_ok=True)
    for anim in anims:
        rows = [(name, o, anim) for name, o in base_rows]
        sheet(doll, rows, args.scale, frames.get).save(out / f"{anim}.png")
        print(f"wrote {out / (anim + '.png')} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
