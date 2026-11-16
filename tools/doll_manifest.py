#!/usr/bin/env python3
"""Write assets/asset-manifest.json: one entry per layer strip with its source images and prompt.

Usage: python3 tools/doll_manifest.py source/doll/doll.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = "OpenAI image generation via Codex CLI 0.161.0 (driver model gpt-6.1-sol), ChatGPT subscription; tools/doll.py"


def main(config: str) -> None:
    cfg = json.loads((ROOT / config).read_text())
    out_dir = ROOT / cfg["out_dir"]
    doll = json.loads((out_dir / "doll.json").read_text())
    anims = {a["id"]: a for a in cfg["animations"]}

    heads = cfg["heads"] if isinstance(cfg["heads"], list) else [cfg["heads"]]
    weapons = cfg["weapons"]
    gear = cfg.get("gear", {}).get("src")
    sheet_of = {cell: g["src"] for g in cfg.get("gear_sheets", []) for cell in g["cells"]}  # polearms etc.

    derived = {(d["slot"], d["id"]): d for d in cfg.get("derived", [])}

    def sources(slot: str, item: str, anim: str) -> list[str]:
        if slot == "body" or (slot, item) in derived:
            return [anims[anim]["src"]]
        if slot in ("head", "headgear"):
            return [h["src"] for h in heads if f"{slot}:{item}" in h["cells"]][:1]
        if slot == "shield":
            return [gear]
        if slot == "weapon":
            w = weapons[item]
            own = sheet_of.get(w.get("carry", ""), gear)
            if "recolor_of" in w:
                return sources(slot, w["recolor_of"], anim)
            if anims[anim].get("weapon") == item:
                return [anims[anim]["src"]]
            if anims[anim].get("weapon"):  # another weapon's attack: own sprite posed like that weapon
                return [own, anims[anim]["src"]]
            if "carry_from" in w:
                return [anims[w["carry_from"]["anim"]]["src"]]
            return [own]
        for kit in cfg["kits"]:
            if any(i["slot"] == slot and i["id"] == item for i in kit["items"]):
                return [kit["src"].format(anim=anim), anims[anim]["src"]]
        return []

    def prompt_of(raw: str) -> str:
        stem = Path(raw).stem
        if stem.startswith("kit"):
            stem = stem.split("_", 1)[0]
        return f"source/doll/prompts/{stem}.txt"

    entries = []
    for slot in doll["slots"]:
        for item in slot["items"]:
            versions = [(None, item["strips"])] + ([("tones", item["tones"])] if item.get("tones") else [])
            for mat, strips in versions:
                for anim, rel in strips.items():
                    srcs = sources(slot["id"], item["id"], anim)
                    prompts = [prompt_of(x) for x in srcs]
                    entries.append({
                        "id": f"{slot['id']}-{item['id']}{'-' + mat if mat else ''}-{anim}",
                        "role": f"paper-doll layer: {slot['label']} / {item['label']} / {anim}",
                        "path": f"{cfg['out_dir']}/{rel}",
                        "status": "test",
                        "technical": {"frame": doll["frame_size"], "frames": anims[anim]["frames"], "fps": anims[anim]["fps"],
                                      "pivot": doll["pivot"], "alpha": True, "filtering": "nearest", "draw_order": slot["id"]},
                        "source": {
                            "kind": "ai-generated, then deterministically sliced / registered / cut into layers"
                                    + ("; hand-held: placed per frame at hand points annotated in source/doll/doll.json"
                                       if slot["id"] in ("weapon", "shield") else "")
                                    + (f"; derived: the mannequin's own {derived[(slot['id'], item['id'])]['from']} "
                                       "recoloured (no separate generation)" if (slot["id"], item["id"]) in derived else "")
                                    + (f"; derived: the {weapons[item['id']]['recolor_of']} limbs repainted "
                                       + ("(lacquer, gold recurve tips, grip wrap)" if "lacquer" in weapons[item["id"]] else "(recolour)")
                                       if slot["id"] == "weapon" and "recolor_of" in weapons.get(item["id"], {}) else "")
                                    + ("; tone map (data): the piece's own pixels by lightness, grey 0..240 = tone 0..4, "
                                     "to recolour it into any material" if mat == "tones" else ""),
                            "url_or_tool": TOOL,
                            "created_at": "2026-11-16",
                            "license": "check OpenAI terms for generated images before commercial release",
                            "prompt_or_source_record": {"raw": srcs, "prompts": prompts,
                                                        "reference": "source/doll/seed_approved.png"},
                        },
                        "verification": {"native_scale_reviewed": True, "in_engine_reviewed": False,
                                         "technical_checks_passed": True, "approved_by": ""},
                    })
    (ROOT / "assets" / "asset-manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "art_direction_brief": "docs/doll-brief.md",
        "steam_ai_disclosure": "all doll layers are pre-generated AI content made during development",
        "assets": entries,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"assets/asset-manifest.json: {len(entries)} entries")


if __name__ == "__main__":
    main(sys.argv[1])
