# Paper doll: art direction and layer contract

## Game frame

- 2D side-scrolling action game, side view only, character faces RIGHT (mirror for left)
- One base body ("mannequin") for every animation; everything else is a separate layer on top

## Technical contract

- Frame 112x96 for every layer and animation (the greatsword thrust needs room in front)
- Pivot (48, 92): feet stand on row 91
- Mannequin ~48 px tall at idle; heads are rigid sprites ~15 px, all at one scale
- Animations: idle 4 f @ 6 fps, walk 6 @ 10, run 6 @ 12,
  attack_sword1h 6 @ 12, attack_sword2h 6 @ 10, attack_bow 6 @ 10 (attacks do not loop)
- Draw order (back to front): body, pants, cloak, greaves, skirt, chest, bracers, shoulders, shield, head, headgear, weapon
- Each layer contains only the pixels it shows over the bare body, so any combination stacks
- Shield: only together with a one-handed weapon. It is held in guard at the front of the chest (near shoulder + a
  per-animation offset, `shield.guard`), so it never swings behind the back with the far hand, and it is seen at
  an angle (the face-on sheet sprite squashed across, `shield.squash` 0.45), so it does not hide the body. It
  follows the torso's bob and sways (and tilts) with a clamped, smoothed share of the far hand's swing
  (`shield.sway`), so the shield arm looks alive without going behind the back. While it is worn,
  cut `masks/far_arm/<anim>.png` out of the slots in the shield slot's `cuts` (body, bracers, chest, shoulders: the
  far arm is behind the shield), and right after the shield draw every earlier layer again clipped to
  `masks/near_arm/<anim>.png` (the sword arm is in front of the shield)
- Pants and steel greaves are not cut from GPT redraws: they are the mannequin's own legs recoloured (`derived`):
  pants = cloth below the waist line (`waist`), greaves = the leg from the shoe up 8 px along the leg. GPT redraws
  every kit frame with the legs a little elsewhere, so leg armour cut from a kit never sat on the body's legs.
  Where the back shin passes behind the front leg, `shin_exclude` closes the front leg above its own greave
- Bracers are derived the same way from the sleeves: from the wrist up a share of the arm (`coverage` 0.25 / 0.5 /
  0.75 / 1.0) along the annotated arm (`hands`, or `arms` for the 2-handed and bow attacks). The far arm gets one
  only where it shows behind the back (the `far_arm_cut` regions). Every derived piece is drawn in a 4-tone ramp,
  so another material is another ramp
- Cloaks: gaps between an arm and the torso show the cape's shadow tone; where GPT drew the far arm over the cape,
  the cape is continued under it in `layers/cloak/<id>/<anim>_under.png`, drawn before the body (visible only when
  a shield cuts that arm away)
- Hand-held items in idle / walk / run are placed per frame at hand points annotated in `source/doll/doll.json`
  (`hands`: near-arm shoulder, elbow, hand + far hand): one-handed weapons only ever in the near hand
- Headgear: before drawing it, cut its `hides` mask (`masks/headgear/<id>/<anim>.png`) out of the head layer, so
  hair never pokes out above a helmet; all helmets are drawn on the bald head and fit every head
- Nearest filtering, real binary alpha, lossless PNG strips (`<frames> x 112` wide)

## Visual system

- 16-bit pixel art, 1 px dark navy outline (#1b1f33), flat cel shading, light from upper left
- Mannequin: bald, off-white linen undershirt and trousers, dark brown shoes, skin hands
- Every equipment piece has its own colour family so it can be cut out by colour:
  red / green capes, blue / purple skirts, steel / leather greaves and bracers,
  steel breastplate, bronze pauldrons, steel swords with gold or steel guards, wooden bow

## Items

| Slot | Items |
|---|---|
| head (required) | A: short dark hair · B: blond, ponytail, beard · C: auburn hair in a bun |
| headgear | closed knight helmet · archer helmet (open-face cap) · conical helmet (cut again from its sheet 5 % bigger than the bald skull it was drawn on, centred on the head, so it covers the forehead; outline restored on its lit slope; nothing of it on the face: `headgear_fit`, with pixel touch-ups `erase` / `paint`) |
| chest | plate cuirass · riveted cuirass with seams · chain-mail hauberk to mid-thigh · brigandine |
| shoulders | banded pauldrons · domed pauldrons · large three-plate pauldrons · small round spaulders |
| pants | green wool · blue · dark leather |
| greaves | plate (derived) · tall boot shafts · cuffed greaves |
| bracers (by shape: share of the arm from the wrist) | 25 % wrist cuff · 50 % to the elbow · 75 % with elbow cop · 100 % whole arm |
| cloak | red · green · blue |
| skirt | blue · purple · red |
| shield (one-handed weapon only) | round wooden · heater (crimson, gold band) |
| weapon | one-handed: sword, falchion (sheet sprite mirrored, so the blade curves up: `mirror`) · two-handed: sword, zweihander · polearms (two-handed): glaive, halberd, spear · bow, red lacquered bow |

Each weapon kind has one attack: `attack_sword1h`, `attack_sword2h`, `attack_bow`. The first weapon of a kind is
cut out of the attack drawing; the second is its own sprite matched to that weapon's pose in every frame. The red
bow is built on the wooden bow's animated limbs (so grip, flex and string match): red lacquer, gold recurve tips
that curl forward, a gold grip wrap (`lacquer` in the config).

## Materials

Armour is chosen by shape, and its material by colour, separately for every piece. Chest, shoulders, bracers,
greaves and headgear have a tone map per animation (`layers/<slot>/<shape>/<anim>_tones.png`): the piece's own
pixels (its colour family as drawn, or its derived ramp) graded by lightness into 5 tones, grey 0/60/120/180/240 =
highlight .. deepest; outline and details of other colours are not in it. Any colour recolours the piece: tone 2 is
that colour, tones 0-1 lighter and a little warmer, 3-4 darker and a little cooler (`ramp_from` in
tools/doll_preview.py, `rampFrom` in server/public/app.js). Ready materials (`materials` in doll.json: steel,
silver, gold, copper, bronze, leather) carry hand-tuned ramps; a slot's `materials` lists those it offers. In a
game this is a palette swap: one strip, one tone map, any number of materials.

## Polearms

Glaive, halberd and spear come from `source/doll/raw/polearm_sheet.png` (drawn upright, one scale: `length` 50 px).
Carried in the near hand on the shaft (`grip_frac` 0.6 from the top), head up and forward (`carry_angle.polearm`),
behind the head where they cross it; in the two-handed attack they take the two-handed sword's pose frame by frame,
the grip in the fists, the head kept above the ground. Two-handed, so no shield.

## Approved references

- Body: `source/doll/seed_approved.png` (v1 rejected: head 1/7 of height, unreadable at 48 px)
- Heads: `source/doll/raw/heads_sheet.png`, `source/doll/raw/heads_sheet2.png`; helmets:
  `source/doll/raw/headgear_sheet.png` (drawn on the bald head). Every sheet is scaled and placed through its own
  bald head, using the common neck line and the face profile, so all heads share one size and position
- Swords and shields: `source/doll/raw/gear_sheet.png` (upright, one shared scale)
- Two-handed attack: v2 (v1 raised the hands above the head, which read as a second head)
- Approval: delegated to Claude by the user, 2026-11-16
