# test2d assets

Paper-doll pixel character for a 2D game, side view and 8-direction isometric: one mannequin body per drawn
direction, every piece of equipment on its own layer, plus a NestJS viewer to look at any combination.

## Look at it

```bash
cd server
cp .env.example .env   # first time only: PORT, HOST, ASSETS_DIR
npm install            # first time only
npm start              # builds with tsc, then serves http://HOST:PORT (default http://127.0.0.1:3000)
```

`server/.env` is not committed; variables set in the shell override it (`PORT=4000 npm start`).

The page has an equipment panel (presets: Mannequin, Knight, Paladin, Archer, Mercenary, Halberdier, Guard, Huntress,
Random; under every armour piece: ready materials, a colour picker for any material colour, and × for the piece
as drawn), a big
view with frame stepping and per-layer toggles, a scene (arrows / A D to walk, Shift to run, Space to attack with
the equipped weapon, hold F to block, H to take a hit, K to fall / get up again, Q to switch weapons) and every
animation in the current outfit. The tabs are Idle, Walk, Run, one "Attack" that plays the equipped weapon's
attack and switches with the weapon, one "Block", then Hurt and Death. Block follows the equipment: with a shield
the shield is raised (the sword held back), without one the weapon parries (one-handed: the blade up before the
face; two-handed swords and polearms: held across above the head); a bow has no block, and without a weapon
there is neither attack nor block. A shield can only be picked with a one-handed weapon.

## Eight directions

One doll per drawn direction; the other three are mirrors:

| Doll | Facing | Mirror gives | Animations and items |
|---|---|---|---|
| `doll` | east (side view) | west | everything above |
| `doll_se` | south-east | south-west | everything the side view has: idle, walk, run, the three attacks, the three blocks, hurt, death; every item, weapon and shield |
| `doll_s` | south (front) | — | the same |
| `doll_n` | north (back) | — | the same |
| `doll_ne` | north-east | north-west | the same |

The scene always walks in 8 directions, whichever doll the "Doll" menu shows (that menu picks the doll of the
big view and the animation cards). Click the scene: W A S D (or the arrows) walk in 8 directions. A or D alone walks with the side doll, S toward the camera, W away, S + A / S + D and W + A / W + D
along the diagonals. Space attacks, F (held) blocks, H plays the hit, K falls and stays down until K again, each in
the direction faced. Item ids are the same in every doll, so one outfit works in all directions.

What differs per view in `source/doll_<dir>/doll.json`:

- `facing` and `label` (the viewer picks the doll by `facing`);
- slot order: seen from behind (`doll_n`, `doll_ne`) the cloak is drawn over the chest and arms, and in
  `doll_n` the sword is drawn behind the body (the hand covers the grip; a hand swung forward hides it);
- kits: from behind, the cloak and the cuirass are separate kits, so the cuirass exists whole under the cloak;
- `head_rows` / `head_height`: front and back heads are found by their silhouette (a back of a head has no single
  skin tone) and sized like the other views (`head_nudge` [dx, dy] sets the head down by eye: `doll_ne` [-1, 2],
  else the bigger head read as a hunch); `head_fit` repaints pixels of a head sprite (`patch`: rows of
  characters in sprite coordinates), e.g. the front faces and head C's crown (its bun is behind the head);
- `hands` / `arms`: both arms annotated (`tools/auto_arms.py assets/doll_s --near left` gives a first guess,
  check it with an overlay grid and fix single points);
- `rigid` on a kit item: GPT redraws a piece a little elsewhere in every frame, and a symmetric front view gives
  the registration nothing to lock on, so one good frame is used for the whole walk, moved with the head
  (`doll_s` chest and cloak, `doll_ne` chest);
- `fixed_colors` include the hair colours of heads B and C, so the shared 48-colour palette keeps them.

## Hurt, block, death

Three sheets per view (2 rows x 3 cells each, `source/<doll>/raw/base_{hurt_block,blocks,death}.png` and their
kits), job lists `source/jobs/jobs_actions_base.txt` and `jobs_actions_kits.txt`:

| Sheet | Row 1 | Row 2 |
|---|---|---|
| `hurt_block` | `hurt`: recoil, stagger, recover (empty hands; weapons and shield are placed in the hands) | `block_shield`: the shield arm comes up, the sword held back |
| `blocks` | `block_sword1h`: the one-handed blade raised before the face | `block_sword2h`: the greatsword held across above the head (polearms use it too) |
| `death` | struck, knees buckle, kneeling | falling, lying, lying still |

What the pipeline does for them (keys in `source/<doll>/doll.json`):

- `sheet` {name, frames, rows, first}: an animation that is one row of a shared sheet; its kits are named after the
  sheet (`kit5_hurt_block.png`), so two animations cost one generation per kit. `scale_as`: the other animation of the
  same sheet (GPT draws a sheet at one scale); `height` / `height_frame` set the stature from a standing frame.
- `role`: `hurt`, `death`, `block_shield`, `block`. The viewer finds the animations by role, and every weapon gets
  `block` in doll.json (the parry of its kind; polearms name `block_sword2h`).
- `anchor: "cell"`: the fall keeps GPT's placement inside its cell (neither the head nor a foot stays put).
- `head_turn` {anim: {frame: degrees clockwise}}: the head of a falling or lying body is turned; the head template is
  turned for the search, and every head / helmet sprite and helmet mask is drawn turned (quarter turns exactly).
  Armour bands, the waist (pants) and the feet (greaves) are measured along the turned body.
- `resprite` on an animation: GPT drew its weapon too thin or too dark to read; the weapon's own sheet sprite is
  posed like it frame by frame. `weapon_cut` {anim: {frame: [[x0, y0, x1, y1]]}}: boxes whose outline pixels belong
  to a blade that downsampling thinned to its outline.
- `carry_angle[kind][anim]` may list one angle per frame (the weapon goes down with the hand); the side view's bow,
  cut from an attack frame, turns with `carry_from.turn`. `shield.follow` (the shield goes with the far hand all the
  way), `shield.angle` (per frame: where the shield's top points, 180 = lying, head side), `shield.offset` per frame,
  `shield.hand_anims` (the side view straps it to the hand in these animations).
- The side view uses five kits for these animations, grouped like the isometric ones, and `recolors` with `anims`
  for the green / blue cloaks and the purple skirt; its ten older kits keep `anims` of the first six animations.
- `palette_anims`: the animations whose sheets make the palette and the tone edges, so adding animations does not
  recolour the existing ones.

## Rebuild the layers

```bash
python3 tools/doll.py source/doll/doll.json        # raw strips -> assets/doll (layers + doll.json)
for d in se s n ne; do python3 tools/doll.py source/doll_$d/doll.json; done   # the isometric directions
python3 tools/doll_preview.py assets/doll 2         # contact sheets in assets/doll/_preview
python3 tools/doll_preview.py assets/doll 6 --outfit knight --anims walk --out /tmp/walk.png   # one outfit
python3 tools/doll_manifest.py source/doll/doll.json
```

Then press "Reload" on the page.

## Generate more

- `tools/gen_ref_image.sh OUT SIZE PROMPT_FILE REF...` — one image with reference images
- `tools/gen_batch.sh JOBS [PARALLEL]` — many at once; finished outputs are skipped
- Prompts: `source/doll/prompts/`, `source/doll_<dir>/prompts/`; job lists: `source/doll/jobs_*.txt`,
  `source/jobs/` (the isometric directions)

A new armour piece = a "kit": the mannequin strip redrawn with the piece in its own colour
family (see `kits` in `source/doll/doll.json`). The pipeline registers each kit frame onto the
body, keeps what changed, and assigns it to the piece by colour.

Pants, steel greaves and bracers are `derived` in the config: the body's own legs / sleeves recoloured with a
4-colour ramp, so they fit every frame. Bracers come by shape (`coverage` of the arm: 25 / 50 / 75 / 100 %). A new
material is one more `derived` entry with another ramp; no generation needed.

Materials: every armour piece can take any material colour. Its tone map (`<anim>_tones.png`) marks the piece's
own pixels by lightness; the viewer recolours them on the fly from a colour picker, or with a ready material's
ramp (steel, silver, gold, copper, bronze, leather). Presets write `shape@material` or `shape@#rrggbb`, e.g.
`plate@gold`, `mail@#2f8f5a`.

A new weapon or shield = a cell on a gear sheet (drawn upright) plus an entry in `weapons` / `gear`. In idle,
walk and run it is placed at the hand points in `hands` (near arm: shoulder, elbow, hand; far hand) at the
angle in `carry_angle`; in its kind's attack it is matched to the first weapon's pose frame by frame. Shields are
held in guard at the front of the chest (`shield.guard`), seen at an angle (`shield.squash`) and swaying with the
walk (`shield.sway`); where GPT's far
arm swings out behind the back, it is cut away (`far_arm_cut` polygons in the config -> `masks/far_arm/<anim>.png`).
`source/doll/unused/` keeps the old "carry" redraws, which put the sword where GPT's hands were, not where the
body's hands are.

## Layout

| Path | What |
|---|---|
| `docs/doll-brief.md` | layer contract, palette roles, item list |
| `source/doll/` | approved seed, raw generated strips, prompts, config |
| `assets/doll/` | `doll.json`, `body/<anim>.png`, `layers/<slot>/<item>/<anim>.png` (+ `<anim>_under.png`: drawn before the body; `<anim>_tones.png`: tone map for material colours), `masks/near_arm/<anim>.png`, `masks/far_arm/<anim>.png`, `masks/headgear/<id>/<anim>.png` |
| `source/doll_<dir>/`, `assets/doll_<dir>/` | the same for each isometric direction (`se`, `s`, `n`, `ne`) |
| `assets/doll/_preview/` | contact sheets per animation |
| `assets/asset-manifest.json` | per-layer provenance |
| `tools/` | pipeline, previews, generation helpers |
| `server/` | NestJS viewer (`/api/dolls` + static `assets/`) |

## Known limitations

- Every frame of every layer is a separate GPT redraw, so cloth and armour shimmer by a pixel or two between
  frames; torso armour in idle / walk / run is completed from its fullest frame to keep it steady.
- Heads and helmets are rigid sprites: the head stays upright in the run lean.
- Isometric views: green and blue cloaks and the purple skirt are recolours of the red cloak / blue skirt (`recolors`);
  two-handed swords are carried low, the tip forward and down to the ground, as in the side view; arms in run and attack frames
  are annotated automatically (`tools/auto_arms.py --bent`), so bracers can sit a pixel off in single frames; the
  bow string is the one GPT drew (a redrawn straight string crossed the body). Shields there are strapped to the
  hand without the weapon (`shield.mode: hand`), seen from behind (`back`) they are their back side behind the body.
- The source run cycle has uneven arm timing (the hand jumps between frames 2 and 3), so a carried weapon and
  the shield jump with it.
- Pants follow the mannequin's trouser folds (same shading, new colour).
- The red bow shares the wooden bow's limb animation (its own colours, recurve tips and grip wrap).
- Hurt, block and death: arm annotations of the lying frames come from the hands' skin blobs and the turned head;
  in a three-frame hit the head stays upright (only the fall turns it). While lying, the weapon lies along the body
  and the shield lies on it (from behind it is under the body).
