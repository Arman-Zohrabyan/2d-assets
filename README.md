# test2d assets

Paper-doll pixel character for a side-view 2D game: one mannequin body, every piece of equipment
on its own layer, plus a NestJS viewer to look at any combination.

## Look at it

```bash
cd server
cp .env.example .env   # first time only: PORT, HOST, ASSETS_DIR
npm install            # first time only
npm start              # builds with tsc, then serves http://HOST:PORT (default http://127.0.0.1:3000)
```

`server/.env` is not committed; variables set in the shell override it (`PORT=4000 npm start`).

The page has an equipment panel (presets: Манекен, Рыцарь, Паладин, Лучник, Наёмник, Алебардщик, Страж, Охотница,
Случайно; under every armour piece: ready materials, a colour picker for any material colour, and × for the piece
as drawn), a big
view with frame stepping and per-layer toggles, a scene (arrows / A D to walk, Shift to run, Space to attack with
the equipped weapon, Q to switch weapons) and every animation in the current outfit. The tabs are Стойка, Ходьба,
Бег and one «Атака» that plays the equipped weapon's attack and switches with the weapon; without a weapon there
is no attack tab. A shield can only be picked with a one-handed weapon.

## Rebuild the layers

```bash
python3 tools/doll.py source/doll/doll.json        # raw strips -> assets/doll (layers + doll.json)
python3 tools/doll_preview.py assets/doll 2         # contact sheets in assets/doll/_preview
python3 tools/doll_preview.py assets/doll 6 --outfit knight --anims walk --out /tmp/walk.png   # one outfit
python3 tools/doll_manifest.py source/doll/doll.json
```

Then press "Обновить" on the page.

## Generate more

- `tools/gen_ref_image.sh OUT SIZE PROMPT_FILE REF...` — one image with reference images
- `tools/gen_batch.sh JOBS [PARALLEL]` — many at once; finished outputs are skipped
- Prompts: `source/doll/prompts/`, job lists: `source/doll/jobs_*.txt`

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
| `assets/doll/_preview/` | contact sheets per animation |
| `assets/asset-manifest.json` | per-layer provenance |
| `tools/` | pipeline, previews, generation helpers |
| `server/` | NestJS viewer (`/api/dolls` + static `assets/`) |

## Known limitations

- Every frame of every layer is a separate GPT redraw, so cloth and armour shimmer by a pixel or two between
  frames; torso armour in idle / walk / run is completed from its fullest frame to keep it steady.
- Heads and helmets are rigid sprites: the head stays upright in the run lean.
- The source run cycle has uneven arm timing (the hand jumps between frames 2 and 3), so a carried weapon and
  the shield jump with it.
- Pants follow the mannequin's trouser folds (same shading, new colour).
- The red bow shares the wooden bow's limb animation (its own colours, recurve tips and grip wrap).
