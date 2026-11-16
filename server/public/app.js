'use strict';

// `shape@material` picks a material version of a piece (e.g. plate@gold); without it the piece is as drawn
const PRESETS = {
  Mannequin: { head: 'a' },
  Knight: { head: 'a', headgear: 'knight', chest: 'plate', shoulders: 'bands', greaves: 'plate', bracers: 'full', cloak: 'red', skirt: 'blue', shield: 'heater', weapon: 'sword1h' },
  Paladin: { head: 'b', headgear: 'knight@gold', chest: 'plate@gold', shoulders: 'layered', greaves: 'plate@gold', bracers: 'full@gold', cloak: 'blue', skirt: 'red', shield: 'heater', weapon: 'sword1h' },
  Archer: { head: 'b', headgear: 'archer', pants: 'green', chest: 'riveted', greaves: 'boots', bracers: 'forearm@leather', cloak: 'green', weapon: 'bow' },
  Mercenary: { head: 'b', pants: 'leather', chest: 'mail', shoulders: 'layered@steel', greaves: 'plate', bracers: 'elbow', weapon: 'sword2h' },
  Halberdier: { head: 'a', headgear: 'nasal@copper', pants: 'blue', chest: 'mail@copper', shoulders: 'round@copper', greaves: 'cuffed', bracers: 'forearm@copper', weapon: 'halberd' },
  Guard: { head: 'c', headgear: 'nasal', pants: 'blue', chest: 'brigandine', shoulders: 'round', greaves: 'cuffed', bracers: 'forearm', cloak: 'blue', shield: 'round', weapon: 'falchion' },
  Huntress: { head: 'c', headgear: 'archer@silver', pants: 'green', chest: 'riveted', greaves: 'boots', bracers: 'wrist@silver', cloak: 'green', weapon: 'bow_dark' },
};

/** Animation tabs: the movement animations, plus one attack that follows the weapon (none without a weapon). */
const ATTACK = 'attack';

const $ = (sel) => document.querySelector(sel);

const state = {
  doll: null,
  sel: {},
  mat: {}, // slot -> a ready material id ('gold'), a colour ('#c08040'), or null for the piece as drawn
  hidden: new Set(),
  view: 'idle', // an animation id, or ATTACK (the equipped weapon's attack)
  anim: 'idle',
  frame: 0,
  acc: 0,
  playing: true,
  flip: false,
  scale: 4,
  speed: 1,
  guides: false,
  stamp: Date.now(),
  cards: [],
};

/* ---------- images & composition ---------- */

const images = new Map();
function loadImage(src) {
  const key = `${src}?v=${state.stamp}`;
  if (!images.has(key)) {
    images.set(key, new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load: ${src}`));
      img.src = key;
    }));
  }
  return images.get(key);
}
const ready = new Map(); // url -> HTMLImageElement once loaded

async function preload(doll) {
  const urls = doll.slots.flatMap((s) => s.items.flatMap((i) => [...Object.values(i.strips), ...Object.values(i.hides ?? {}),
    ...Object.values(i.under ?? {}), ...Object.values(i.tones ?? {})]))
    .concat(Object.values(doll.masks?.near_arm ?? {}), Object.values(doll.masks?.far_arm ?? {}));
  await Promise.all(urls.map(async (u) => ready.set(u, await loadImage(u))));
}

function animDef(id) {
  return state.doll.animations.find((a) => a.id === id);
}

function slotItem(slotId, itemId = state.sel[slotId]) {
  return state.doll.slots.find((s) => s.id === slotId)?.items.find((i) => i.id === itemId) ?? null;
}

/* ---------- material colours ---------- */

const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const rgb2hex = (c) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;

function rgbToHls([r, g, b]) {
  [r, g, b] = [r / 255, g / 255, b / 255];
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, l, 0];
  const d = mx - mn;
  const s = l <= 0.5 ? d / (mx + mn) : d / (2 - mx - mn);
  const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [((h * 60) + 360) % 360, l, s];
}
function hlsToRgb([h, l, s]) {
  if (s === 0) return [l, l, l].map((v) => v * 255);
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
  const f = (t) => {
    t = (t + 1) % 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [f(h / 360 + 1 / 3), f(h / 360), f(h / 360 - 1 / 3)].map((v) => v * 255);
}
const shiftHue = (h, toward, deg) => {
  const d = ((toward - h + 540) % 360) - 180;
  return (h + Math.max(-deg, Math.min(deg, d)) + 360) % 360;
};
/** 5 tones from one material colour (tone 2): lighter and a little warmer up, darker and a little cooler down.
 *  Same formula as tools/doll_preview.py (ramp_from). */
function rampFrom(hex) {
  const [h, l, s] = rgbToHls(hex2rgb(hex));
  const spec = [[60, 8, 0.6, (x) => x + (1 - x) * 0.6], [60, 4, 0.85, (x) => x + (1 - x) * 0.3], [0, 0, 1, (x) => x],
    [240, 6, 1.05, (x) => x * 0.68], [240, 12, 1.1, (x) => x * 0.42]];
  return spec.map(([toward, deg, ks, fl]) =>
    rgb2hex(hlsToRgb([deg ? shiftHue(h, toward, deg) : h, Math.min(1, Math.max(0, fl(l))), Math.min(1, s * ks)])));
}
/** The ramp for a slot's choice: a ready material's own ramp, or one built from a picked colour. */
function rampOf(choice) {
  if (!choice) return null;
  return choice.startsWith('#') ? rampFrom(choice) : state.doll.materials?.[choice]?.ramp ?? null;
}

const recoloured = new Map(); // `${strip}|${ramp}` -> canvas
/** The piece's own pixels (its tone map: grey 0..240 = tone 0..4) take the ramp's colours. */
function recolouredStrip(stripUrl, tonesUrl, ramp) {
  const key = `${stripUrl}|${ramp.join(',')}`;
  if (recoloured.has(key)) return recoloured.get(key);
  const src = ready.get(stripUrl), tones = ready.get(tonesUrl);
  if (!src || !tones) return src;
  const c = document.createElement('canvas');
  c.width = src.width;
  c.height = src.height;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(tones, 0, 0);
  const t = g.getImageData(0, 0, c.width, c.height).data;
  g.clearRect(0, 0, c.width, c.height);
  g.drawImage(src, 0, 0);
  const img = g.getImageData(0, 0, c.width, c.height);
  const rgb = ramp.map(hex2rgb);
  for (let i = 0; i < t.length; i += 4) {
    if (t[i + 3] === 0) continue;
    const k = Math.min(4, Math.round(t[i] / 60));
    [img.data[i], img.data[i + 1], img.data[i + 2]] = rgb[k];
  }
  g.putImageData(img, 0, 0);
  if (recoloured.size > 600) recoloured.clear(); // dragging the colour picker makes a new colour every step
  recoloured.set(key, c);
  return c;
}

/** The image of a worn piece for one animation: as drawn, or recoloured into the slot's material. */
function wornImage(slotId, item, animId) {
  const ramp = rampOf(state.mat[slotId]);
  if (ramp && item.tones?.[animId]) return recolouredStrip(item.strips[animId], item.tones[animId], ramp);
  return ready.get(item.strips[animId]);
}

/** A slot with `requires` (the shield) is only worn with a matching item: a one-handed weapon. */
function allowed(slot, sel = state.sel) {
  if (!slot.requires) return true;
  return slotItem(slot.requires.slot, sel[slot.requires.slot])?.hands === slot.requires.hands;
}

/** Selected item of every slot, in draw order. */
function layersFor(animId) {
  const out = [];
  const maskUrl = state.doll.masks?.near_arm?.[animId];
  const shieldSlot = state.doll.slots.find((s) => s.id === 'shield');
  const shieldOn = shieldSlot && allowed(shieldSlot) && !state.hidden.has('shield')
    && slotItem('shield')?.strips[animId];
  // with a shield, the far arm is behind it: cut from the body and the arm-bound layers
  const farUrl = shieldOn ? state.doll.masks?.far_arm?.[animId] : null;
  const cuts = new Set(shieldSlot?.cuts ?? []);
  for (const slot of state.doll.slots) {  // `under` strips go first: drawn before the body
    const want = slot.id === 'body' ? 'base' : state.sel[slot.id];
    const item = slot.items.find((i) => i.id === want);
    if (item?.under?.[animId] && allowed(slot) && !state.hidden.has(slot.id)) {
      out.push({ slot, item, img: ready.get(item.under[animId]), under: true });
    }
  }
  for (const slot of state.doll.slots) {
    const want = slot.id === 'body' ? 'base' : state.sel[slot.id];
    const item = slot.items.find((i) => i.id === want);
    if (!item || !item.strips[animId] || !allowed(slot)) continue;
    const layer = { slot, item, img: wornImage(slot.id, item, animId) };
    if (slot.id === 'shield' && maskUrl) layer.mask = ready.get(maskUrl);
    if (slot.id === 'head') {
      const hideUrl = slotItem('headgear')?.hides?.[animId];
      if (hideUrl && !state.hidden.has('headgear')) layer.hide = ready.get(hideUrl);
    }
    if (farUrl && cuts.has(slot.id)) layer.hide = ready.get(farUrl);
    out.push(layer);
  }
  return out;
}

const frameBuf = document.createElement('canvas');
const armBuf = document.createElement('canvas');
const headBuf = document.createElement('canvas');

/** Compose one frame at 1x (the shield sits between the body and the sword arm), then blit it scaled. */
function drawLayers(ctx, layers, frame, x, y, scale, flip) {
  const [fw, fh] = state.doll.frame_size;
  for (const c of [frameBuf, armBuf, headBuf]) if (c.width !== fw) { c.width = fw; c.height = fh; }
  const f = frameBuf.getContext('2d');
  f.clearRect(0, 0, fw, fh);
  const under = [];
  for (const l of layers) {
    if (l.img && l.hide) {
      // a mask cut out first: hair under headgear, or the far arm behind a shield
      const h = headBuf.getContext('2d');
      h.clearRect(0, 0, fw, fh);
      h.drawImage(l.img, frame * fw, 0, fw, fh, 0, 0, fw, fh);
      h.globalCompositeOperation = 'destination-out';
      h.drawImage(l.hide, frame * fw, 0, fw, fh, 0, 0, fw, fh);
      h.globalCompositeOperation = 'source-over';
      f.drawImage(headBuf, 0, 0);
    } else if (l.img) f.drawImage(l.img, frame * fw, 0, fw, fh, 0, 0, fw, fh);
    if (l.mask) {
      // the near (sword) arm goes in front of the shield: redraw what is under it, clipped to the arm
      const a = armBuf.getContext('2d');
      a.clearRect(0, 0, fw, fh);
      for (const u of under) a.drawImage(u.img, frame * fw, 0, fw, fh, 0, 0, fw, fh);
      a.globalCompositeOperation = 'destination-in';
      a.drawImage(l.mask, frame * fw, 0, fw, fh, 0, 0, fw, fh);
      a.globalCompositeOperation = 'source-over';
      f.drawImage(armBuf, 0, 0);
    }
    if (l.img && l.slot.id !== 'shield' && !l.under) under.push(l);
  }
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  if (flip) {
    ctx.translate(x + fw * scale, y);
    ctx.scale(-1, 1);
    x = 0;
    y = 0;
  }
  ctx.drawImage(frameBuf, 0, 0, fw, fh, x, y, fw * scale, fh * scale);
  ctx.restore();
}

function drawGuides(ctx, scale, flip) {
  const [fw] = state.doll.frame_size;
  const { x, y } = state.doll.pivot;
  ctx.fillStyle = 'rgba(255, 92, 92, 0.85)';
  ctx.fillRect(0, y * scale - 1, ctx.canvas.width, 1);
  const px = (flip ? fw - x : x) * scale;
  for (let yy = 0; yy < ctx.canvas.height; yy += 6) ctx.fillRect(px, yy, 1, 3);
}

/* ---------- equipment panel ---------- */

function renderEquipment() {
  const presets = $('#presets');
  presets.textContent = '';
  for (const [name, outfit] of Object.entries(PRESETS)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = name;
    b.addEventListener('click', () => applyOutfit(outfit));
    presets.append(b);
  }
  const random = document.createElement('button');
  random.type = 'button';
  random.textContent = 'Random';
  random.addEventListener('click', () => {
    const outfit = {};
    for (const slot of state.doll.slots) {
      if (slot.id === 'body' || !slot.items.length || slot.requires) continue;
      const pool = slot.required ? slot.items : [null, ...slot.items];
      const pick = pool[Math.floor(Math.random() * pool.length)];
      if (pick) {
        const mats = pick.tones ? [null, ...(slot.materials ?? [])] : [null];
        const mat = mats[Math.floor(Math.random() * mats.length)];
        outfit[slot.id] = mat ? `${pick.id}@${mat}` : pick.id;
      }
    }
    for (const slot of state.doll.slots.filter((s) => s.requires)) {  // e.g. a shield, half the time
      if (allowed(slot, outfit) && Math.random() < 0.5) {
        outfit[slot.id] = slot.items[Math.floor(Math.random() * slot.items.length)].id;
      }
    }
    applyOutfit(outfit);
  });
  presets.append(random);

  const root = $('#slots');
  root.textContent = '';
  for (const slot of state.doll.slots) {
    if (slot.id === 'body') continue;
    const row = document.createElement('div');
    row.className = 'slot';
    const label = document.createElement('label');
    label.textContent = slot.label;
    label.htmlFor = `slot-${slot.id}`;
    const select = document.createElement('select');
    select.id = `slot-${slot.id}`;
    if (!slot.required) select.append(new Option('— none —', ''));
    for (const item of slot.items) select.append(new Option(item.label, item.id));
    select.value = state.sel[slot.id] ?? '';
    select.addEventListener('change', () => {
      state.sel[slot.id] = select.value || null;
      onOutfitChanged();
    });
    row.append(label, select);
    if (slot.items.some((i) => i.tones)) {
      // material colour of this piece: a ready material (exact ramp), any colour, or as drawn
      const box = document.createElement('div');
      box.className = 'mat';
      box.id = `mat-${slot.id}`;
      for (const m of slot.materials ?? []) {
        const def = state.doll.materials?.[m];
        if (!def?.ramp) continue;
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'swatch';
        b.dataset.mat = m;
        b.title = def.label;
        b.style.background = `linear-gradient(135deg, ${def.ramp[1]}, ${def.ramp[2]} 55%, ${def.ramp[3]})`;
        b.addEventListener('click', () => { state.mat[slot.id] = m; onOutfitChanged(); });
        box.append(b);
      }
      const pick = document.createElement('input');
      pick.type = 'color';
      pick.title = 'Any material colour';
      pick.addEventListener('input', () => { state.mat[slot.id] = pick.value; onOutfitChanged(); });
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'mat-reset';
      reset.textContent = '×';
      reset.title = 'As drawn';
      reset.addEventListener('click', () => { state.mat[slot.id] = null; onOutfitChanged(); });
      const name = document.createElement('span');
      name.className = 'hint mat-name';
      const line = document.createElement('div');
      line.className = 'mat-line';
      line.append(reset, name);
      box.append(pick, line);
      row.append(box);
    }
    if (slot.requires) {
      const hint = document.createElement('span');
      hint.className = 'hint slot-hint';
      hint.id = `hint-${slot.id}`;
      hint.textContent = 'one-handed weapons only';
      row.append(hint);
    }
    root.append(row);
  }
}

/** Slots whose requirement is not met are cleared and locked (a shield without a one-handed weapon). */
function enforceRequirements() {
  for (const slot of state.doll.slots.filter((s) => s.requires)) {
    const ok = allowed(slot);
    if (!ok) state.sel[slot.id] = null;
    const select = $(`#slot-${slot.id}`);
    if (select) {
      select.disabled = !ok;
      select.value = state.sel[slot.id] ?? '';
    }
    $(`#hint-${slot.id}`)?.classList.toggle('warn', !ok);
  }
}

/** The material row shows the slot's choice; it is off for a piece without a tone map (or no piece). */
function renderMaterials() {
  for (const slot of state.doll.slots) {
    const box = $(`#mat-${slot.id}`);
    if (!box) continue;
    const item = slotItem(slot.id);
    const on = Boolean(item?.tones);
    if (!on) state.mat[slot.id] = null;
    box.classList.toggle('off', !on);
    const choice = state.mat[slot.id];
    for (const b of box.querySelectorAll('.swatch')) b.classList.toggle('on', b.dataset.mat === choice);
    const pick = box.querySelector('input[type=color]');
    pick.classList.toggle('on', Boolean(choice?.startsWith('#')));
    pick.value = choice?.startsWith('#') ? choice : (rampOf(choice) ?? [0, 0, '#808080'])[2];
    const names = state.doll.materials ?? {};
    box.querySelector('.mat-name').textContent = !on ? '' : !choice ? (names[item.material]?.label ?? 'as drawn')
      : choice.startsWith('#') ? choice : names[choice]?.label ?? choice;
  }
}

function applyOutfit(outfit) {
  state.sel = {};
  state.mat = {};
  for (const slot of state.doll.slots) {
    if (slot.id === 'body') continue;
    const [want, mat] = (outfit[slot.id] ?? '').split('@');
    const item = slot.items.find((i) => i.id === want) ?? (slot.required ? slot.items[0] : null);
    state.sel[slot.id] = item ? item.id : null;
    state.mat[slot.id] = item?.tones && mat && (mat.startsWith('#') || state.doll.materials?.[mat]?.ramp) ? mat : null;
    const select = $(`#slot-${slot.id}`);
    if (select) select.value = state.sel[slot.id] ?? '';
  }
  onOutfitChanged();
}

function onOutfitChanged() {
  enforceRequirements();
  renderMaterials();
  syncAnim();
  renderAnimTabs();
  drawMain();
  renderGrid();
}

/* ---------- main viewer ---------- */

/** What can be played with this outfit: idle / walk / run, and the attack of the equipped weapon's kind. */
function viewList() {
  const views = state.doll.animations.filter((a) => !a.weapon).map((a) => ({ view: a.id, anim: a }));
  const atk = attackAnim();
  if (atk) views.push({ view: ATTACK, anim: atk });
  return views;
}

/** The attack is named after the weapon (a glaive uses the two-handed attack, but it is the glaive's attack). */
function viewLabel(v) {
  if (v.view !== ATTACK) return v.anim.label;
  const w = slotItem('weapon')?.label?.split(' (')[0];
  return w ? `Attack: ${w.toLowerCase()}` : v.anim.label;
}

/** The attack tab follows the weapon: another weapon kind switches it, no weapon falls back to idle. */
function syncAnim() {
  const before = state.anim;
  const views = viewList();
  const cur = views.find((v) => v.view === state.view) ?? views[0];
  state.view = cur.view;
  state.anim = cur.anim.id;
  if (state.anim !== before) { state.frame = 0; state.acc = 0; }
}

function renderAnimTabs() {
  const tabs = $('#anim-tabs');
  tabs.textContent = '';
  for (const v of viewList()) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = viewLabel(v);
    b.classList.toggle('on', v.view === state.view);
    if (v.view === ATTACK) b.title = `Attack of the equipped weapon: ${slotItem('weapon')?.label}`;
    b.addEventListener('click', () => {
      state.view = v.view;
      syncAnim();
      state.frame = 0;
      state.acc = 0;
      state.playing = true;
      renderAnimTabs();
      drawMain();
    });
    tabs.append(b);
  }
}

function drawMain() {
  const [fw, fh] = state.doll.frame_size;
  const canvas = $('#main-canvas');
  const s = state.scale;
  if (canvas.width !== fw * s) { canvas.width = fw * s; canvas.height = fh * s; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const layers = layersFor(state.anim);
  drawLayers(ctx, layers.filter((l) => !state.hidden.has(l.slot.id)), state.frame, 0, 0, s, state.flip);
  if (state.guides) drawGuides(ctx, s, state.flip);
  const a = animDef(state.anim);
  $('#frame-no').textContent = `${a.label} · frame ${state.frame + 1} / ${a.frames} · ${a.fps} fps`;
  document.querySelector('[data-act="play"]').textContent = state.playing ? '❚❚' : '▶';
  document.querySelector('[data-act="flip"]').classList.toggle('on', state.flip);
  drawLayerStrip(layers);
}

function drawLayerStrip(layers) {
  const root = $('#layer-strip');
  const [fw, fh] = state.doll.frame_size;
  root.textContent = '';
  for (const l of layers.filter((x) => !x.under)) {
    const box = document.createElement('div');
    box.className = 'layer';
    box.classList.toggle('hidden', state.hidden.has(l.slot.id));
    box.title = `${l.slot.label}: ${l.item.label}`;
    const stage = document.createElement('div');
    stage.className = 'stage';
    const c = document.createElement('canvas');
    c.width = fw;
    c.height = fh;
    drawLayers(c.getContext('2d'), [l], state.frame, 0, 0, 1, state.flip);
    stage.append(c);
    const name = document.createElement('span');
    name.textContent = l.slot.label;
    box.append(stage, name);
    box.addEventListener('click', () => {
      if (state.hidden.has(l.slot.id)) state.hidden.delete(l.slot.id);
      else state.hidden.add(l.slot.id);
      drawMain();
    });
    root.append(box);
  }
}

document.querySelector('.viewer .controls').addEventListener('click', (e) => {
  const act = e.target.closest('button')?.dataset.act;
  if (!act) return;
  const n = animDef(state.anim).frames;
  if (act === 'play') state.playing = !state.playing;
  if (act === 'prev') { state.playing = false; state.frame = (state.frame - 1 + n) % n; }
  if (act === 'next') { state.playing = false; state.frame = (state.frame + 1) % n; }
  if (act === 'flip') state.flip = !state.flip;
  drawMain();
});

/* ---------- grid of every animation ---------- */

function renderGrid() {
  const root = $('#grid');
  root.textContent = '';
  state.cards = [];
  const [fw, fh] = state.doll.frame_size;
  for (const v of viewList()) {
    const a = v.anim;
    const card = document.createElement('article');
    card.className = 'card';
    card.innerHTML = '<div class="stage"><canvas></canvas></div><div class="title"><span></span><span class="meta"></span></div>';
    card.querySelector('.title span').textContent = viewLabel(v);
    card.querySelector('.meta').textContent = `${a.frames} frames · ${a.fps} fps`;
    const canvas = card.querySelector('canvas');
    canvas.width = fw * 2;
    canvas.height = fh * 2;
    root.append(card);
    state.cards.push({ a, canvas, frame: 0, acc: 0, layers: layersFor(a.id) });
  }
  state.cards.forEach(drawCard);
}

function drawCard(c) {
  const ctx = c.canvas.getContext('2d');
  ctx.clearRect(0, 0, c.canvas.width, c.canvas.height);
  drawLayers(ctx, c.layers, c.frame, 0, 0, 2, false);
}

/* ---------- playground ---------- */

const WORLD = { w: 320, h: 120, ground: 108 };
const PG_SCALE = 3;
const pg = {
  canvas: $('#pg-canvas'),
  buffer: document.createElement('canvas'),
  keys: new Set(),
  x: 80,
  facing: 1,
  action: 'idle',
  attack: null,
  frame: 0,
  acc: 0,
  walk: 34,
  run: 78,
};
pg.buffer.width = WORLD.w;
pg.buffer.height = WORLD.h;
pg.canvas.width = WORLD.w * PG_SCALE;
pg.canvas.height = WORLD.h * PG_SCALE;

function weaponCycle() {
  return [null, ...(state.doll.slots.find((s) => s.id === 'weapon')?.items.map((i) => i.id) ?? [])];
}
function attackAnim() {
  const id = slotItem('weapon')?.attack;
  return id ? animDef(id) : null;
}

function pgUpdate(dt) {
  if (!state.doll) return;
  const left = pg.keys.has('ArrowLeft') || pg.keys.has('KeyA');
  const right = pg.keys.has('ArrowRight') || pg.keys.has('KeyD');
  const dx = (right ? 1 : 0) - (left ? 1 : 0);
  const running = pg.keys.has('ShiftLeft') || pg.keys.has('ShiftRight');
  let anim;
  if (pg.attack) {
    anim = animDef(pg.attack);
  } else {
    if (dx) pg.facing = dx;
    const next = dx ? (running ? 'run' : 'walk') : 'idle';
    if (next !== pg.action && !(['walk', 'run'].includes(next) && ['walk', 'run'].includes(pg.action))) {
      pg.frame = 0;
      pg.acc = 0;
    }
    pg.action = next;
    pg.x += dx * (running ? pg.run : pg.walk) * dt * state.speed;
    if (pg.x < -30) pg.x = WORLD.w + 30;
    if (pg.x > WORLD.w + 30) pg.x = -30;
    anim = animDef(pg.action);
  }
  pg.acc += dt * state.speed;
  const d = 1 / anim.fps;
  while (pg.acc >= d) {
    pg.acc -= d;
    pg.frame += 1;
    if (pg.frame >= anim.frames) {
      if (pg.attack) { pg.attack = null; pg.action = 'idle'; pg.frame = 0; anim = animDef('idle'); }
      else pg.frame = 0;
    }
  }
  pg.anim = anim;
  const shield = allowed(state.doll.slots.find((s) => s.id === 'shield') ?? {}) ? slotItem('shield')?.label : null;
  $('#pg-status').textContent = `${anim.label} · frame ${pg.frame + 1}/${anim.frames} · weapon: ${
    slotItem('weapon')?.label ?? 'none'}${shield ? ` · shield: ${shield}` : ''}`;
}

function pgDraw() {
  if (!state.doll || !pg.anim) return;
  const ctx = pg.buffer.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#8fbbe3';
  ctx.fillRect(0, 0, WORLD.w, WORLD.h);
  ctx.fillStyle = '#5c8f4a';
  ctx.fillRect(0, WORLD.ground - 20, WORLD.w, 20);
  ctx.fillStyle = '#4f9a3f';
  ctx.fillRect(0, WORLD.ground, WORLD.w, 3);
  ctx.fillStyle = '#7a5534';
  ctx.fillRect(0, WORLD.ground + 3, WORLD.w, WORLD.h - WORLD.ground - 3);
  const { x: px, y: py } = state.doll.pivot;
  const flip = pg.facing < 0;
  const [fw] = state.doll.frame_size;
  const left = Math.round(pg.x - (flip ? fw - px : px));
  drawLayers(ctx, layersFor(pg.anim.id), pg.frame, left, WORLD.ground - py, 1, flip);
  const out = pg.canvas.getContext('2d');
  out.imageSmoothingEnabled = false;
  out.drawImage(pg.buffer, 0, 0, pg.canvas.width, pg.canvas.height);
}

const PG_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyA', 'KeyD', 'ShiftLeft', 'ShiftRight', 'Space', 'KeyQ']);
pg.canvas.addEventListener('keydown', (e) => {
  if (PG_KEYS.has(e.code)) e.preventDefault();
  if (e.code === 'Space' && !e.repeat && !pg.attack) {
    const a = attackAnim();
    if (a) { pg.attack = a.id; pg.frame = 0; pg.acc = 0; }
  }
  if (e.code === 'KeyQ' && !e.repeat) {
    const cycle = weaponCycle();
    const i = cycle.indexOf(state.sel.weapon ?? null);
    state.sel.weapon = cycle[(i + 1) % cycle.length];
    $('#slot-weapon').value = state.sel.weapon ?? '';
    onOutfitChanged();
  }
  pg.keys.add(e.code);
});
pg.canvas.addEventListener('keyup', (e) => pg.keys.delete(e.code));
pg.canvas.addEventListener('blur', () => pg.keys.clear());
pg.canvas.addEventListener('pointerdown', () => pg.canvas.focus());

/* ---------- toolbar & loop ---------- */

$('#scale').addEventListener('change', (e) => { state.scale = Number(e.target.value); drawMain(); });
$('#bg').addEventListener('change', (e) => {
  document.body.className = `bg-${e.target.value}`;
});
$('#speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); });
$('#guides').addEventListener('change', (e) => { state.guides = e.target.checked; drawMain(); });
$('#reload').addEventListener('click', () => { state.stamp = Date.now(); load(); });

let last = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (state.doll) {
    if (state.playing) {
      const a = animDef(state.anim);
      state.acc += dt * state.speed;
      let changed = false;
      while (state.acc >= 1 / a.fps) { state.acc -= 1 / a.fps; state.frame = (state.frame + 1) % a.frames; changed = true; }
      if (changed) drawMain();
    }
    for (const c of state.cards) {
      c.acc += dt * state.speed;
      let changed = false;
      while (c.acc >= 1 / c.a.fps) { c.acc -= 1 / c.a.fps; c.frame = (c.frame + 1) % c.a.frames; changed = true; }
      if (changed) drawCard(c);
    }
    pgUpdate(dt);
    pgDraw();
  }
  requestAnimationFrame(tick);
}

async function load() {
  const error = $('#error');
  error.hidden = true;
  try {
    const res = await fetch('/api/dolls', { cache: 'no-store' });
    if (!res.ok) throw new Error(`/api/dolls: HTTP ${res.status}`);
    const dolls = await res.json();
    if (!dolls.length) throw new Error('no doll.json under assets/ — run tools/doll.py');
    state.doll = dolls[0];
    await preload(state.doll);
  } catch (e) {
    error.textContent = `Could not load the doll: ${e.message}`;
    error.hidden = false;
    return;
  }
  const keep = Object.keys(state.sel).length ? { ...state.sel } : PRESETS.Knight;
  renderEquipment();
  applyOutfit(keep);
}

load().then(() => requestAnimationFrame(tick));
