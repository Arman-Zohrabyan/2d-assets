import { Injectable, Logger } from '@nestjs/common';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ASSETS_DIR } from '../paths';

/** assets/<doll>/doll.json as written by tools/doll.py. */
interface DollFile {
  name: string;
  /** shown in the doll picker, e.g. "South (front)" */
  label?: string;
  frame_size: [number, number];
  pivot: { x: number; y: number };
  /** right (side view), or a compass direction: south-east, south, north, north-east (the mirrored ones are flips) */
  facing: string;
  view: string;
  compose?: string;
  animations: { id: string; label: string; frames: number; fps: number; loop: boolean; weapon?: string }[];
  /** masks.near_arm[anim]: the sword arm, drawn again over a shield; masks.far_arm[anim]: the far arm, cut
   *  out of the shield slot's `cuts` while a shield is worn */
  masks?: { near_arm?: Record<string, string>; far_arm?: Record<string, string> };
  /** ready materials: label and 5-tone ramp (tone 2 = the material colour) */
  materials?: Record<string, { label: string; ramp?: string[] }>;
  slots: {
    id: string;
    label: string;
    required?: boolean;
    /** e.g. a shield needs a one-handed weapon: { slot: 'weapon', hands: 1 } */
    requires?: { slot: string; hands: number };
    /** shield: slots that lose masks.far_arm while it is worn */
    cuts?: string[];
    /** ready materials offered for this slot */
    materials?: string[];
    items: {
      id: string;
      label: string;
      strips: Record<string, string>;
      kind?: string;
      hands?: number;
      attack?: string | null;
      /** headgear: per animation, the head pixels to cut before drawing it */
      hides?: Record<string, string>;
      /** drawn before the body (a cape continued under the far arm; it shows where a shield cut that arm) */
      under?: Record<string, string>;
      /** what the piece is made of as drawn */
      material?: string | null;
      /** per animation: the piece's own pixels by tone (grey 0..240 = highlight..deepest), to recolour it */
      tones?: Record<string, string>;
    }[];
  }[];
}

/** Same shape, with every strip path turned into a URL under /assets. */
export type Doll = DollFile & { id: string };

const url = (...parts: string[]): string =>
  '/assets/' + parts.flatMap((p) => p.split('/')).map(encodeURIComponent).join('/');

@Injectable()
export class DollsService {
  private readonly log = new Logger(DollsService.name);

  /** Every assets/<dir>/doll.json, read fresh on each call so a re-run of the pipeline shows up on reload. */
  async list(): Promise<Doll[]> {
    const entries = await readdir(ASSETS_DIR, { withFileTypes: true }).catch(() => []);
    const dolls: Doll[] = [];
    for (const dir of entries.filter((e) => e.isDirectory() && !e.name.startsWith('.')).map((e) => e.name).sort()) {
      const file = join(ASSETS_DIR, dir, 'doll.json');
      let data: DollFile;
      try {
        data = JSON.parse(await readFile(file, 'utf8')) as DollFile;
      } catch {
        continue; // not a doll folder
      }
      try {
        const toUrls = (m: Record<string, string> = {}) =>
          Object.fromEntries(Object.entries(m).map(([anim, rel]) => [anim, url(dir, rel)]));
        dolls.push({
          ...data,
          id: dir,
          masks: { near_arm: toUrls(data.masks?.near_arm), far_arm: toUrls(data.masks?.far_arm) },
          slots: data.slots.map((slot) => ({
            ...slot,
            items: slot.items.map((item) => ({
              ...item,
              strips: toUrls(item.strips),
              ...(item.hides ? { hides: toUrls(item.hides) } : {}),
              ...(item.under ? { under: toUrls(item.under) } : {}),
              ...(item.tones ? { tones: toUrls(item.tones) } : {}),
            })),
          })),
        });
      } catch (error) {
        this.log.warn(`Skipping ${file}: ${(error as Error).message}`);
      }
    }
    return dolls;
  }
}
