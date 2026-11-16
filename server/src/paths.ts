import { resolve } from 'node:path';

/** server/dist -> project root (the folder that holds assets/, source/, tools/). */
const PROJECT_ROOT = resolve(__dirname, '..', '..');

/** Sprite output of tools/slice_strips.py; override with ASSETS_DIR. */
export const ASSETS_DIR = process.env.ASSETS_DIR
  ? resolve(process.env.ASSETS_DIR)
  : resolve(PROJECT_ROOT, 'assets');

/** The viewer page. */
export const PUBLIC_DIR = resolve(__dirname, '..', 'public');
