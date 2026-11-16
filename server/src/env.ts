import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

/** server/.env (next to package.json), if present; variables already set in the shell win. */
const envFile = resolve(__dirname, '..', '.env');
if (existsSync(envFile)) process.loadEnvFile(envFile);
