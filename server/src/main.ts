import './env'; // first: settings from server/.env before anything reads process.env
import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import { NestExpressApplication } from '@nestjs/platform-express';
import { AppModule } from './app.module';
import { ASSETS_DIR, PUBLIC_DIR } from './paths';

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  // Sprites are re-read on every request, so freshly sliced frames show up after a page reload.
  app.useStaticAssets(ASSETS_DIR, { prefix: '/assets/', etag: false, lastModified: false, maxAge: 0 });
  app.useStaticAssets(PUBLIC_DIR);

  const port = Number(process.env.PORT ?? 3000);
  const host = process.env.HOST ?? '127.0.0.1';
  await app.listen(port, host);
  console.log(`Sprite viewer: http://${host}:${port}  (assets: ${ASSETS_DIR})`);
}

void bootstrap();
