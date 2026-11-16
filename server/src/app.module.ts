import { Module } from '@nestjs/common';
import { DollsController } from './dolls/dolls.controller';
import { DollsService } from './dolls/dolls.service';

@Module({
  controllers: [DollsController],
  providers: [DollsService],
})
export class AppModule {}
