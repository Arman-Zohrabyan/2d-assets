import { Controller, Get } from '@nestjs/common';
import { Doll, DollsService } from './dolls.service';

@Controller('api')
export class DollsController {
  constructor(private readonly dolls: DollsService) {}

  /** Paper dolls found under assets/: animations, slots, items and their layer strips. */
  @Get('dolls')
  list(): Promise<Doll[]> {
    return this.dolls.list();
  }
}
