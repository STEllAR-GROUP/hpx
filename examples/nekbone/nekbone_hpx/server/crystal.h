#ifndef CRYSTAL_H
#define CRYSTAL_H

#if !defined(COMM_H) || !defined(MEM_H)
#warning "crystal.h" requires "comm.h" and "mem.h"
#endif

#define crystal_init   PREFIXED_NAME(crystal_init  )
#define crystal_free   PREFIXED_NAME(crystal_free  )
#define crystal_router PREFIXED_NAME(crystal_router)

struct crystal {
  struct comm comm;
  buffer data, work;
};

void crystal_init(struct crystal *cr, const struct comm *comm);
void crystal_free(struct crystal *cr);
void crystal_router(struct crystal *cr);

#endif
