#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "types.h"
#include "fail.h"
#include "mem.h"
#include "sort.h"

#define sarray_permute_     PREFIXED_NAME(sarray_permute_)
#define sarray_permute_buf_ PREFIXED_NAME(sarray_permute_buf_)

void sarray_permute_(size_t size, void *A, size_t n, uint *perm, void *work)
{
  char *const ar = A, *const item = work;
  sint *const fperm = (sint*)perm;
  uint i;
  for(i=0;i<n;++i) {
    sint pi = fperm[i];
    if(pi<0) { fperm[i] = -pi-1; continue; }
    else if((uint)pi==i) continue;
    else {
      char *dst = ar+i*size, *src = ar+pi*size;
      memcpy(item, dst, size);
      for(;;) {
        sint ppi;
        memcpy(dst, src, size);
        dst=src;
        ppi=fperm[pi], fperm[pi]=-ppi-1, pi=ppi;
        if((uint)pi==i) break;
        src=ar+pi*size;
      }
      memcpy(dst, item, size);
    }
  }
}

void sarray_permute_buf_(size_t align, size_t size, void *A, size_t n,
                         buffer *buf)
{
  buffer_reserve(buf,align_as_(align,n*sizeof(uint)+size));
  sarray_permute_(size,A,n, buf->ptr,
                 (char*)buf->ptr + align_as_(align,n*sizeof(uint)));
}
