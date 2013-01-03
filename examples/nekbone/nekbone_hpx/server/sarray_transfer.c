#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "comm.h"
#include "mem.h"
#include "crystal.h"
#include "sort.h"

#define sarray_transfer_many PREFIXED_NAME(sarray_transfer_many)
#define sarray_transfer_     PREFIXED_NAME(sarray_transfer_    )
#define sarray_transfer_ext_ PREFIXED_NAME(sarray_transfer_ext_)

static void pack_int(
  buffer *const data, const unsigned row_size, const uint id,
  const char *const restrict input, const uint n, const unsigned size,
  const unsigned p_off, const uint *const restrict perm)
{
  const unsigned after = p_off + sizeof(uint), after_len = size-after;

#define GET_P() memcpy(&p,row+p_off,sizeof(uint))
#define COPY_ROW() memcpy(out,row,p_off), \
                   memcpy((char*)out + p_off,row+after,after_len)

#define PACK_BODY() do {                                                  \
  uint dummy, *len_ptr=&dummy;                                            \
  uint i, p,lp = -(uint)1, len=0;                                         \
  uint *restrict out = buffer_reserve(data, n*(row_size+3)*sizeof(uint)); \
  for(i=0;i<n;++i) {                                                      \
    const char *row = input + size*perm[i];                               \
    GET_P();                                                              \
    if(p!=lp) {                                                           \
      lp = p;                                                             \
      *len_ptr = len;       /* previous message length */                 \
      *out++ = p;           /* target */                                  \
      *out++ = id;          /* source */                                  \
      len_ptr=out++; len=0; /* length (t.b.d.) */                         \
    }                                                                     \
    COPY_ROW();                                                           \
    out += row_size, len += row_size;                                     \
  }                                                                       \
  *len_ptr = len; /* last message length */                               \
  data->n = out - (uint*)data->ptr;                                       \
} while(0)
  PACK_BODY();
#undef COPY_ROW
#undef GET_P
}

static void pack_ext(
  buffer *const data, const unsigned row_size, const uint id,
  const char *const restrict input, const uint n, const unsigned size,
  const uint *const restrict proc, const unsigned proc_stride,
  const uint *const restrict perm)
{
  #define GET_P() p=*(const uint*)((const char*)proc+proc_stride*perm[i])
  #define COPY_ROW() memcpy(out,row,size)
  PACK_BODY();
  #undef PACK_BODY
  #undef COPY_ROW
  #undef GET_P
}

static void pack_more(
  buffer *const data, const unsigned off, const unsigned row_size,
  const char *const restrict input, const unsigned size,
  const uint *restrict perm)
{
  uint *restrict buf = data->ptr, *buf_end = buf+data->n;
  while(buf!=buf_end) {
    uint *msg_end = buf+3+buf[2]; buf+=3;
    while(buf!=msg_end)
      memcpy(buf+off, input+size*(*perm++), size), buf+=row_size;
  }
}

static void unpack_more(
  char *restrict out, const unsigned size,
  const buffer *const data, const unsigned off, const unsigned row_size)
{
  const uint *restrict buf = data->ptr, *buf_end = buf+data->n;
  while(buf!=buf_end) {
    const uint *msg_end = buf+3+buf[2]; buf+=3;
    while(buf!=msg_end)
      memcpy(out, buf+off, size), out+=size, buf+=row_size;
  }
}

static void unpack_int(
  char *restrict out, const unsigned size, const unsigned p_off,
  const buffer *const data, const unsigned row_size, int set_src)
{
  const unsigned after = p_off + sizeof(uint), after_len = size-after;
  const uint *restrict buf = data->ptr, *buf_end = buf+data->n;
  const unsigned pi = set_src ? 1:0;
  while(buf!=buf_end) {
    const uint p=buf[pi], *msg_end = buf+3+buf[2]; buf+=3;
    while(buf!=msg_end) {
      memcpy(out,buf,p_off);
      memcpy(out+p_off,&p,sizeof(uint));
      memcpy(out+after,(const char *)buf+p_off,after_len);
      out+=size, buf+=row_size;
    }
  }
}

static uint num_rows(const buffer *const data, const unsigned row_size)
{
  const uint *buf = data->ptr, *buf_end = buf + data->n;
  uint n=0;
  while(buf!=buf_end) { uint len=buf[2]; n+=len, buf+=len+3; }
  return n/row_size;
}

static uint cap_rows(buffer *const data, const unsigned row_size,const uint max)
{
  uint *buf = data->ptr, *buf_end = buf + data->n;
  const uint maxn = max*row_size;
  uint n=0;
  while(buf!=buf_end) {
    uint len=buf[2]; n+=len;
    if(n<maxn) buf+=len+3;
    else {
      buf[2]-=(maxn-n); data->n = (buf-(uint*)data->ptr)+3+buf[2];
      buf+=len+3;
      while(buf!=buf_end) { uint len=buf[2]; n+=len, buf+=len+3; }
      break;
    }
  }
  return n/row_size;
}

/* An must be >= 1 */
uint sarray_transfer_many(
  struct array *const *const A, const unsigned *const size, const unsigned An,
  const int fixed, const int ext, const int set_src, const unsigned p_off,
  const uint *const restrict proc, const unsigned proc_stride,
  struct crystal *const cr)
{
  uint n, *perm;
  unsigned i,row_size,off,off1;

  off1 = size[0];
  if(!ext) off1 -= sizeof(uint);
  row_size=off1; for(i=1;i<An;++i) row_size += size[i];
  row_size = (row_size+sizeof(uint)-1)/sizeof(uint);
  
  perm = sortp(&cr->work,0, proc,A[0]->n,proc_stride);

  if(!ext) pack_int(&cr->data, row_size, cr->comm.id, A[0]->ptr,A[0]->n,size[0],
                    p_off, perm);
  else     pack_ext(&cr->data, row_size, cr->comm.id, A[0]->ptr,A[0]->n,size[0],
                    proc,proc_stride, perm);
  for(off=off1,i=1;i<An;++i) if(size[i])
    pack_more(&cr->data,off,row_size, A[i]->ptr,size[i], perm),off+=size[i];
    
  crystal_router(cr);
  
  if(!fixed) {
    n = num_rows(&cr->data,row_size);
    for(i=0;i<An;++i)
      array_reserve_(A[i],n,size[i],__FILE__,__LINE__), A[i]->n=n;
  } else {
    uint max=A[0]->max, an;
    for(i=1;i<An;++i) if(A[i]->max<max) max=A[i]->max;
    n = cap_rows(&cr->data,row_size, max);
    an = n>max?max:n;
    for(i=0;i<An;++i) A[i]->n=an;
  }
  
  if(!ext) unpack_int (A[0]->ptr,size[0],p_off, &cr->data,  row_size, set_src);
  else     unpack_more(A[0]->ptr,size[0],       &cr->data,0,row_size);
  for(off=off1,i=1;i<An;++i) if(size[i])
    unpack_more(A[i]->ptr,size[i], &cr->data,off,row_size),off+=size[i];
    
  return n;
}
  

void sarray_transfer_(struct array *const A, const unsigned size,
                      const unsigned p_off, const int set_src,
                      struct crystal *const cr)
{
  sarray_transfer_many(&A,&size,1, 0,0,set_src,p_off,
                       (uint*)((char*)A->ptr+p_off),size, cr);
}

void sarray_transfer_ext_(struct array *const A, const unsigned size,
                          const uint *const proc, const unsigned proc_stride,
                          struct crystal *const cr)
{
  sarray_transfer_many(&A,&size,1, 0,1,0,0, proc,proc_stride, cr);
}

