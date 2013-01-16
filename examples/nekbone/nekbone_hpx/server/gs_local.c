#include <string.h>
#include <limits.h>
#include <float.h>
#include "c99.h"
#include "name.h"
#include "types.h"

#define gs_gather_array        PREFIXED_NAME(gs_gather_array       )
#define gs_init_array          PREFIXED_NAME(gs_init_array         )
#define gs_gather              PREFIXED_NAME(gs_gather             )
#define gs_scatter             PREFIXED_NAME(gs_scatter            )
#define gs_init                PREFIXED_NAME(gs_init               )
#define gs_gather_vec          PREFIXED_NAME(gs_gather_vec         )
#define gs_scatter_vec         PREFIXED_NAME(gs_scatter_vec        )
#define gs_init_vec            PREFIXED_NAME(gs_init_vec           )
#define gs_gather_many         PREFIXED_NAME(gs_gather_many        )
#define gs_scatter_many        PREFIXED_NAME(gs_scatter_many       )
#define gs_init_many           PREFIXED_NAME(gs_init_many          )
#define gs_gather_vec_to_many  PREFIXED_NAME(gs_gather_vec_to_many )
#define gs_scatter_many_to_vec PREFIXED_NAME(gs_scatter_many_to_vec)
#define gs_scatter_vec_to_many PREFIXED_NAME(gs_scatter_vec_to_many)

#include "gs_defs.h"
GS_DEFINE_IDENTITIES()
GS_DEFINE_DOM_SIZES()

/*------------------------------------------------------------------------------
  The array gather kernel
------------------------------------------------------------------------------*/
#define DEFINE_GATHER(T,OP) \
static void gather_array_##T##_##OP( \
  T *restrict out, const T *restrict in, uint n) \
{                                                                \
  for(;n;--n) { T q = *in++, *p = out++; GS_DO_##OP(*p,q); }      \
}

/*------------------------------------------------------------------------------
  The array initialization kernel
------------------------------------------------------------------------------*/
#define DEFINE_INIT(T) \
static void init_array_##T(T *restrict out, uint n, gs_op op) \
{                                                             \
  const T e = gs_identity_##T[op];                            \
  for(;n;--n) *out++=e;                                       \
}

#define DEFINE_PROCS(T) \
  GS_FOR_EACH_OP(T,DEFINE_GATHER) \
  DEFINE_INIT(T)

GS_FOR_EACH_DOMAIN(DEFINE_PROCS)

#undef DEFINE_PROCS
#undef DEFINE_INIT
#undef DEFINE_GATHER

/*------------------------------------------------------------------------------
  The basic gather kernel
------------------------------------------------------------------------------*/
#define DEFINE_GATHER(T,OP) \
static void gather_##T##_##OP( \
  T *restrict out, const T *restrict in, const unsigned in_stride,           \
  const uint *restrict map)                                                  \
{                                                                            \
  uint i,j;                                                                  \
  while((i=*map++)!=-(uint)1) {                                              \
    T t=out[i];                                                              \
    j=*map++; do GS_DO_##OP(t,in[j*in_stride]); while((j=*map++)!=-(uint)1); \
    out[i]=t;                                                                \
  }                                                                          \
}

/*------------------------------------------------------------------------------
  The basic scatter kernel
------------------------------------------------------------------------------*/
#define DEFINE_SCATTER(T) \
static void scatter_##T( \
  T *restrict out, const unsigned out_stride,                      \
  const T *restrict in, const unsigned in_stride,                  \
  const uint *restrict map)                                        \
{                                                                  \
  uint i,j;                                                        \
  while((i=*map++)!=-(uint)1) {                                    \
    T t=in[i*in_stride];                                           \
    j=*map++; do out[j*out_stride]=t; while((j=*map++)!=-(uint)1); \
  }                                                                \
}

/*------------------------------------------------------------------------------
  The basic initialization kernel
------------------------------------------------------------------------------*/
#define DEFINE_INIT(T) \
static void init_##T(T *restrict out, const uint *restrict map, gs_op op) \
{                                                       \
  uint i; const T e = gs_identity_##T[op];              \
  while((i=*map++)!=-(uint)1) out[i]=e;                 \
}

#define DEFINE_PROCS(T) \
  GS_FOR_EACH_OP(T,DEFINE_GATHER) \
  DEFINE_SCATTER(T) \
  DEFINE_INIT(T)

GS_FOR_EACH_DOMAIN(DEFINE_PROCS)

#undef DEFINE_PROCS
#undef DEFINE_INIT
#undef DEFINE_SCATTER
#undef DEFINE_GATHER

/*------------------------------------------------------------------------------
  The vector gather kernel
------------------------------------------------------------------------------*/
#define DEFINE_GATHER(T,OP) \
static void gather_vec_##T##_##OP( \
  T *restrict out, const T *restrict in, const unsigned vn,                  \
  const uint *restrict map)                                                  \
{                                                                            \
  uint i,j;                                                                  \
  while((i=*map++)!=-(uint)1) {                                              \
    T *restrict p = &out[i*vn], *pe = p+vn;                                  \
    j=*map++; do {                                                           \
      const T *restrict q = &in[j*vn];                                       \
      T *restrict pk=p; do { GS_DO_##OP(*pk,*q); ++pk, ++q; } while(pk!=pe); \
    } while((j=*map++)!=-(uint)1);                                           \
  }                                                                          \
}

/*------------------------------------------------------------------------------
  The vector scatter kernel
------------------------------------------------------------------------------*/
void gs_scatter_vec(
  void *restrict out, const void *restrict in, const unsigned vn,
  const uint *restrict map, gs_dom dom)
{
  unsigned unit_size = vn*gs_dom_size[dom];
  uint i,j;
  while((i=*map++)!=-(uint)1) {
    const char *t = (const char *)in + i*unit_size;
    j=*map++; do
      memcpy((char *)out+j*unit_size,t,unit_size);
    while((j=*map++)!=-(uint)1);
  }
}

/*------------------------------------------------------------------------------
  The vector initialization kernel
------------------------------------------------------------------------------*/
#define DEFINE_INIT(T) \
static void init_vec_##T(T *restrict out, const unsigned vn, \
                         const uint *restrict map, gs_op op) \
{                                                            \
  uint i; const T e = gs_identity_##T[op];                   \
  while((i=*map++)!=-(uint)1) {                              \
    T *restrict u = (T*)out + vn*i, *ue = u+vn;              \
    do *u++ = e; while(u!=ue);                               \
  }                                                          \
}

#define DEFINE_PROCS(T) \
  GS_FOR_EACH_OP(T,DEFINE_GATHER) \
  DEFINE_INIT(T)

GS_FOR_EACH_DOMAIN(DEFINE_PROCS)

#undef DEFINE_PROCS
#undef DEFINE_INIT
#undef DEFINE_GATHER

#undef DO_bpr
#undef DO_max
#undef DO_min
#undef DO_mul
#undef DO_add

#define SWITCH_DOMAIN_CASE(T) case gs_##T: WITH_DOMAIN(T); break;
#define SWITCH_DOMAIN(dom) do switch(dom) { \
    GS_FOR_EACH_DOMAIN(SWITCH_DOMAIN_CASE) case gs_dom_n: break; } while(0)

#define SWITCH_OP_CASE(T,OP) case gs_##OP: WITH_OP(T,OP); break;
#define SWITCH_OP(T,op) do switch(op) { \
    GS_FOR_EACH_OP(T,SWITCH_OP_CASE) case gs_op_n: break; } while(0)

/*------------------------------------------------------------------------------
  Array kernels
------------------------------------------------------------------------------*/
void gs_gather_array(void *out, const void *in, uint n, gs_dom dom, gs_op op)
{
#define WITH_OP(T,OP) gather_array_##T##_##OP(out,in,n)
#define WITH_DOMAIN(T) SWITCH_OP(T,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
#undef  WITH_OP
}

void gs_init_array(void *out, uint n, gs_dom dom, gs_op op)
{
#define WITH_DOMAIN(T) init_array_##T(out,n,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

/*------------------------------------------------------------------------------
  Plain kernels; vn parameter ignored but present for consistent signatures
------------------------------------------------------------------------------*/
void gs_gather(void *out, const void *in, const unsigned vn,
               const uint *map, gs_dom dom, gs_op op)
{
#define WITH_OP(T,OP) gather_##T##_##OP(out,in,1,map)
#define WITH_DOMAIN(T) SWITCH_OP(T,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
#undef  WITH_OP
}

void gs_scatter(void *out, const void *in, const unsigned vn,
                const uint *map, gs_dom dom)
{
#define WITH_DOMAIN(T) scatter_##T(out,1,in,1,map)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

void gs_init(void *out, const unsigned vn, const uint *map,
             gs_dom dom, gs_op op)
{
#define WITH_DOMAIN(T) init_##T(out,map,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

/*------------------------------------------------------------------------------
  Vector kernels
------------------------------------------------------------------------------*/
void gs_gather_vec(void *out, const void *in, const unsigned vn,
                   const uint *map, gs_dom dom, gs_op op)
{
#define WITH_OP(T,OP) gather_vec_##T##_##OP(out,in,vn,map)
#define WITH_DOMAIN(T) SWITCH_OP(T,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
#undef  WITH_OP
}

void gs_init_vec(void *out, const unsigned vn, const uint *map,
                 gs_dom dom, gs_op op)
{
#define WITH_DOMAIN(T) init_vec_##T(out,vn,map,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

/*------------------------------------------------------------------------------
  Multiple array kernels
------------------------------------------------------------------------------*/
void gs_gather_many(void *out, const void *in, const unsigned vn,
                    const uint *map, gs_dom dom, gs_op op)
{
  uint k;
  typedef void *ptr_to_void; typedef const void *ptr_to_const_void;
  const ptr_to_void *p = out; const ptr_to_const_void *q = in;
#define WITH_OP(T,OP) for(k=0;k<vn;++k) gather_##T##_##OP(p[k],q[k],1,map)
#define WITH_DOMAIN(T) SWITCH_OP(T,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
#undef  WITH_OP
}

void gs_scatter_many(void *out, const void *in, const unsigned vn,
                     const uint *map, gs_dom dom)
{
  uint k;
  typedef void *ptr_to_void; typedef const void *ptr_to_const_void;
  const ptr_to_void *p = out; const ptr_to_const_void *q = in;
#define WITH_DOMAIN(T) for(k=0;k<vn;++k) scatter_##T(p[k],1,q[k],1,map)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

void gs_init_many(void *out, const unsigned vn, const uint *map,
                  gs_dom dom, gs_op op)
{
  uint k;
  typedef void *ptr_to_void; const ptr_to_void *p = out;
#define WITH_DOMAIN(T) for(k=0;k<vn;++k) init_##T(p[k],map,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

/*------------------------------------------------------------------------------
  Gather from strided array -> multiple arrays
  Scatter from multiple arrays -> strided array,
  Scatter from strided array -> multiple arrays,
------------------------------------------------------------------------------*/
void gs_gather_vec_to_many(void *out, const void *in, const unsigned vn,
                           const uint *map, gs_dom dom, gs_op op)
{
  unsigned i; const unsigned unit_size = gs_dom_size[dom];
  typedef void *ptr_to_void;
  const ptr_to_void *p = out; const char *q = in;
#define WITH_OP(T,OP) \
  for(i=vn;i;--i) gather_##T##_##OP(*p++,(const T*)q,vn,map), q+=unit_size
#define WITH_DOMAIN(T) SWITCH_OP(T,op)
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
#undef  WITH_OP
}

void gs_scatter_many_to_vec(void *out, const void *in, const unsigned vn,
                            const uint *map, gs_dom dom)
{
  unsigned i; const unsigned unit_size = gs_dom_size[dom];
  typedef const void *ptr_to_const_void;
  char *p = out; const ptr_to_const_void *q = in;
#define WITH_DOMAIN(T) \
  for(i=vn;i;--i) scatter_##T((T*)p,vn,*q++,1,map), p+=unit_size
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

void gs_scatter_vec_to_many(void *out, const void *in, const unsigned vn,
                            const uint *map, gs_dom dom)
{
  unsigned i; const unsigned unit_size = gs_dom_size[dom];
  typedef void *ptr_to_void;
  const ptr_to_void *p = out; const char *q = in;
#define WITH_DOMAIN(T) \
  for(i=vn;i;--i) scatter_##T(*p++,1,(const T*)q,vn,map), q+=unit_size
  SWITCH_DOMAIN(dom);
#undef  WITH_DOMAIN
}

#undef SWITCH_OP
#undef SWITCH_OP_CASE
#undef SWITCH_DOMAIN
#undef SWITCH_DOMAIN_CASE
