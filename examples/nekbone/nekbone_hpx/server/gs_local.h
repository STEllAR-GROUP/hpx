#ifndef GS_LOCAL_H
#define GS_LOCAL_H

#if !defined(NAME_H) || !defined(TYPES_H) || !defined(GS_DEFS_H)
#warning "gs_local.h" requires "name.h", "types.h", and "gs_defs.h"
#endif

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

void gs_gather_array(void *out, const void *in, uint n,
                     gs_dom dom, gs_op op);
void gs_init_array(void *out, uint n, gs_dom dom, gs_op op);

typedef void gs_gather_fun(
  void *out, const void *in, const unsigned vn,
  const uint *map, gs_dom dom, gs_op op);
typedef void gs_scatter_fun(
  void *out, const void *in, const unsigned vn,
  const uint *map, gs_dom dom);
typedef void gs_init_fun(
  void *out, const unsigned vn,
  const uint *map, gs_dom dom, gs_op op);

extern gs_gather_fun gs_gather, gs_gather_vec, gs_gather_many,
                     gs_gather_vec_to_many;
extern gs_scatter_fun gs_scatter, gs_scatter_vec, gs_scatter_many,
                      gs_scatter_many_to_vec, gs_scatter_vec_to_many;
extern gs_init_fun gs_init, gs_init_vec, gs_init_many;

#endif
