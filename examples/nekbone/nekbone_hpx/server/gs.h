#ifndef GS_H
#define GS_H

#if !defined(COMM_H) || !defined(GS_DEFS_H) || !defined(MEM_H)
#warning "gs.h" requires "comm.h", "gs_defs.h", and "mem.h"
#endif

/*
  Gather/Scatter Library

  The code
  
    struct comm c;  // see "comm.h"
    slong id[n];    // the slong type is defined in "types.h"
    ...
    struct gs_data *g = gs_setup(id,n, &c, 0,gs_auto,1);
    
  defines a partition of the set of (processor, local index) pairs,
    (p,i) \in S_j  iff   abs(id[i]) == j  on processor p
  That is, all (p,i) pairs are grouped together (in group S_j) that have the
    same id (=j).
  S_0 is treated specially --- it is ignored completely
    (i.e., when id[i] == 0, local index i does not participate in any
    gather/scatter operation
  If id[i] on proc p is negative then the pair (p,i) is "flagged". This
  determines the non-symmetric behavior. For the simpler, symmetric case,
  all id's should be positive.
  
  The second to last argument to gs_setup is the method to use, one of
    gs_pairwise, gs_crystal_router, gs_all_reduce, gs_auto
  The method "gs_auto" tries ~10 runs of each and chooses the fastest.
  For a single-use handle, it makes more sense to use "gs_crystal_router".
  
  When "g" is no longer needed, free it with
  
    gs_free(g);
  
  A basic gather/scatter operation is, e.g.,
  
    double v[n]; buffer buf;  // see "mem.h" for "buffer"
    ...
    gs(v, gs_double,gs_add, 0, g,&buf);
    
  The buffer pointer can be null, in which case, a static buffer is used,
  shared across all gs handles.
  This gs call has the effect, (in the simple, symmetric, unflagged case)
  
    v[i] <--  \sum_{ (p,j) \in S_{id[i]} } v_(p) [j]
    
  where v_(p) [j] means v[j] on proc p. In other words, every v[i] is replaced
  by the sum of all v[j]'s with the same id, given by id[i]. This accomplishes
  "direct stiffness summation" corresponding to the action of QQ^T, where
  "Q" is a boolean matrix that copies from a global vector (indexed by id)
  to the local vectors indexed by (p,i) pairs.
  
  Summation on doubles is not the only operation and datatype supported. The
  full list is defined in "gs_defs.h", and includes the operations
    gs_add, gs_mul, gs_max, gs_min
  and datatypes
    gs_double, gs_float, gs_int, gs_long, gs_sint, gs_slong.
  (The int and long types are the plain C types, whereas sint and slong
   are defined in "types.h").
   
  For the nonsymmetric behavior, the "transpose" parameter is important:
  
    gs(v, gs_double,gs_add, transpose, g,&buf);
    
  When transpose == 0, any "flagged" (p,i) pairs (id[i] negative on p)
  do not participate in the sum, but *do* still receive the sum on output.
  As a special case, when only one (p,i) pair is unflagged per group this
  corresponds to the rectangular "Q" matrix referred to above.
  
  When transpose == 1, the "flagged" (p,i) pairs *do* participate in the sum,
  but do *not* get set on output. In the special case of only one unflagged
  (p,i) pair, this corresponds to the transpose of "Q" referred to above.



  A version for vectors (contiguously packed) is, e.g.,
  
    double v[n][k];
    gs_vec(v,k, gs_double,gs_add, transpose, g,&buf);
  
  which is like "gs" operating on the datatype double[k],
  with summation here being vector summation. Number of messages sent
  is independent of k.
  
  For combining the communication for "gs" on multiple arrays:
  
    double v1[n], v2[n], ..., vk[n];
    double (*vs)[k] = {v1, v2, ..., vk};
    
    gs_many(vs,k, gs_double,op, t, g,&buf);
  
  This call is equivalent to
  
    gs(v1, gs_double,op, t, g, &buf);
    gs(v2, gs_double,op, t, g, &buf);
    ...
    gs(vk, gs_double,op, t, g, &buf);
    
  except that all communication is done together.
  


  Finally, gs_unique has the same basic signature as gs_setup:
  
    gs_unique(id,n, &c);
    
  This call modifies id, "flagging" (by negating id[i]) all (p,i) pairs in
  each group except one. The sole "unflagged" member of the group is chosen
  in an arbitrary but consistent way. If the "unique" flag is set when
  calling gs_setup, the behavior is equivalent to first calling gs_unique,
  except that the id array is left unmodified.
  

*/  

#define gs         PREFIXED_NAME(gs       )
#define gs_vec     PREFIXED_NAME(gs_vec   )
#define gs_many    PREFIXED_NAME(gs_many  )
#define gs_setup   PREFIXED_NAME(gs_setup )
#define gs_free    PREFIXED_NAME(gs_free  )
#define gs_unique  PREFIXED_NAME(gs_unique)

struct gs_data;
typedef enum { gs_pairwise, gs_crystal_router, gs_all_reduce,
               gs_auto } gs_method;

void gs(void *u, gs_dom dom, gs_op op, unsigned transpose,
        struct gs_data *gsh, buffer *buf);
void gs_vec(void *u, unsigned vn, gs_dom dom, gs_op op,
            unsigned transpose, struct gs_data *gsh, buffer *buf);
void gs_many(void *const*u, unsigned vn, gs_dom dom, gs_op op,
             unsigned transpose, struct gs_data *gsh, buffer *buf);
struct gs_data *gs_setup(const slong *id, uint n, const struct comm *comm,
                         int unique, gs_method method, int verbose);
void gs_free(struct gs_data *gsh);
void gs_unique(slong *id, uint n, const struct comm *comm);

#endif
