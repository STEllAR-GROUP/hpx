#ifndef SARRAY_TRANSFER_H
#define SARRAY_TRANSFER_H

#if !defined(CRYSTAL_H)
#warning "sarray_transfer.h" requires "crystal.h"
#endif

/*
  High-level interface for the crystal router.
  Given an array of structs, transfers each to the process indicated
  by a field of the struct, which gets set to the source process on output.
  
  For the dynamic "array" type, see "mem.h".
  
  Requires a "crystal router" object:
  
    struct comm c;
    struct crystal cr;
    
    comm_init(&c, MPI_COMM_WORLD);
    crystal_init(&cr, &c);
    
  Example sarray_transfer usage:
  
    struct T { ...; uint proc; ...; };
    struct array A = null_array;
    struct T *p, *e;
    
    // resize A to 100 struct T's, fill up with data
    p = array_reserve(struct T, &A, 100), A.n=100;
    for(e=p+A.n;p!=e;++p) {
      ...
      p->proc = ...;
      ...
    }
    
    // array A represents the array
    //   struct T ar[A.n]    where &ar[0] == A.ptr
    // transfer ar[i] to processor ar[i].proc  for each i=0,...,A.n-1:
    
    sarray_transfer(struct T, A, proc,set_src, &cr);
    
    // now array A represents a different array with a different size
    //   struct T ar[A.n]    where &ar[0] == A.ptr
    // the ordering is arbitrary
    // if set_src != 0, ar[i].proc is set to the proc where ar[i] came from
    // otherwise ar[i].proc is unchanged (and == this proc id)
    
    // note: two calls of
    sarray_transfer(struct T, A, proc,1, &cr);
    // in a row should return A to its original state, up to ordering
 
  Cleanup:
    array_free(&A);
    crystal_free(&cr);
    comm_free(&c);

  Example sarray_transfer_ext usage:
  
    struct T { ... };
    struct array A;
    uint proc[A.n];
    
    // array A represents the array
    //   struct T ar[A.n]    where &ar[0] == A.ptr
    // transfer ar[i] to processor proc[i]  for each i=0,...,A.n-1:
    sarray_transfer_ext(struct T, &A, proc, &cr);
    
    // no information is available now on where each struct came from

*/

#define sarray_transfer_many PREFIXED_NAME(sarray_transfer_many)
#define sarray_transfer_     PREFIXED_NAME(sarray_transfer_    )
#define sarray_transfer_ext_ PREFIXED_NAME(sarray_transfer_ext_)

uint sarray_transfer_many(
  struct array *const *const A, const unsigned *const size, const unsigned An,
  const int fixed, const int ext, const int set_src, const unsigned p_off,
  const uint *const restrict proc, const unsigned proc_stride,
  struct crystal *const cr);
void sarray_transfer_(struct array *const A, const unsigned size,
                      const unsigned p_off, const int set_src,
                      struct crystal *const cr);
void sarray_transfer_ext_(struct array *const A, const unsigned size,
                          const uint *const proc, const unsigned proc_stride,
                          struct crystal *const cr);

#define sarray_transfer(T,A,proc_field,set_src,cr) \
  sarray_transfer_(A,sizeof(T),offsetof(T,proc_field),set_src,cr)

#define sarray_transfer_ext(T,A,proc,proc_stride,cr) \
  sarray_transfer_ext_(A,sizeof(T),proc,proc_stride,cr)

#endif
