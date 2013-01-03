#ifndef MEM_H
#define MEM_H

/* requires:
     <stddef.h> for size_t, offsetof
     <stdlib.h> for malloc, calloc, realloc, free
     <string.h> for memcpy
     "c99.h"
     "fail.h"
*/

#if !defined(C99_H) || !defined(FAIL_H)
#error "mem.h" requires "c99.h" and "fail.h"
#endif

/* 
   All memory management goes through the wrappers defined in this
   header. Diagnostics can be turned on with
     -DPRINT_MALLOCS=1
   Then all memory management operations will be printed to stdout.
   
   Most memory management occurs through use of the "array" type,
   defined below, which defines a generic dynamically-sized array
   that grows in bursts. The "buffer" type is a "char" array and
   is often passed around by code to provide a common area for
   scratch work.
*/

#ifndef PRINT_MALLOCS
#  define PRINT_MALLOCS 0
#else
#  include <stdio.h>
#  ifndef comm_gbl_id
#    define comm_gbl_id PREFIXED_NAME(comm_gbl_id)
#    define comm_gbl_np PREFIXED_NAME(comm_gbl_np)
#    include "types.h"
     extern uint comm_gbl_id, comm_gbl_np;
#  endif
#endif

/*--------------------------------------------------------------------------
   Memory Allocation Wrappers to Catch Out-of-memory
  --------------------------------------------------------------------------*/

static inline void *smalloc(size_t size, const char *file, unsigned line)
{
  void *restrict res = malloc(size);
  #if PRINT_MALLOCS
  fprintf(stdout,"MEM: proc %04d: %p = malloc(%ld) @ %s(%u)\n",
          (int)comm_gbl_id,res,(long)size,file,line), fflush(stdout);
  #endif
  if(!res && size)
    fail(1,file,line,"allocation of %ld bytes failed\n",(long)size);
  return res;
}

static inline void *scalloc(
  size_t nmemb, size_t size, const char *file, unsigned line)
{
  void *restrict res = calloc(nmemb, size);
  #if PRINT_MALLOCS
  fprintf(stdout,"MEM: proc %04d: %p = calloc(%ld) @ %s(%u)\n",
          (int)comm_gbl_id,res,(long)size*nmemb,file,line), fflush(stdout);
  #endif
  if(!res && nmemb)
    fail(1,file,line,"allocation of %ld bytes failed\n",
           (long)size*nmemb);
  return res;
}

static inline void *srealloc(
  void *restrict ptr, size_t size, const char *file, unsigned line)
{
  void *restrict res = realloc(ptr, size);
  #if PRINT_MALLOCS
  if(res!=ptr) {
    if(ptr)
      fprintf(stdout,"MEM: proc %04d: %p freed by realloc @ %s(%u)\n",
              (int)comm_gbl_id,ptr,file,line), fflush(stdout);
    fprintf(stdout,"MEM: proc %04d: %p = realloc of %p to %lu @ %s(%u)\n",
            (int)comm_gbl_id,res,ptr,(long)size,file,line), fflush(stdout);
  } else
    fprintf(stdout,"MEM: proc %04d: %p realloc'd to %lu @ %s(%u)\n",
            (int)comm_gbl_id,res,(long)size,file,line), fflush(stdout);
  #endif
  if(!res && size)
    fail(1,file,line,"allocation of %ld bytes failed\n",(long)size);
  return res;
}

#define tmalloc(type, count) \
  ((type*) smalloc((count)*sizeof(type),__FILE__,__LINE__) )
#define tcalloc(type, count) \
  ((type*) scalloc((count),sizeof(type),__FILE__,__LINE__) )
#define trealloc(type, ptr, count) \
  ((type*) srealloc((ptr),(count)*sizeof(type),__FILE__,__LINE__) )

#if PRINT_MALLOCS
static inline void sfree(void *restrict ptr, const char *file, unsigned line)
{
  free(ptr);
  fprintf(stdout,"MEM: proc %04d: %p freed @ %s(%u)\n",
          (int)comm_gbl_id,ptr,file,line), fflush(stdout);
}
#define free(x) sfree(x,__FILE__,__LINE__)
#endif

/*--------------------------------------------------------------------------
   A dynamic array
  --------------------------------------------------------------------------*/
struct array { void *ptr; size_t n,max; };
#define null_array {0,0,0}
static void array_init_(struct array *a, size_t max, size_t size,
                        const char *file, unsigned line)
{
  a->n=0, a->max=max, a->ptr=smalloc(max*size,file,line);
}
static void array_resize_(struct array *a, size_t max, size_t size,
                          const char *file, unsigned line)
{
  a->max=max, a->ptr=srealloc(a->ptr,max*size,file,line);
}
static void *array_reserve_(struct array *a, size_t min, size_t size,
                            const char *file, unsigned line)
{
  size_t max = a->max;
  if(max<min) {
    max+=max/2+1;
    if(max<min) max=min;
    array_resize_(a,max,size,file,line);
  }
  return a->ptr;
}

#define array_free(a) (free((a)->ptr))
#define array_init(T,a,max) array_init_(a,max,sizeof(T),__FILE__,__LINE__)
#define array_resize(T,a,max) array_resize_(a,max,sizeof(T),__FILE__,__LINE__)
#define array_reserve(T,a,min) array_reserve_(a,min,sizeof(T),__FILE__,__LINE__)

static void array_cat_(size_t size, struct array *d, const void *s, size_t n,
                       const char *file, unsigned line)
{
  char *out = array_reserve_(d,d->n+n,size, file,line);
  memcpy(out+d->n*size, s, n*size);
  d->n+=n;
}

#define array_cat(T,d,s,n) array_cat_(sizeof(T),d,s,n,__FILE__,__LINE__)

/*--------------------------------------------------------------------------
   Buffer = char array
  --------------------------------------------------------------------------*/
typedef struct array buffer;
#define null_buffer null_array
#define buffer_init(b,max) array_init(char,b,max)
#define buffer_resize(b,max) array_resize(char,b,max)
#define buffer_reserve(b,max) array_reserve(char,b,max)
#define buffer_free(b) array_free(b)

/*--------------------------------------------------------------------------
   Alignment routines
  --------------------------------------------------------------------------*/
#define ALIGNOF(T) offsetof(struct { char c; T x; }, x)
static size_t align_as_(size_t a, size_t n) { return (n+a-1)/a*a; }
#define align_as(T,n) align_as_(ALIGNOF(T),n)
#define align_ptr(T,base,offset) ((T*)((char*)(base)+align_as(T,offset)))
#endif

