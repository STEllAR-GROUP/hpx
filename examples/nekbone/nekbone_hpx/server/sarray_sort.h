#ifndef SARRAY_SORT_H
#define SARRAY_SORT_H

#if !defined(SORT_H)
#warning "sarray_sort.h" requires "sort.h"
#endif

/*------------------------------------------------------------------------------
  
  Array of Structs Sort
  
  buffer *buf;
  typedef struct { ... } T;
  T A[n];

  sarray_sort(T,A,n, field_name,is_long, buf)
    - sort A according to the struct field "field_name",
      which is a ulong/uint field according as is_long is true/false

  sarray_sort_2(T,A,n, field1,is_long1, field2,is_long2, buf)
    - sort A by field1 then field2

  sarray_permute(T,A,n, perm, work)
    - permute A  (in-place)
      A[0] <- A[perm[0]], etc.
      work needs to hold sizeof(T) bytes  (i.e., 1 T)

  sarray_permute_buf(T,A,n, buf);
    - permute A according to the permutation in buf
      A[0] <- A[perm[0]], etc.
      where uint *perm = buf->ptr   (see "sort.h")

  ----------------------------------------------------------------------------*/


#define sarray_permute_     PREFIXED_NAME(sarray_permute_)
#define sarray_permute_buf_ PREFIXED_NAME(sarray_permute_buf_)

void sarray_permute_(size_t size, void *A, size_t n, uint *perm, void *work);
void sarray_permute_buf_(
  size_t align, size_t size, void *A, size_t n, buffer *buf);

#define sarray_permute(T,A,n, perm, work) \
  sarray_permute_(sizeof(T),A,n, perm, work)
#define sarray_permute_buf(T,A,n, buf) \
  sarray_permute_buf_(ALIGNOF(T),sizeof(T),A,n,buf)

#define sarray_sort_field(T,A,n, field,is_long, buf,keep) do { \
  if(is_long) \
    sortp_long(buf,keep, (ulong*)((char*)(A)+offsetof(T,field)),n,sizeof(T)); \
  else \
    sortp     (buf,keep, (uint *)((char*)(A)+offsetof(T,field)),n,sizeof(T)); \
} while (0)

#define sarray_sort(T,A,n, field,is_long, buf) do { \
  sarray_sort_field(T,A,n, field,is_long, buf,0); \
  sarray_permute_buf(T,A,n, buf); \
} while (0)

#define sarray_sort_2(T,A,n, field1,is_long1, field2,is_long2, buf) do { \
  sarray_sort_field(T,A,n, field2,is_long2, buf,0); \
  sarray_sort_field(T,A,n, field1,is_long1, buf,1); \
  sarray_permute_buf(T,A,n, buf); \
} while (0)

#define sarray_sort_3(T,A,n, field1,is_long1, field2,is_long2, \
                             field3,is_long3, buf) do { \
  sarray_sort_field(T,A,n, field3,is_long3, buf,0); \
  sarray_sort_field(T,A,n, field2,is_long2, buf,1); \
  sarray_sort_field(T,A,n, field1,is_long1, buf,1); \
  sarray_permute_buf(T,A,n, buf); \
} while (0)

#define sarray_sort_4(T,A,n, field1,is_long1, field2,is_long2, \
                             field3,is_long3, field4,is_long4, buf) do { \
  sarray_sort_field(T,A,n, field4,is_long4, buf,0); \
  sarray_sort_field(T,A,n, field3,is_long3, buf,1); \
  sarray_sort_field(T,A,n, field2,is_long2, buf,1); \
  sarray_sort_field(T,A,n, field1,is_long1, buf,1); \
  sarray_permute_buf(T,A,n, buf); \
} while (0)

static void sarray_perm_invert(
  uint *const pinv, const uint *const perm, const uint n)
{
  uint i; for(i=0;i<n;++i) pinv[perm[i]] = i;
}

#endif
