#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"
#include "comm.h"
#include "crystal.h"
#include "sort.h"
#include "sarray_sort.h"
#include "sarray_transfer.h"

/*--------------------------------------------------------------------------

  FORTRAN Interface to crystal router
   
  integer h, np
  MPI_Comm comm
  call crystal_setup(h,comm,np)  ! set h to handle to new instance
  ! it is a runtime error if MPI_Comm_size gives a value different than np
  call crystal_free(h)         ! release instance

  integer*? ituple(m,max)   ! integer type matching sint from "types.h"
  call crystal_ituple_transfer(h, ituple,m,n,max, kp)
    - moves each column ituple(:,i), 1 <= i <= n,
      to proc ituple(kp,i)
    - sets n to the number of columns received,
      which may be larger than max (indicating loss of n-max columns)
    - also sets ituple(kp,i) to the source proc of column ituple(:,i)

  call crystal_ituple_sort(h, ituple,m,n, key,nkey)
    - locally sorts columns ituple(:,1...n) in ascending order,
      ranked by ituple(key(1),i),
           then ituple(key(2),i),
           ...
           then ituple(key(nkey),i)
    - no communication; h used for scratch area
    - linear time
    - assumes nonnegative integers

  integer*? vi(mi,max)   ! integer type matching sint  from "types.h"
  integer*? vl(ml,max)   ! integer type matching slong from "types.h"
  real      vr(mr,max)
  call crystal_tuple_transfer(h,n,max, vi,mi,vl,ml,vr,mr, kp)
    - moves each column vi(:,i),vl(:,i),vr(:,i) 1 <= i <= n,
      to proc vi(kp,i)
    - sets n to the number of columns received,
      which may be larger than max (indicating loss of n-max columns)
    - also sets vi(kp,i) to the source proc of columns vi(:,i),vl(:,i),vr(:,i)

  call crystal_tuple_sort(h,n, vi,mi,vl,ml,vr,mr, key,nkey)
    - locally sorts columns vi/vl/vr (:,1...n) in ascending order,
      ranked by vi(key(1),i) [ or vl(key(1)-mi,i) if key(1)>mi ],
           then vi(key(2),i) [ or vl(key(2)-mi,i) if key(2)>mi ],
           ...
           then vi(key(nkey),i) or vl(key(nkey)-mi,i)
    - no communication; h used for scratch area
    - linear time
    - assumes nonnegative integers
    - sorting on reals not yet implemented

  --------------------------------------------------------------------------*/

#undef   crystal_free
#define ccrystal_free  PREFIXED_NAME(crystal_free)

#define fcrystal_setup           \
  FORTRAN_NAME(crystal_setup          ,CRYSTAL_SETUP          )
#define fcrystal_ituple_sort     \
  FORTRAN_NAME(crystal_ituple_sort    ,CRYSTAL_ITUPLE_SORT    )
#define fcrystal_tuple_sort      \
  FORTRAN_NAME(crystal_tuple_sort     ,CRYSTAL_TUPLE_SORT     )
#define fcrystal_ituple_transfer \
  FORTRAN_NAME(crystal_ituple_transfer,CRYSTAL_ITUPLE_TRANSFER)
#define fcrystal_tuple_transfer  \
  FORTRAN_NAME(crystal_tuple_transfer ,CRYSTAL_TUPLE_TRANSFER )
#define fcrystal_free            \
  FORTRAN_NAME(fcrystal_free           ,FCRYSTAL_FREE           )

static struct crystal **handle_array = 0;
static int handle_max = 0;
static int handle_n = 0;

void fcrystal_setup(sint *handle, const sint *np)
{
  struct crystal *p;
  if(handle_n==handle_max)
    handle_max+=handle_max/2+1,
    handle_array=trealloc(struct crystal*,handle_array,handle_max);
  handle_array[handle_n]=p=tmalloc(struct crystal,1);
  //comm_init_check(&p->comm, *comm, *np);
  buffer_init(&p->data,1000);
  buffer_init(&p->work,1000);
  *handle = handle_n++;
}

#define CHECK_HANDLE(func) do \
  if(*handle<0 || *handle>=handle_n || !handle_array[*handle]) \
    fail(1,__FILE__,__LINE__,func ": invalid handle"); \
while(0)

void fcrystal_ituple_sort(const sint *handle,
                          sint A[], const sint *m, const sint *n,
                          const sint keys[], const sint *nkey)
{
  const size_t size = (*m)*sizeof(sint);
  sint nk = *nkey;
  buffer *buf;
  CHECK_HANDLE("crystal_ituple_sort");
  buf = &handle_array[*handle]->data;
  if(--nk>=0) {
    sortp(buf,0, (uint*)&A[keys[nk]-1],*n,size);
    while(--nk>=0)
      sortp(buf,1, (uint*)&A[keys[nk]-1],*n,size);
    sarray_permute_buf_(ALIGNOF(sint),size,A,*n, buf);
  }
}

void fcrystal_tuple_sort(const sint *const handle, const sint *const n,
                         sint   Ai[], const sint *const mi,
                         slong  Al[], const sint *const ml,
                         double Ad[], const sint *const md,
                         const sint keys[], const sint *const nkey)
{
  const size_t size_i = (*mi)*sizeof(sint),
               size_l = (*ml)*sizeof(slong),
               size_d = (*md)*sizeof(double);
  int init=0;
  sint nk = *nkey;
  buffer *buf;
  CHECK_HANDLE("crystal_tuple_sort");
  buf = &handle_array[*handle]->data;
  if(nk<=0) return;
  while(--nk>=0) {
    sint k = keys[nk]-1;
    if(k<0 || k>=*mi+*ml)
      fail(1,__FILE__,__LINE__,"crystal_tuple_sort: invalid key");
    else if(k<*mi) sortp     (buf,init, (uint *)&Ai[k],    *n,size_i);
    else           sortp_long(buf,init, (ulong*)&Al[k-*mi],*n,size_l);
    init=1;
  }
  if(*mi) sarray_permute_buf_(ALIGNOF(sint  ),size_i,Ai,*n, buf);
  if(*ml) sarray_permute_buf_(ALIGNOF(slong ),size_l,Al,*n, buf);
  if(*md) sarray_permute_buf_(ALIGNOF(double),size_d,Ad,*n, buf);
}

void fcrystal_ituple_transfer(const sint *handle,
                              sint A[], const sint *m, sint *n,
                              const sint *nmax, const sint *proc_key)
{
  struct array ar, *const ar_ptr = &ar;
  const unsigned size=(*m)*sizeof(sint);
  CHECK_HANDLE("crystal_ituple_transfer");
  ar.ptr=A, ar.n=*n, ar.max=*nmax;
  *n = sarray_transfer_many(&ar_ptr,&size,1, 1,0,1,(*proc_key-1)*sizeof(sint),
         (uint*)&A[*proc_key-1],size, handle_array[*handle]);
}

void fcrystal_tuple_transfer(
  const sint *const handle, sint *const n, const sint *const max,
  sint   Ai[], const sint *const mi,
  slong  Al[], const sint *const ml,
  double Ad[], const sint *const md,
  const sint *const proc_key)
{
  struct array ar_i, ar_l, ar_d, *ar[3];
  unsigned size[3];
  CHECK_HANDLE("crystal_tuple_transfer");
  size[0]=*mi*sizeof(sint);
  size[1]=*ml*sizeof(slong);
  size[2]=*md*sizeof(double);
  ar[0]=&ar_i, ar[1]=&ar_l, ar[2]=&ar_d;
  ar_i.ptr=Ai,ar_l.ptr=Al,ar_d.ptr=Ad;
  ar_i.n=ar_l.n=ar_d.n = *n;
  ar_i.max=ar_l.max=ar_d.max=*max;
  *n = sarray_transfer_many(ar,size,3, 1,0,1,(*proc_key-1)*sizeof(sint),
         (uint*)&Ai[*proc_key-1],size[0], handle_array[*handle]);
}

void fcrystal_free(sint *handle)
{
  CHECK_HANDLE("crystal_free");
  ccrystal_free(handle_array[*handle]);
  free(handle_array[*handle]);
  handle_array[*handle] = 0;
}


