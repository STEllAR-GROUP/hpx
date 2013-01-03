#ifndef TENSOR_H
#define TENSOR_H

#if !defined(TYPES_H) || !defined(NAME_H)
#warning "tensor.h" requires "types.h" and "name.h"
#endif

#if defined(USE_CBLAS)
#  include <cblas.h>
#  define tensor_dot(a,b,n) cblas_ddot((int)(n),a,1,b,1)
#  define tensor_mxv(y,ny,A,x,nx) \
     cblas_dgemv(CblasColMajor,CblasNoTrans,(int)ny,(int)nx, \
                 1.0,A,(int)ny,x,1,0.0,y,1)
#  define tensor_mtxv(y,ny,A,x,nx) \
     cblas_dgemv(CblasColMajor,CblasTrans,(int)nx,(int)ny, \
                 1.0,A,(int)nx,x,1,0.0,y,1)
#  define tensor_mxm(C,nc,A,na,B,nb) \
     cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, \
                 (int)nc,(int)nb,(int)na,1.0, \
                 A,(int)nc,B,(int)na,0.0,C,(int)nc)
#  define tensor_mtxm(C,nc,A,na,B,nb) \
     cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans, \
                 (int)nc,(int)nb,(int)na,1.0, \
                 A,(int)na,B,(int)na,0.0,C,(int)nc)
#else
#  define tensor_dot  PREFIXED_NAME(tensor_dot )
#  define tensor_mtxm PREFIXED_NAME(tensor_mtxm)
double tensor_dot(const double *a, const double *b, uint n);

/* C (nc x nb) = [A (na x nc)]^T * B (na x nb); all column-major */
void tensor_mtxm(double *C, uint nc,
                 const double *A, uint na, const double *B, uint nb);
#  if defined(USE_NAIVE_BLAS)
#    define tensor_mxv  PREFIXED_NAME(tensor_mxv )
#    define tensor_mtxv PREFIXED_NAME(tensor_mtxv)
#    define tensor_mxm  PREFIXED_NAME(tensor_mxm )
/* y = A x */
void tensor_mxv(double *y, uint ny, const double *A, const double *x, uint nx);

/* y = A^T x */
void tensor_mtxv(double *y, uint ny, const double *A, const double *x, uint nx);

/* C (nc x nb) = A (nc x na) * B (na x nb); all column-major */
void tensor_mxm(double *C, uint nc,
                const double *A, uint na, const double *B, uint nb);
#  else
#    define nek_mxm FORTRAN_UNPREFIXED(mxm,MXM)
/* C (na x nc) = A (na x nb) * B (nb x nc); all column-major */
void nek_mxm(const double *A, const uint *na,
             const double *B, const uint *nb,
             double *C, const uint *nc);
/* C (nc x nb) = A (nc x na) * B (na x nb); all column-major */
static void tensor_mxm(double *C, uint nc,
                       const double *A, uint na, const double *B, uint nb)
{ nek_mxm(A,&nc,B,&na,C,&nb); }

/* y = A x */
static void tensor_mxv(double *y, uint ny,
                       const double *A, const double *x, uint nx)
{ uint one=1; nek_mxm(A,&ny,x,&nx,y,&one); }

/* y = A^T x */
static void tensor_mtxv(double *y, uint ny,
                        const double *A, const double *x, uint nx)
{ uint one=1; nek_mxm(x,&one,A,&nx,y,&ny); }

#  endif
#endif

/*--------------------------------------------------------------------------
   1-,2-,3-d Tensor Application of Row Vectors (for Interpolation)
   
   the 3d case:
   v = tensor_i3(Jr,nr, Js,ns, Jt,nt, u, work)
     gives v = [ Jr (x) Js (x) Jt ] u
     where Jr, Js, Jt are row vectors (interpolation weights)
     u is nr x ns x nt in column-major format (inner index is r)
     v is a scalar
  --------------------------------------------------------------------------*/

static double tensor_i1(const double *Jr, uint nr, const double *u)
{
  return tensor_dot(Jr,u,nr);
}

/* work holds ns doubles */
static double tensor_i2(const double *Jr, uint nr,
                        const double *Js, uint ns,
                        const double *u, double *work)
{
  tensor_mtxv(work,ns, u, Jr,nr);
  return tensor_dot(Js,work,ns);
}

/* work holds ns*nt + nt doubles */
static double tensor_i3(const double *Jr, uint nr,
                        const double *Js, uint ns,
                        const double *Jt, uint nt,
                        const double *u, double *work)
{
  double *work2 = work+nt;
  tensor_mtxv(work2,ns*nt,   u,     Jr,nr);
  tensor_mtxv(work ,nt   ,   work2, Js,ns);
  return tensor_dot(Jt,work,nt);
}

/*--------------------------------------------------------------------------
   1-,2-,3-d Tensor Application of Row Vectors
             for simultaneous Interpolation and Gradient computation
   
   the 3d case:
   v = tensor_ig3(g, wtr,nr, wts,ns, wtt,nt, u, work)
     gives v   = [ Jr (x) Js (x) Jt ] u
           g_0 = [ Dr (x) Js (x) Jt ] u
           g_1 = [ Jr (x) Ds (x) Jt ] u
           g_2 = [ Jr (x) Js (x) Dt ] u
     where Jr,Dr,Js,Ds,Jt,Dt are row vectors,
       Jr=wtr, Dr=wtr+nr, etc.
       (interpolation & derivative weights)
     u is nr x ns x nt in column-major format (inner index is r)
     v is a scalar, g is an array of 3 doubles
  --------------------------------------------------------------------------*/

static double tensor_ig1(double g[1],
                         const double *wtr, uint nr,
                         const double *u)
{
  g[0] = tensor_dot(wtr+nr,u,nr);
  return tensor_dot(wtr   ,u,nr);
}

/* work holds 2*nr doubles */
static double tensor_ig2(double g[2],
                         const double *wtr, uint nr,
                         const double *wts, uint ns,
                         const double *u, double *work)
{
  tensor_mxm(work,nr, u,ns, wts,2);
  g[0] = tensor_dot(wtr+nr,work   ,nr);
  g[1] = tensor_dot(wtr   ,work+nr,nr);
  return tensor_dot(wtr   ,work   ,nr);
}

/* work holds 2*nr*ns + 3*nr doubles */
static double tensor_ig3(double g[3],
                         const double *wtr, uint nr,
                         const double *wts, uint ns,
                         const double *wtt, uint nt,
                         const double *u, double *work)
{
  const uint nrs = nr*ns;
  double *a = work, *b = work+2*nrs, *c=b+2*nr;
  tensor_mxm(a,nrs, u,nt, wtt,2);
  tensor_mxm(b,nr,  a,ns, wts,2);
  tensor_mxv(c,nr, a+nrs, wts,ns);
  g[0] = tensor_dot(b   , wtr+nr, nr);
  g[1] = tensor_dot(b+nr, wtr   , nr);
  g[2] = tensor_dot(c   , wtr   , nr);
  return tensor_dot(b   , wtr   , nr);
}

/*
  out - nr x ns
  u   - mr x ms
  Jrt - mr x nr, Jst - ms x ns
  work - nr x ms
*/
static void tensor_2t(double *out,
                      const double *Jrt, uint nr, uint mr,
                      const double *Jst, uint ns, uint ms,
                      const double *u, double *work)
{
  tensor_mtxm(work,nr, Jrt,mr, u,ms);
  tensor_mxm(out,nr, work,ms, Jst,ns);
}

/*
  out - nr x ns x nt
  u   - mr x ms x mt
  Jrt - mr x nr, Jst - ms x ns, Jtt - mt x nt
  work - nr*ms*mt + nr*ns*mt = nr*(ms+ns)*mt
*/
static void tensor_3t(double *out,
                      const double *Jrt, uint nr, uint mr,
                      const double *Jst, uint ns, uint ms,
                      const double *Jtt, uint nt, uint mt,
                      const double *u, double *work)
{
  const uint nrs=nr*ns, mst=ms*mt, nrms=nr*ms;
  uint k;
  double *work2 = work+nr*mst;
  double *p; const double *q;
  tensor_mtxm(work,nr, Jrt,mr, u,mst);
  for(k=0,p=work2,q=work;k<mt;++k,p+=nrs,q+=nrms)
    tensor_mxm(p,nr, q,ms, Jst,ns);
  tensor_mxm(out,nrs, work2,mt, Jtt,nt);
}

#endif
