/*  Copyright (c) 2009 Maciej Brodowicz
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying 
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include "mpi.h"

#define DEFXSZ 1024
#define DEFYSZ 1024
#define DEFGH 1
#define DEFITER 100
#define RNDMAPSZ 10000


MPI_Datatype xghosttype, yghosttype;

struct coords
{ /* global stuff */
  int x, y;     /* size of mesh */
  int px, py;   /* processor grid */
  int ngh;      /* number of ghost cells */
  int stcnt;    /* number of points in stencil */
  int dx, dy;   /* stencil range */
  int world;    /* world size */
  /* local */
  int rank;         /* process rank */
  int lx, ly;       /* local section of mesh */
  int lgx, lgy;     /* size of local mesh including ghosts */
  int xoff, yoff;   /* offset of local mesh */
  int cartx, carty; /* local Cartesian coords */
};

char rdist;        /* random distribution type */
long *randmap;     /* computed map function for required distribution */
double mean, var;  /* random distribution parameters */


double gf(double *buf, struct coords *co, int i, int j)
{
  double sum = 0.0;
  int x, y;
  long t, n = randmap[(long)(drand48()*RNDMAPSZ)];

  /* i and j are actual buffer indices */
  for (t = 0; t < n; t++)
    for (y = j-co->dy; y <= j+co->dy; y++)
      for (x = i-co->dx; x <= i+co->dx; x++)
	sum += buf[y*co->lgx+x];
  return sum/(co->stcnt*t);
}

void eval(double *buf, struct coords *co, double *tmp)
{
  int i, j, bwr, twr = 0, trr = 0;

  for (j = 0; j < co->ly; j++)
  {
    double *res = tmp+twr*co->lx;
    for (i = 0; i < co->lx; i++)
    {
      *res++ = gf(buf, co, i+co->ngh, j+co->ngh);
    }
    if (++twr > co->ngh) twr = 0;
    if (j >= co->ngh)
    {
      memcpy(buf+j*co->lgx+co->ngh, tmp+trr*co->lx, co->lx*sizeof(double));
      if (++trr > co->ngh) trr = 0;
    }
  }
  for (; j < co->ly+co->ngh; j++)
  { /* copy the remainder of temporary buffer */
    memcpy(buf+j*co->lgx+co->ngh, tmp+trr*co->lx, co->lx*sizeof(double));
    if (++trr > co->ngh) trr = 0;
  }
}

void comm(double *buf, struct coords *co)
{
  MPI_Request req[4];
  MPI_Status stat[4];
  int nreq = 0;

  /* initialize all receives */
  if (co->px > 1)
  {
    if (co->cartx > 0)
      MPI_Irecv(buf+co->ngh*co->lgx, 1, yghosttype, co->rank-1, 0,
		MPI_COMM_WORLD, &req[nreq++]);
    if (co->cartx < co->px-1)
      MPI_Irecv(buf+(co->ngh+1)*co->lgx-co->ngh, 1, yghosttype, co->rank+1, 0,
		MPI_COMM_WORLD, &req[nreq++]);
  }
  if (co->py > 1)
  {
    if (co->carty > 0)
      MPI_Irecv(buf+co->ngh, 1, xghosttype, co->rank-co->px, 0,
		MPI_COMM_WORLD, &req[nreq++]);
    if (co->carty < co->py-1)
      MPI_Irecv(buf+(co->ngh+co->ly)*co->lgx+co->ngh, 1, xghosttype,
		co->rank+co->px, 0, MPI_COMM_WORLD, &req[nreq++]);
  }
  /* perform sends */
  if (co->px > 1)
  {
    if (co->cartx < co->px-1)
      MPI_Send(buf+co->ngh*co->lgx+co->lx, 1, yghosttype, co->rank+1, 0,
		MPI_COMM_WORLD);
    if (co->cartx > 0)
      MPI_Send(buf+co->ngh*co->lgx+co->ngh, 1, yghosttype, co->rank-1, 0,
		MPI_COMM_WORLD);
  }
  if (co->py > 1)
  {
    if (co->carty < co->py-1)
      MPI_Send(buf+co->ly*co->lgx+co->ngh, 1, xghosttype, co->rank+co->px, 0,
		MPI_COMM_WORLD);
    if (co->carty > 0)
      MPI_Send(buf+co->ngh*co->lgx+co->ngh, 1, xghosttype, co->rank-co->px, 0,
		MPI_COMM_WORLD);
  }
  if (MPI_Waitall(nreq, req, stat) != MPI_SUCCESS)
  {
    printf("Communication failed on rank %d\n", co->rank);
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }
}

/* from: http://home.online.no/~pjacklam/notes/invnorm/impl/natarajan/normsinv.h */
double normicdf(double p)
{
#define  A1  (-3.969683028665376e+01)
#define  A2   2.209460984245205e+02
#define  A3  (-2.759285104469687e+02)
#define  A4   1.383577518672690e+02
#define  A5  (-3.066479806614716e+01)
#define  A6   2.506628277459239e+00

#define  B1  (-5.447609879822406e+01)
#define  B2   1.615858368580409e+02
#define  B3  (-1.556989798598866e+02)
#define  B4   6.680131188771972e+01
#define  B5  (-1.328068155288572e+01)

#define  C1  (-7.784894002430293e-03)
#define  C2  (-3.223964580411365e-01)
#define  C3  (-2.400758277161838e+00)
#define  C4  (-2.549732539343734e+00)
#define  C5   4.374664141464968e+00
#define  C6   2.938163982698783e+00

#define  D1   7.784695709041462e-03
#define  D2   3.224671290700398e-01
#define  D3   2.445134137142996e+00
#define  D4   3.754408661907416e+00

#define P_LOW   0.02425
/* P_HIGH = 1 - P_LOW */
#define P_HIGH  0.97575

  double x, q, r, u, e;

  if ((0 < p )  && (p < P_LOW))
  {
    q = sqrt(-2*log(p));
    x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6)/((((D1*q+D2)*q+D3)*q+D4)*q+1);
  }
  else
  {
    if ((P_LOW <= p) && (p <= P_HIGH))
    {
      q = p - 0.5;
      r = q*q;
      x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q/(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1);
    }
    else
    {
      if ((P_HIGH < p) && (p < 1))
      {
	q = sqrt(-2*log(1-p));
	x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6)/((((D1*q+D2)*q+D3)*q+D4)*q+1);
      }
    }
  }
  return x;
}

double *init(struct coords *co)
{
  int i, j, npos = 0;
  double *buf;

  co->lx = (co->x == 1? 1: co->x/co->px);
  co->ly = (co->y == 1? 1: co->y/co->py);
  co->lgx = co->lx+2*co->ngh;
  co->lgy = co->ly+2*co->ngh;

  buf = calloc(co->lgx*co->lgy, sizeof(double));
  if (!buf)
  {
    printf("Malloc failed for %d double elements\n", co->lgx*co->lgy);
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  co->cartx = co->rank%co->px;
  co->carty = co->rank/co->px;
  co->xoff = co->lx*co->cartx;
  co->yoff = co->ly*co->carty;
  co->dx = co->ngh*(co->x > 1);
  co->dy = co->ngh*(co->y > 1);
  co->stcnt = (2*co->dx+1)*(2*co->dy+1);

  /* datatype for ghost cells along x direction (single zone) */
  MPI_Type_vector(co->ngh, co->lx, co->lgx, MPI_DOUBLE, &xghosttype);
  MPI_Type_commit(&xghosttype);
  /* datatype for ghost cells along y direction (single zone) */
  MPI_Type_vector(co->ly, co->ngh, co->lgx, MPI_DOUBLE, &yghosttype);
  MPI_Type_commit(&yghosttype);

  /* RNG initialization, distribution table setup */
  srand48(((co->yoff+j)<<16) | (co->xoff+i));
  randmap = malloc(RNDMAPSZ*sizeof(long));
  if (rdist == 'u')
  { /* uniform distribution */
    double h = 0.5*sqrt(12*var);
    
    for (i = 0; i < RNDMAPSZ; i++)
    {
      randmap[i] = mean-h+((double)i+0.5)/RNDMAPSZ*(2*h);
      if (randmap[i] <= 0) ++npos;
    }
  }
  else if (rdist == 'n')
  { /* normal distribution */
    for (i = 0; i < RNDMAPSZ; i++)
    {
      randmap[i] = mean+sqrt(var)*normicdf(((double)i+0.5)/RNDMAPSZ);
      if (randmap[i] <= 0) ++npos;
    }
  }

  if (npos)
    printf("Warning (node %d): %d non-positive entries in distribution table!\n", co->rank, npos);

  /* data initialization */
  for (j = co->ngh; j < co->ly+co->ngh; j++)
    for (i = co->ngh; i < co->lx+co->ngh; i++)
      buf[j*co->lgx+i] = drand48();

  return buf;
}

void show(int iter, char *msg, double *buf, struct coords *co)
{
  static double *tmp = 0;

  if (!co->rank)
  {
    MPI_Status st;
    double *t;
    int i, j, n;

    if (!tmp) tmp = malloc(co->lgx*co->lgy*sizeof(double));

    for (n = 0; n < co->world; n++)
    {
      if (n)
      {
	MPI_Recv(tmp, co->lgx*co->lgy, MPI_DOUBLE, n, 0, MPI_COMM_WORLD, &st);
	t = tmp;
      }
      else t = buf;
      printf("\nNode %d, iteration %d, %s:\n", n, iter, msg);
      for (j = 0; j < co->lgy; j++)
      {
	for (i = 0; i < co->lgx; i++) printf("%10f ", t[j*co->lgx+i]);
	printf("\n");
      }
    }
    fflush(stdout);
  }
  else
  {
    MPI_Send(buf, co->lgx*co->lgy, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

int main(int argc, char **argv)
{
  struct timespec tstart, tend;
  struct coords loc;
  int iters = DEFITER;
  double *data, *temp, rtime;

  MPI_Init(&argc, &argv);
  loc.ngh = DEFGH;
  loc.x = DEFXSZ; loc.y = DEFYSZ;
  loc.px = loc.py = 1;
  rdist = 'u'; mean = 1, var = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &loc.rank);
  {
    char **ap;
    for (ap = argv+1; ap < argv+argc; ap++)
      if ((*ap)[0] == '-' && (*ap)[1] && !(*ap)[2])
      {
	if ((*ap)[1] == 'x') loc.x = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'y') loc.y = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'p') loc.px = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'q') loc.py = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'n') iters = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'g') loc.ngh = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'm') mean = strtod(*++ap, 0);
	else if ((*ap)[1] == 'v') var = strtod(*++ap, 0);
	else if ((*ap)[1] == 'r')
	{
	  rdist = tolower((*++ap)[0]);
	  if (!strchr("nu", rdist))
	  {
	    if (!loc.rank) printf("Invalid distribution in \"-r\" option: %c\n", rdist);
	    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	  }
	}
	else
	{
	  if (!loc.rank) printf("Unknown option: %s\n", *ap);
	  MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
      }
      else
      {
	if (!loc.rank) printf("Unexpected command line argument: \"%s\"\n", *ap);
	MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
      }
  }
  
  MPI_Comm_size(MPI_COMM_WORLD, &loc.world);
  if (loc.px*loc.py != loc.world ||
      (loc.x == 1 && loc.px != 1) || (loc.x != 1 && loc.x%loc.px) ||
      (loc.y == 1 && loc.py != 1) || (loc.y != 1 && loc.y%loc.py))
  {
    printf("Invalid processor grid definition\n");
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  data = init(&loc);
  temp = malloc((loc.ngh+1)*loc.lx*sizeof(double));

  /* main computations */
  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_REALTIME, &tstart);
  {
    int i;
    for (i = 0; i < iters; i++)
    {
      /*show(i, "start", data, &loc);*/
      comm(data, &loc);
      /*show(i, "after update", data, &loc);*/
      eval(data, &loc, temp);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_REALTIME, &tend);

  if (!loc.rank)
  {
    printf("Mesh size: %dx%d, processor grid: %dx%d, ghost cells: %d\n",
	   loc.x, loc.y, loc.px, loc.py, loc.ngh);
    rtime = tend.tv_sec-tstart.tv_sec+1e-9*(tend.tv_nsec-tstart.tv_nsec);
    printf("Time for %d iterations: %f s (%f s per iteration)\n",
	   iters, rtime, rtime/iters);
  }

  MPI_Finalize();
}
