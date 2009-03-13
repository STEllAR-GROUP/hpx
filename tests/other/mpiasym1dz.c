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
#define DEFGH 1
#define DEFITER 100


MPI_Datatype yghosttype;

struct coords
{ /* global stuff */
  int x;        /* size of mesh */
  int world;    /* world size */
  int ngh;      /* number of ghost cells */
  int zone;     /* number of points in "uniform workload" zone */
  int nzones;   /* number of zones per grid */
  /* local */
  int rank;     /* process rank */
  int lx;       /* local section of mesh */
  int lgx;      /* size of local mesh including ghosts */
  int xoff;     /* offset of local mesh */
};

char rdist;          /* random distribution type */
long *work;          /* precomputed workload table for required distribution */
double mean, stddev; /* random distribution parameters */


/* perform workload on a single mesh point; parameters:
 *   pointer to buffer,
 *   pointer to problem geometry description,
 *   current iteration number,
 *   index into local data buffer,
 *   global point coordinate
 */
double gf(double *buf, struct coords *co, int iter, int i, int gx)
{
  double sum = 0.0;
  long t, n = work[iter*co->nzones+gx/co->zone];

  /*printf("[%d] %ld\n", co->rank, n);*/
  for (t = 0; t < n; t++)
    sum += buf[i-1]+buf[i]+buf[i+1];
  return sum/(3.0*t);
}

/* single time step evaluation */
void eval(double *buf, struct coords *co, double *tmp, int iter)
{
  int i;
  double *res = tmp;

  for (i = 0; i < co->lx; i++)
  {
    *res++ = gf(buf, co, iter, i+co->ngh, co->xoff+i);
  }
  memcpy(buf+co->ngh, res, co->lx*sizeof(double));
}

/* communicate in a ring */
void comm(double *buf, struct coords *co)
{
  MPI_Request req[2];
  MPI_Status stat[2];
  int nreq = 0;

  /*MPI_Barrier(MPI_COMM_WORLD);*/
  /* initialize all receives */
  if (co->world > 1)
  {
    /* recv from left, deposit in the ghost zone */
    MPI_Irecv(buf, 1, yghosttype,
	      (co->rank? co->rank-1: co->world-1), 0,
	      MPI_COMM_WORLD, &req[nreq++]);
    /* recv from right, deposit in the ghost zone */
    MPI_Irecv(buf+co->lx+co->ngh, 1, yghosttype,
	      (co->rank == co->world-1? 0: co->rank+1), 0,
	      MPI_COMM_WORLD, &req[nreq++]);
  }
  /* perform sends */
  if (co->world > 1)
  {
    /* send to the right */
    MPI_Send(buf+co->ngh+co->lx, 1, yghosttype,
	     (co->rank == co->world-1? 0: co->rank+1), 0, MPI_COMM_WORLD);
    /* send to the left */
    MPI_Send(buf+co->ngh, 1, yghosttype,
	     (co->rank > 0? co->rank-1: co->world-1), 0, MPI_COMM_WORLD);
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

/* initialize required data structures */
double *init(struct coords *co, int n)
{
  int i, npos = 0, nz = 0;
  double *buf;

  co->lx = co->x/co->world;
  co->lgx = co->lx+2*co->ngh;

  buf = calloc(co->lgx, sizeof(double));
  if (!buf)
  {
    printf("Error: malloc failed for %d double elements\n", co->lgx);
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  co->xoff = co->lx*co->rank;
  if (co->zone <= 0) co->zone = co->lx;
  if (co->x%co->zone)
  {
    if (!co->rank)
      printf("Error: Non-integral number of zones per domain\n");
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }
  /* all random numbers are pre-generated before computations start to avoid
   * additional and possibly inconsistent overheads during runtime
   */
  co->nzones = co->x/co->zone;
  nz = co->nzones*co->world*n;

  /* datatype for ghost cells along y direction (single zone) */
  MPI_Type_vector(1, co->ngh, co->lgx, MPI_DOUBLE, &yghosttype);
  MPI_Type_commit(&yghosttype);

  /* RNG initialization, distribution table setup */
  srand48(42);
  work = malloc(nz*sizeof(long));

  if (rdist == 'u')
  { /* uniform distribution */
    double h = sqrt(3.0)*stddev;
    
    for (i = 0; i < nz; i++)
    {
      work[i] = mean-h+drand48()*(2*h);
      if (work[i] <= 0) ++npos;
    }
  }
  else if (rdist == 'n')
  { /* normal distribution */
    for (i = 0; i < nz; i++)
    {
      work[i] = mean+stddev*normicdf(drand48());
      if (work[i] <= 0) ++npos;
    }
  }

  if (npos)
    printf("Warning (node %d): %d non-positive entries in distribution table!\n", co->rank, npos);

  /* make sure there are differences between nodes */
  /*srand48(42*co->rank+1);*/
  /* mesh data initialization */
  for (i = co->ngh; i < co->lx+co->ngh; i++)
    buf[i] = drand48();

  return buf;
}

void show(int iter, char *msg, double *buf, struct coords *co)
{
  static double *tmp = 0;

  if (!co->rank)
  {
    MPI_Status st;
    double *t;
    int i, n;

    if (!tmp) tmp = malloc(co->lgx*sizeof(double));

    for (n = 0; n < co->world; n++)
    {
      if (n)
      {
	MPI_Recv(tmp, co->lgx, MPI_DOUBLE, n, 0, MPI_COMM_WORLD, &st);
	t = tmp;
      }
      else t = buf;
      printf("\nNode %d, iteration %d, %s:\n", n, iter, msg);
      for (i = 0; i < co->lgx; i++) printf("%10f ", t[i]);
      printf("\n");
    }
    fflush(stdout);
  }
  else
  {
    MPI_Send(buf, co->lgx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
  loc.x = DEFXSZ;
  loc.zone = 0;
  rdist = 'u'; mean = 1; stddev = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &loc.rank);
  /* process options */
  {
    char **ap;
    for (ap = argv+1; ap < argv+argc; ap++)
      if ((*ap)[0] == '-' && (*ap)[1] && !(*ap)[2])
      {
	if ((*ap)[1] == 'x') loc.x = strtol(*++ap, 0, 0);
	else if ((*ap)[1] == 'n') iters = strtol(*++ap, 0, 0);
	/*else if ((*ap)[1] == 'g') loc.ngh = strtol(*++ap, 0, 0);*/
	else if ((*ap)[1] == 'm') mean = strtod(*++ap, 0);
	else if ((*ap)[1] == 's') stddev = strtod(*++ap, 0);
	else if ((*ap)[1] == 'z') loc.zone = strtol(*++ap, 0, 0);
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
  if (loc.x%loc.world != 0)
  {
    if (!loc.rank) printf("Invalid processor grid definition\n");
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  data = init(&loc, iters);
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
      eval(data, &loc, temp, i);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_REALTIME, &tend);

  if (!loc.rank)
  {
    printf("Mesh size: %d, processors: %d, ghost cells: %d\n",
	   loc.x, loc.world, loc.ngh);
    rtime = tend.tv_sec-tstart.tv_sec+1e-9*(tend.tv_nsec-tstart.tv_nsec);
    printf("Time for %d iterations: %f s (%f s per iteration)\n",
	   iters, rtime, rtime/iters);
  }

  MPI_Finalize();
}
