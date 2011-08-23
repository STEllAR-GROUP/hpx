// 2 May 2011
// Copyright (c) 2011 Matt Anderson
// HAD clustering algorithm in C++ 

#include <iostream>
#include <vector>
#include <math.h>
#include <sdf.h>
#include "parse.h"
using namespace std;

struct Globals
{
  int nx0;
  int ny0;
  int nz0;
  int ghostwidth; 
  int boundwidth; 
  int allowedl;
  int shadow;
  double minx0;
  double maxx0;
  double miny0;
  double maxy0;
  double minz0;
  double maxz0;
  double h;
  int refine_factor;
  std::vector<int> Levelp;
  std::vector<double> gr_minx;
  std::vector<double> gr_miny;
  std::vector<double> gr_minz;
  std::vector<double> gr_maxx;
  std::vector<double> gr_maxy;
  std::vector<double> gr_maxz;
  std::vector<int> gr_sibling;
  std::vector<double> gr_h;
  std::vector<double> gr_t;
  std::vector<int> gr_proc;
  int numprocs;
  std::vector<int> gr_alive;
  std::vector<double> ref_level;
};

int level_return_start(int,struct Globals *);
int level_return_finest(struct Globals *par);
int level_apply(int lev,int action,struct Globals *par);
int level_refine(int level,struct Globals *par);
int level_find_bounds(int level, double &minx, double &maxx,
                                 double &miny, double &maxy,
                                 double &minz, double &maxz, struct Globals *par);
int grid_return_existence(int gridnum,struct Globals *par);
int grid_find_bounds(int gi,double &minx,double &maxx,
                            double &miny,double &maxy,
                            double &minz,double &maxz,struct Globals *par);
double grid_return_h(int gi,struct Globals *par);
double grid_return_time(int gi,struct Globals *par);
int grid_return_sibling(int gi,struct Globals *par);
int level_mkall_dead(int level,struct Globals *par);
int grid_mk_dead(int grid,struct Globals *par);
int floatcmp(double const& x1, double const& x2);

const int maxlev = 25;
const int REFINE = 5; 
const int ALIVE = 1;
const int DEAD = 0;
const int PENDING = -1;

int main(int argc,char* argv[]) {

  Record *list;

  int nx0 = 99;
  int ny0 = 99;
  int nz0 = 99;
  int ghostwidth = 6;
  int boundwidth = 5;
  int allowedl = 0;
  int nlevels;
  int numprocs = 1;
  int shadow = 0;
  double minx0,maxx0,miny0,maxy0,minz0,maxz0;
  minx0 = -4.0;
  maxx0 =  4.0;
  miny0 = -4.0;
  maxy0 =  4.0;
  minz0 = -4.0;
  maxz0 =  4.0;

  if ( argc < 2 ) {
    std::cerr << " Paramter file required " << std::endl;
    exit(0);
  }

  list = Parse(argv[1],'=');
  if ( list == NULL ) {
    std::cerr << " Parameter file open error : " << argv[1] << " not found " << std::endl;
    exit(0);
  }

  if ( GetInt(list, "allowedl", &allowedl) == 0) {
    std::cerr << " Parameter allowedl not found, using default " << std::endl;
  } 
  nlevels = allowedl + 1;

  if ( GetInt(list, "nx0", &nx0) == 0) {
    std::cerr << " Parameter nx0 not found, using default " << std::endl;
  } 
  if ( GetInt(list, "ny0", &ny0) == 0) {
    std::cerr << " Parameter ny0 not found, using default " << std::endl;
  } 
  if ( GetInt(list, "nz0", &nz0) == 0) {
    std::cerr << " Parameter nz0 not found, using default " << std::endl;
  } 

  if ( GetInt(list, "numprocs", &numprocs) == 0) {
    std::cerr << " Parameter numprocs not found, using default " << std::endl;
  } 
  if ( GetInt(list, "ghostwidth", &ghostwidth) == 0) {
    std::cerr << " Parameter ghostwidth not found, using default " << std::endl;
  } 
  if ( GetInt(list, "boundwidth", &boundwidth) == 0) {
    std::cerr << " Parameter boundwidth not found, using default " << std::endl;
  } 
  if ( GetInt(list, "shadow", &shadow) == 0) {
    std::cerr << " Parameter shadow not found, using default " << std::endl;
  } 

  if ( GetDouble(list, "maxx0", &maxx0) == 0) {
    std::cerr << " Parameter maxx0 not found, using default " << std::endl;
  }
  if ( GetDouble(list, "minx0", &minx0) == 0) {
    std::cerr << " Parameter minx0 not found, using default " << std::endl;
  }
  if ( GetDouble(list, "maxy0", &maxy0) == 0) {
    std::cerr << " Parameter maxy0 not found, using default " << std::endl;
  }
  if ( GetDouble(list, "miny0", &miny0) == 0) {
    std::cerr << " Parameter miny0 not found, using default " << std::endl;
  }
  if ( GetDouble(list, "maxz0", &maxz0) == 0) {
    std::cerr << " Parameter maxz0 not found, using default " << std::endl;
  }
  if ( GetDouble(list, "minz0", &minz0) == 0) {
    std::cerr << " Parameter minz0 not found, using default " << std::endl;
  }

  if ( nx0%2 == 0 ) {
    std::cerr << " PROBLEM: hadcc cannot handle even nx0 " << nx0 << std::endl;
    return 0;
  }

  // print out parameters
  std::cout << " allowedl     : " <<  allowedl << std::endl;
  std::cout << " nx0          : " <<  nx0 << std::endl;
  std::cout << " ny0          : " <<  ny0 << std::endl;
  std::cout << " nz0          : " <<  nz0 << std::endl;
  std::cout << " shadow       : " <<  shadow << std::endl;
  std::cout << " numprocs     : " <<  numprocs << std::endl;
  std::cout << " maxx0        : " <<  maxx0 << std::endl;
  std::cout << " minx0        : " <<  minx0 << std::endl;
  std::cout << " maxy0        : " <<  maxy0 << std::endl;
  std::cout << " miny0        : " <<  miny0 << std::endl;
  std::cout << " maxz0        : " <<  maxz0 << std::endl;
  std::cout << " minz0        : " <<  minz0 << std::endl;
  std::cout << " ghostwidth   : " <<  ghostwidth << std::endl;
  std::cout << " boundwidth   : " <<  boundwidth << std::endl;

  struct Globals par;
  par.nx0 = nx0;
  par.allowedl = allowedl;
  par.ghostwidth = ghostwidth;
  par.boundwidth = boundwidth;
  par.maxx0 = maxx0;
  par.minx0 = minx0;
  par.maxy0 = maxy0;
  par.miny0 = miny0;
  par.maxz0 = maxz0;
  par.minz0 = minz0;
  par.refine_factor = 2;
  par.numprocs = numprocs;
  par.shadow = shadow;
  par.ref_level.resize(maxlev);
  for (int i=0;i<maxlev;i++) {
    par.ref_level[i] = 0.0;
  }

  // derived parameters
  double hx = (maxx0 - minx0)/(nx0-1);
  double hy = (maxy0 - miny0)/(ny0-1);
  double hz = (maxz0 - minz0)/(nz0-1);
  double h = hx;

  // verify hx == hy == hz
  if ( fabs(hx - hy) < 1.e-10 && fabs(hy - hz) < 1.e-10 ) {
    h = hx;
  } else {
    std::cerr << " PROBLEM: hadcc cannot handle unequal spacings " << std::endl;
    return 0;
  }
  par.h = h;

  par.Levelp.resize(maxlev+1);
  // initialize
  for (int i=0;i<=maxlev;i++) {
    par.Levelp[i] = 0;
  }

  //-------------------------------------------------------------------------
  // the work starts here

  // Create coarse grid and refine:
  int lev_i = -1;
  level_apply(lev_i,REFINE,&par);

  return 0;
}

int level_apply(int lev,int action,struct Globals *par) {
  int gi = level_return_start(lev,par); 
  int flev = level_return_finest(par);
  //level_refine(lev);
}

int level_refine(int level,struct Globals *par) {
  double minx,miny,minz;
  double maxx,maxy,maxz;
  int gi;
  int rc;
  double time;
  double hl;
  int myproc;
  if ( level == -1 ) {
    minx = par->minx0; 
    maxx = par->maxx0; 
    miny = par->miny0; 
    maxy = par->maxy0; 
    minz = par->minz0; 
    maxz = par->maxz0; 
    gi = -1;
    hl = par->h * par->refine_factor;
    time = 0.0;
    myproc = 0;
  } else {
    rc = level_find_bounds(level,minx,maxx,miny,maxy,minz,maxz,par);

    // Find the beginning of level
    gi = level_return_start(level,par); 

    // Grid spacing for the level:
    hl = grid_return_h(gi,par);
    
    // Time on level (just for tracing/output)
    time = grid_return_time(level_return_start(level,par),par);

    myproc = par->gr_proc[gi];
  }

  // Dimensions of bounding box for entire level:
  int nxl   =  (int) ((maxx-minx)/hl+0.5);
  int nyl   =  (int) ((maxy-miny)/hl+0.5);
  int nzl   =  (int) ((maxz-minz)/hl+0.5);
  nxl++;
  nyl++;
  nzl++;

  rc = level_mkall_dead(level+1,par);

  bool usingfmr;
  // For creating coarse grid or for FMR, can skip much of this stuff:
  if ( level > 0 ) {
    if ( par->ref_level[level] < 0.0 ) {
      usingfmr = true;
    } else {
      usingfmr = false;
    }    
  } else {
    usingfmr = true;
  }

  int totalrefining = true;
  int tmplev = level;
  while ( tmplev > 0 && totalrefining ) {
    if ( totalrefining && floatcmp(par->ref_level[tmplev],-1.0) ) {
      tmplev = tmplev - 1;
    } else {
      totalrefining = false;
    }
  }

  std::vector<int> bbox_minx,bbox_maxx,bbox_miny,bbox_maxy,bbox_minz,bbox_maxz;
  if ( level == -1 || totalrefining || ( level == 0 && par->shadow != 0 && !usingfmr ) ) {
    bbox_minx.push_back(0);
    bbox_maxx.push_back(nxl);
    bbox_miny.push_back(0);
    bbox_maxy.push_back(nyl);
    bbox_minz.push_back(0);
    bbox_maxz.push_back(nzl);
  }

  // Loop over all grids in level
  //   and get its error data:
  // Find the beginning of level
  int gi_tmp = level_return_start(level,par); 


  return 0;
}

int level_mkall_dead(int level,struct Globals *par)
{
  int rc;
  // Find the beginning of level
  int gi = level_return_start(level,par); 

  while ( grid_return_existence(gi,par) ) {
    rc = grid_mk_dead(gi,par);
    gi = grid_return_sibling(gi,par);
  }
    
  return 0;
}

int grid_mk_dead(int grid,struct Globals *par) 
{
  par->gr_alive[grid] = DEAD;
  return 0;
}

int level_find_bounds(int level, double &minx, double &maxx,
                                 double &miny, double &maxy,
                                 double &minz, double &maxz, struct Globals *par) 
{
  int rc;
  minx = 0.0;
  miny = 0.0;
  minz = 0.0;
  maxx = 0.0;
  maxy = 0.0;
  maxz = 0.0;

  double tminx,tmaxx,tminy,tmaxy,tminz,tmaxz;

  // Find the beginning of level
  int gi = level_return_start(level,par); 

  if ( !grid_return_existence(gi,par) ) {
    std::cerr << " level_find_bounds PROBLEM: level doesn't exist " << level << std::endl;
    exit(0);
  }

  rc = grid_find_bounds(gi,minx,maxx,miny,maxy,minz,maxz,par);

  gi = grid_return_sibling(gi,par);
  while ( grid_return_existence(gi,par) ) {
    rc = grid_find_bounds(gi,tminx,tmaxx,tminy,tmaxy,tminz,tmaxz,par);
    if ( tminx < minx ) minx = tminx;
    if ( tminy < miny ) miny = tminy;
    if ( tminz < minz ) minz = tminz;
    if ( tmaxx > maxx ) maxx = tmaxx;
    if ( tmaxy > maxy ) maxy = tmaxy;
    if ( tmaxz > maxz ) maxz = tmaxz;
    gi = grid_return_sibling(gi,par);
  }

  return 0;
}

int grid_return_sibling(int gi,struct Globals *par)
{
  return par->gr_sibling[gi];
}

double grid_return_h(int gi,struct Globals *par)
{
  return par->gr_h[gi];
}

double grid_return_time(int gi,struct Globals *par) {
  return par->gr_t[gi];
}

int grid_find_bounds(int gi,double &minx,double &maxx,
                            double &miny,double &maxy,
                            double &minz,double &maxz,struct Globals *par)
{
  minx = par->gr_minx[gi];
  miny = par->gr_miny[gi];
  minz = par->gr_minz[gi];

  maxx = par->gr_maxx[gi];
  maxy = par->gr_maxy[gi];
  maxz = par->gr_maxz[gi];

  return 0;
}

int grid_return_existence(int gridnum,struct Globals *par) 
{
  if ( (gridnum > 0 ) && (gridnum < par->gr_minx.size()) ) {
    return 1;
  } else {
    return 0;
  }
}

int level_return_start(int level,struct Globals *par) 
{
  if ( level >= maxlev  || level < 0 ) {
    return -1;
  } else {
    return par->Levelp[level];
  }
}

int level_return_finest(struct Globals *par) 
{
  int lrf = 0;  
  while ( par->Levelp[lrf+1] > 0 ) {
    lrf++;  
    if ( lrf == maxlev ) break;
  }
  return lrf;
}

int floatcmp(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}

