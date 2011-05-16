// 2 May 2011
// Matt Anderson
// HAD clustering algorithm in C++ 

#include <iostream>
#include <vector>
#include <math.h>
#include <sdf.h>
#include "parse.h"
#include "fname.h"

using namespace std;

struct Globals
{
  int nx0;
  int ny0;
  int nz0;
  int ghostwidth; 
  int bound_width; 
  int allowedl;
  int shadow;
  double ethreshold;
  double minx0;
  double maxx0;
  double miny0;
  double maxy0;
  double minz0;
  double maxz0;
  double h;
  int numprocs;
  int clusterstyle;
  double minefficiency;
  int mindim;
  int refine_factor;
  std::vector<double> refine_level;
  std::vector<int> gr_sibling;
  std::vector<double> gr_t;
  std::vector<double> gr_minx;
  std::vector<double> gr_miny;
  std::vector<double> gr_minz;
  std::vector<double> gr_maxx;
  std::vector<double> gr_maxy;
  std::vector<double> gr_maxz;
  std::vector<int> gr_nx;
  std::vector<int> gr_ny;
  std::vector<int> gr_nz;
  std::vector<int> gr_proc;
  std::vector<double> gr_h;
  std::vector<int> gr_alive;
  std::vector<int> levelp;

};

extern "C" {void FNAME(level_cluster)(double *flag,
                          double *sigi,double *sigj,double *sigk,
                          double *lapi, double *lapj, double *lapk,
                          double *asigi, double *asigj, double *asigk,
                          double *alapi, double *alapj, double *alapk,
                          double *time, 
                          int *b_minx, int *b_maxx,int *b_miny,int *b_maxy,
                          int *b_minz, int *b_maxz,
                          double *minx, double *maxx,
                          double *miny, double *maxy,
                          double *minz, double *maxz,
                          int *numbox, int *nx, int *ny, int *nz,
                          int *clusterstyle, double *minefficiency, int *mindim,
                          int *ghostwidth, int *refine_factor,
                          double *minx0,double *miny0, double *minz0,
                          double *maxx0,double *maxy0, double *maxz0); }

extern "C" {void FNAME(load_scal_mult3d)(double *,double *,
                                         double *,int *,int *,int *); }
extern "C" {void FNAME(level_makeflag_simple)(double *flag,double *error,int *level,
                                       double *minx,double *miny,double *minz,
                                       double *h,int *nx,int *ny,int *nz,
                                       double *ethreshold); }

extern "C" {void FNAME(level_clusterdd)(double *tmp_mini,double *tmp_maxi,
                         double *tmp_minj,double *tmp_maxj,
                         double *tmp_mink,double *tmp_maxk,
                         int *b_mini,int *b_maxi,
                         int *b_minj,int *b_maxj,
                         int *b_mink,int *b_maxk,
                         int *numbox,int *numprocs,int *maxbboxsize,
                         int *ghostwidth,int *refine_factor,int *mindim,
                         int *bound_width); }

int floatcmp(double const& x1, double const& x2);
int level_refine(int level,Globals & par);
int level_find_bounds(int level, double &minx, double &maxx,
                                 double &miny, double &maxy,
                                 double &minz, double &maxz, Globals &par);
int grid_return_existence(int gridnum,struct Globals &par);
int level_return_start(int level,Globals &par);
int grid_find_bounds(int gi,double &minx,double &maxx,
                            double &miny,double &maxy,
                            double &minz,double &maxz,Globals &par);
int compute_error(std::vector<double> &error,int nx0, int ny0, int nz0,
                                double minx0,double miny0,double minz0,double h);
int level_mkall_dead(int level,Globals &par);
int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,int minj,int mink,
                  int nxl,int nyl,int nzl,
                  int nx,int ny,int nz);

const int maxlevels = 25;
const int ALIVE = 1;
const int DEAD = 0;
const int PENDING = -1;

int main(int argc,char* argv[]) {

  Record *list;

  struct Globals par;
  int nx0 = 99;
  int ny0 = 99;
  int nz0 = 99;
  int ghostwidth = 9;
  int bound_width = 5;
  int allowedl = 0;
  int nlevels;
  int numprocs = 1;
  int shadow = 0;
  double minx0,maxx0,miny0,maxy0,minz0,maxz0;
  double ethreshold = 1.e-4;
  int clusterstyle = 0;
  double minefficiency = 0.9;
  int mindim = 6;
  int refine_factor = 2;
  minx0 = -4.0;
  maxx0 =  4.0;
  miny0 = -4.0;
  maxy0 =  4.0;
  minz0 = -4.0;
  maxz0 =  4.0;
  par.refine_level.resize(maxlevels);
  for (int i=0;i<maxlevels;i++) {
    // default
    par.refine_level[i] = 1.0;
  } 

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
  if ( GetInt(list, "bound_width", &bound_width) == 0) {
    std::cerr << " Parameter bound_width not found, using default " << std::endl;
  } 
  if ( GetInt(list, "shadow", &shadow) == 0) {
    std::cerr << " Parameter shadow not found, using default " << std::endl;
  } 
  if ( GetInt(list, "clusterstyle", &clusterstyle) == 0) {
    std::cerr << " Parameter clusterstyle not found, using default " << std::endl;
  } 
  if ( GetInt(list, "mindim", &mindim) == 0) {
    std::cerr << " Parameter mindim not found, using default " << std::endl;
  } 
  if ( GetInt(list, "refine_factor", &refine_factor) == 0) {
    std::cerr << " Parameter refine_factor not found, using default " << std::endl;
  } 
  if ( GetDouble(list, "minefficiency", &minefficiency) == 0) {
    std::cerr << " Parameter minefficiency not found, using default " << std::endl;
  }
  if ( GetDouble(list, "maxx0", &maxx0) == 0) {
    std::cerr << " Parameter maxx0 not found, using default " << std::endl;
  }
  if ( GetDouble(list, "ethreshold", &ethreshold) == 0) {
    std::cerr << " Parameter ethreshold not found, using default " << std::endl;
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
  for (int i=0;i<allowedl;i++) {
    char tmpname[80];
    double tmp;
    sprintf(tmpname,"refine_level_%d",i);
    if ( GetDouble(list, tmpname, &tmp) == 0) {
      par.refine_level[i] = tmp;
    }
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
  std::cout << " ethreshold   : " <<  ethreshold << std::endl;
  std::cout << " shadow       : " <<  shadow << std::endl;
  std::cout << " numprocs     : " <<  numprocs << std::endl;
  std::cout << " maxx0        : " <<  maxx0 << std::endl;
  std::cout << " minx0        : " <<  minx0 << std::endl;
  std::cout << " maxy0        : " <<  maxy0 << std::endl;
  std::cout << " miny0        : " <<  miny0 << std::endl;
  std::cout << " maxz0        : " <<  maxz0 << std::endl;
  std::cout << " minz0        : " <<  minz0 << std::endl;
  std::cout << " ghostwidth   : " <<  ghostwidth << std::endl;
  std::cout << " bound_width  : " <<  bound_width << std::endl;
  std::cout << " clusterstyle : " <<  clusterstyle << std::endl;
  std::cout << " minefficiency: " <<  minefficiency<< std::endl;
  std::cout << " mindim       : " <<  mindim << std::endl;
  std::cout << " refine_factor: " <<  refine_factor << std::endl;
  for (int i=0;i<allowedl;i++) {
    char tmpname[80];
    std::cout << " refine_level_" << i << " : " << par.refine_level[i] << std::endl;
  }

  par.nx0 = nx0;
  par.ny0 = ny0;
  par.nz0 = nz0;
  par.allowedl = allowedl;
  par.ethreshold = ethreshold;
  par.ghostwidth = 2*ghostwidth;
  par.bound_width = bound_width;
  par.maxx0 = maxx0;
  par.minx0 = minx0;
  par.maxy0 = maxy0;
  par.miny0 = miny0;
  par.maxz0 = maxz0;
  par.minz0 = minz0;
  par.numprocs = numprocs;
  par.shadow = shadow;
  par.clusterstyle = clusterstyle;
  par.mindim = mindim;
  par.minefficiency = minefficiency;
  par.refine_factor = refine_factor;

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

  // memory allocation
  par.levelp.resize(par.allowedl);
  par.levelp[0] = 0; 

  int rc = level_refine(-1,par);

  for (int i=0;i<par.allowedl;i++) {
    rc = level_refine(i,par);
  }

  return 0;
}

int level_refine(int level,Globals & par) 
{
  int rc;
  int numprocs = par.numprocs;
  int ghostwidth = par.ghostwidth;
  int bound_width = par.bound_width;
  int nx0 = par.nx0;
  int ny0 = par.ny0;
  int nz0 = par.nz0;
  double ethreshold = par.ethreshold;
  double minx0 = par.minx0;
  double miny0 = par.miny0;
  double minz0 = par.minz0;
  double maxx0 = par.maxx0;
  double maxy0 = par.maxy0;
  double maxz0 = par.maxz0;
  double h = par.h;
  int clusterstyle = par.clusterstyle;
  double minefficiency = par.minefficiency;
  int mindim = par.mindim;
  int refine_factor = par.refine_factor;

  // local vars
  std::vector<int> b_minx,b_maxx,b_miny,b_maxy,b_minz,b_maxz;
  std::vector<double> tmp_mini,tmp_maxi,tmp_minj,tmp_maxj,tmp_mink,tmp_maxk;
  int numbox;

  // hard coded parameters -- eventually to be made into full parameters
  int maxbboxsize = 1000000;

  if ( level == par.allowedl ) { 
    return 0;
  }

  double minx,miny,minz,maxx,maxy,maxz;
  double hl,time;
  int gi;
  if ( level == -1 ) {
    // we're creating the coarse grid
    minx = minx0; 
    miny = miny0; 
    minz = minz0; 
    maxx = maxx0; 
    maxy = maxy0; 
    maxz = maxz0; 
    gi = -1;
    hl = h * refine_factor;
    time = 0.0;
  } else {
    // find the bounds of the level
    rc = level_find_bounds(level,minx,maxx,miny,maxy,minz,maxz,par);
        
    // find the grid index of the beginning of the level
    gi = level_return_start(level,par);

    // grid spacing for the level
    hl = par.gr_h[gi];

    // find the time on the level
    time = par.gr_t[gi];
  }

  int nxl   =  (int) ((maxx-minx)/hl+0.5);
  int nyl   =  (int) ((maxy-miny)/hl+0.5);
  int nzl   =  (int) ((maxz-minz)/hl+0.5);
  nxl++;
  nyl++;
  nzl++;

  //std::cout << " TEST nls: " << nxl << " " << nyl << " " << nzl << std::endl;
  //std::cout << " TEST min/max: " << minx << " " << maxx << std::endl;
  //std::cout << "               " << miny << " " << maxy << std::endl;
  //std::cout << "               " << minz << " " << maxz << std::endl;

  std::vector<double> error,localerror,flag; 
  error.resize(nxl*nyl*nzl);
  flag.resize(nxl*nyl*nzl);

  if ( level == -1 ) {
    b_minx.resize(maxbboxsize);
    b_maxx.resize(maxbboxsize);
    b_miny.resize(maxbboxsize);
    b_maxy.resize(maxbboxsize);
    b_minz.resize(maxbboxsize);
    b_maxz.resize(maxbboxsize);

    tmp_mini.resize(maxbboxsize);
    tmp_maxi.resize(maxbboxsize);
    tmp_minj.resize(maxbboxsize);
    tmp_maxj.resize(maxbboxsize);
    tmp_mink.resize(maxbboxsize);
    tmp_maxk.resize(maxbboxsize);

    int numbox = 1;
    b_minx[0] = 1;
    b_maxx[0] = nxl;
    b_miny[0] = 1;
    b_maxy[0] = nyl;
    b_minz[0] = 1;
    b_maxz[0] = nzl;
    FNAME(level_clusterdd)(&*tmp_mini.begin(),&*tmp_maxi.begin(),
                           &*tmp_minj.begin(),&*tmp_maxj.begin(),
                           &*tmp_mink.begin(),&*tmp_maxk.begin(),
                           &*b_minx.begin(),&*b_maxx.begin(),
                           &*b_miny.begin(),&*b_maxy.begin(),
                           &*b_minz.begin(),&*b_maxz.begin(),
                           &numbox,&numprocs,&maxbboxsize,
                           &ghostwidth,&refine_factor,&mindim,
                           &bound_width);

    std::cout << " numbox post DD " << numbox << std::endl;
    for (int i=0;i<numbox;i++) {
      if (i == numbox-1 ) {
        par.gr_sibling.push_back(-1);
      } else {
        par.gr_sibling.push_back(i+1);
      }

      int nx = (b_maxx[i] - b_minx[i])*refine_factor+1;
      int ny = (b_maxy[i] - b_miny[i])*refine_factor+1;
      int nz = (b_maxz[i] - b_minz[i])*refine_factor+1;

      double lminx = minx + (b_minx[i]-1)*hl;
      double lminy = miny + (b_miny[i]-1)*hl;
      double lminz = minz + (b_minz[i]-1)*hl;
      double lmaxx = minx + (b_maxx[i]-1)*hl;
      double lmaxy = miny + (b_maxy[i]-1)*hl;
      double lmaxz = minz + (b_maxz[i]-1)*hl;

      par.gr_t.push_back(time);
      par.gr_minx.push_back(lminx);
      par.gr_miny.push_back(lminy);
      par.gr_minz.push_back(lminz);
      par.gr_maxx.push_back(lmaxx);
      par.gr_maxy.push_back(lmaxy);
      par.gr_maxz.push_back(lmaxz);
      par.gr_proc.push_back(i);
      par.gr_h.push_back(hl/refine_factor);
      par.gr_alive.push_back(0);
      par.gr_nx.push_back(nx);
      par.gr_ny.push_back(ny);
      par.gr_nz.push_back(nz);
      //std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
      //std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
      //std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;

      //std::cout << " minx " << lminx << " maxx " << lmaxx << std::endl;
      //std::cout << " miny " << lminy << " maxy " << lmaxy << std::endl;
      //std::cout << " minz " << lminz << " maxz " << lmaxz << std::endl;
      //std::cout << " nx " << nx << " test " << lminx + (nx-1)*hl/refine_factor << std::endl;
      //std::cout << " ny " << nx << "      " << lminy + (ny-1)*hl/refine_factor << std::endl;
      //std::cout << " nz " << nx << "      " << lminz + (nz-1)*hl/refine_factor << std::endl;

      // output
      {
        std::vector<double> localerror;
        localerror.resize(nx*ny*nz);
        double h = hl/refine_factor;
        int rc = compute_error(localerror,nx,ny,nz,
                         lminx,lminy,lminz,h);
        int shape[3];
        std::vector<double> coord;
        coord.resize(nx+ny+nz);
        double hh = hl/refine_factor;
        for (int i=0;i<nx;i++) {
          coord[i] = lminx + i*hh; 
        }
        for (int i=0;i<ny;i++) {
          coord[i+nx] = lminy + i*hh; 
        }
        for (int i=0;i<nz;i++) {
          coord[i+nx+ny] = lminz + i*hh; 
        }
    
        shape[0] = nx;
        shape[1] = ny;
        shape[2] = nz;
        gft_out_full("error",0.0,shape,"x|y|z", 3,&*coord.begin(),&*localerror.begin());
      }
    }
   
    return 0;
  } else {
    // loop over all grids in level and get its error data
    gi = level_return_start(level,par);

    while ( grid_return_existence(gi,par) ) {
      int nx = par.gr_nx[gi];
      int ny = par.gr_ny[gi];
      int nz = par.gr_nz[gi];

      localerror.resize(nx*ny*nz);

      double lminx = par.gr_minx[gi];
      double lminy = par.gr_miny[gi];
      double lminz = par.gr_minz[gi];
      double lmaxx = par.gr_maxx[gi];
      double lmaxy = par.gr_maxy[gi];
      double lmaxz = par.gr_maxz[gi];

      rc = compute_error(localerror,nx,ny,nz,
                         lminx,lminy,lminz,par.gr_h[gi]);

      int mini = (int) ((lminx - minx)/hl+0.5);
      int minj = (int) ((lminy - miny)/hl+0.5);
      int mink = (int) ((lminz - minz)/hl+0.5);

      // sanity check
      if ( floatcmp(lminx-(minx + mini*hl),0.0) == 0 ||
           floatcmp(lminy-(miny + minj*hl),0.0) == 0 ||
           floatcmp(lminz-(minz + mink*hl),0.0) == 0 ||
           floatcmp(lmaxx-(minx + (mini+nx-1)*hl),0.0) == 0 ||
           floatcmp(lmaxy-(miny + (minj+ny-1)*hl),0.0) == 0 ||
           floatcmp(lmaxz-(minz + (mink+nz-1)*hl),0.0) == 0 ) {
        std::cerr << " Index problem " << std::endl;
        std::cerr << " lminx " << lminx << " " << minx + mini*hl << std::endl;
        std::cerr << " lminy " << lminy << " " << miny + minj*hl << std::endl;
        std::cerr << " lminz " << lminz << " " << minz + mink*hl << std::endl;
        std::cerr << " lmaxx " << lminx << " " << minx + (mini+nx-1)*hl << std::endl;
        std::cerr << " lmaxy " << lminy << " " << miny + (minj+ny-1)*hl << std::endl;
        std::cerr << " lmaxz " << lminz << " " << minz + (mink+nz-1)*hl << std::endl;
        exit(0);
      }

      rc = level_combine(error,localerror,
                         mini,minj,mink,
                         nxl,nyl,nzl,
                         nx,ny,nz);

      gi = par.gr_sibling[gi];
    }
  }

  rc = level_mkall_dead(level+1,par);

  double scalar = par.refine_level[level];
  FNAME(load_scal_mult3d)(&*error.begin(),&*error.begin(),&scalar,&nxl,&nyl,&nzl);
  FNAME(level_makeflag_simple)(&*flag.begin(),&*error.begin(),&level,
                                                 &minx,&miny,&minz,&h,
                                                 &nxl,&nyl,&nzl,&ethreshold);
#if 0
  {
    int shape[3];
    std::vector<double> coord;
    coord.resize(nxl+nyl+nzl);
    for (int i=0;i<nxl;i++) {
      coord[i] = minx + i*h; 
    }
    for (int i=0;i<nyl;i++) {
      coord[i+nxl] = miny + i*h; 
    }
    for (int i=0;i<nzl;i++) {
      coord[i+nxl+nyl] = minz + i*h; 
    }
    
    shape[0] = nxl;
    shape[1] = nyl;
    shape[2] = nzl;
    gft_out_full("flag",0.0,shape,"x|y|z", 3,&*coord.begin(),&*flag.begin());
  }
#endif

  // level_cluster
  std::vector<double> sigi,sigj,sigk;
  std::vector<double> asigi,asigj,asigk;
  std::vector<double> lapi,lapj,lapk;
  std::vector<double> alapi,alapj,alapk;

   sigi.resize(nxl); sigj.resize(nyl); sigk.resize(nzl);
  asigi.resize(nxl);asigj.resize(nyl);asigk.resize(nzl);
   lapi.resize(nxl); lapj.resize(nyl); lapk.resize(nzl);
  alapi.resize(nxl);alapj.resize(nyl);alapk.resize(nzl);
 
  b_minx.resize(maxbboxsize);
  b_maxx.resize(maxbboxsize);
  b_miny.resize(maxbboxsize);
  b_maxy.resize(maxbboxsize);
  b_minz.resize(maxbboxsize);
  b_maxz.resize(maxbboxsize);

  FNAME(level_cluster)(&*flag.begin(),&*sigi.begin(),&*sigj.begin(),&*sigk.begin(),
                       &*lapi.begin(),&*lapj.begin(),&*lapk.begin(),
                       &*asigi.begin(),&*asigj.begin(),&*asigk.begin(),
                       &*alapi.begin(),&*alapj.begin(),&*alapk.begin(),
                       &time,
                       &*b_minx.begin(),&*b_maxx.begin(),
                       &*b_miny.begin(),&*b_maxy.begin(),
                       &*b_minz.begin(),&*b_maxz.begin(),
                       &minx,&maxx,
                       &miny,&maxy,
                       &minz,&maxz,
                       &numbox,&nxl,&nyl,&nzl,
                       &clusterstyle,&minefficiency,&mindim,
                       &ghostwidth,&refine_factor, 
                       &minx0,&miny0,&minz0, 
                       &maxx0,&maxy0,&maxz0);

  //for (int i=0;i<numbox;i++) {
  //  std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
  //  std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
  //  std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;
  //}

  // Take the set of subgrids to create and then output the same set but domain decomposed
  tmp_mini.resize(maxbboxsize);
  tmp_maxi.resize(maxbboxsize);
  tmp_minj.resize(maxbboxsize);
  tmp_maxj.resize(maxbboxsize);
  tmp_mink.resize(maxbboxsize);
  tmp_maxk.resize(maxbboxsize);
  FNAME(level_clusterdd)(&*tmp_mini.begin(),&*tmp_maxi.begin(),
                         &*tmp_minj.begin(),&*tmp_maxj.begin(),
                         &*tmp_mink.begin(),&*tmp_maxk.begin(),
                         &*b_minx.begin(),&*b_maxx.begin(),
                         &*b_miny.begin(),&*b_maxy.begin(),
                         &*b_minz.begin(),&*b_maxz.begin(),
                         &numbox,&numprocs,&maxbboxsize,
                         &ghostwidth,&refine_factor,&mindim,
                         &bound_width);

  std::cout << " numbox post DD " << numbox << std::endl;
  for (int i=0;i<numbox;i++) {
    //std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
    //std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
    //std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;

    int start;
    if ( i == 0 ) {
      start = par.gr_sibling.size();
      par.levelp[level+1] = start;
    }

    if (i == numbox-1 ) {
      par.gr_sibling.push_back(-1);
    } else {
      par.gr_sibling.push_back(start+1+i);
    }

    int nx = (b_maxx[i] - b_minx[i])*refine_factor+1;
    int ny = (b_maxy[i] - b_miny[i])*refine_factor+1;
    int nz = (b_maxz[i] - b_minz[i])*refine_factor+1;

    double lminx = minx + (b_minx[i]-1)*hl;
    double lminy = miny + (b_miny[i]-1)*hl;
    double lminz = minz + (b_minz[i]-1)*hl;
    double lmaxx = minx + (b_maxx[i]-1)*hl;
    double lmaxy = miny + (b_maxy[i]-1)*hl;
    double lmaxz = minz + (b_maxz[i]-1)*hl;
    par.gr_t.push_back(time);
    par.gr_minx.push_back(lminx);
    par.gr_miny.push_back(lminy);
    par.gr_minz.push_back(lminz);
    par.gr_maxx.push_back(lmaxx);
    par.gr_maxy.push_back(lmaxy);
    par.gr_maxz.push_back(lmaxz);
    par.gr_proc.push_back(i);
    par.gr_h.push_back(hl/refine_factor);
    par.gr_alive.push_back(0);
    par.gr_nx.push_back(nx);
    par.gr_ny.push_back(ny);
    par.gr_nz.push_back(nz);

    // output
    {
      std::vector<double> localerror;
      localerror.resize(nx*ny*nz);
      double h = hl/refine_factor;
      int rc = compute_error(localerror,nx,ny,nz,
                       lminx,lminy,lminz,h);
      int shape[3];
      std::vector<double> coord;
      coord.resize(nx+ny+nz);
      double hh = hl/refine_factor;
      for (int i=0;i<nx;i++) {
        coord[i] = lminx + i*hh; 
      }
      for (int i=0;i<ny;i++) {
        coord[i+nx] = lminy + i*hh; 
      }
      for (int i=0;i<nz;i++) {
        coord[i+nx+ny] = lminz + i*hh; 
      }
 
      shape[0] = nx;
      shape[1] = ny;
      shape[2] = nz;
      gft_out_full("error",0.0,shape,"x|y|z", 3,&*coord.begin(),&*localerror.begin());
    }
  }

  return 0;
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

int level_find_bounds(int level, double &minx, double &maxx,
                                 double &miny, double &maxy,
                                 double &minz, double &maxz, Globals &par)
{
  int rc;
  minx = 0.0;
  miny = 0.0;
  minz = 0.0;
  maxx = 0.0;
  maxy = 0.0;
  maxz = 0.0;

  double tminx,tmaxx,tminy,tmaxy,tminz,tmaxz;

  int gi = level_return_start(level,par);

  if ( !grid_return_existence(gi,par) ) {
    std::cerr << " level_find_bounds PROBLEM: level doesn't exist " << level << std::endl;
    exit(0);
  }

  rc = grid_find_bounds(gi,minx,maxx,miny,maxy,minz,maxz,par);

  gi = par.gr_sibling[gi];
  while ( grid_return_existence(gi,par) ) {
    rc = grid_find_bounds(gi,tminx,tmaxx,tminy,tmaxy,tminz,tmaxz,par);
    if ( tminx < minx ) minx = tminx;
    if ( tminy < miny ) miny = tminy;
    if ( tminz < minz ) minz = tminz;
    if ( tmaxx > maxx ) maxx = tmaxx;
    if ( tmaxy > maxy ) maxy = tmaxy;
    if ( tmaxz > maxz ) maxz = tmaxz;
    gi = par.gr_sibling[gi];
  }

  return 0;
}

int grid_return_existence(int gridnum,Globals &par)
{
  if ( (gridnum >= 0 ) && (gridnum < par.gr_minx.size()) ) {
    return 1;
  } else {
    return 0;
  }
}

int grid_find_bounds(int gi,double &minx,double &maxx,
                            double &miny,double &maxy,
                            double &minz,double &maxz,Globals &par)
{
  minx = par.gr_minx[gi];
  miny = par.gr_miny[gi];
  minz = par.gr_minz[gi];

  maxx = par.gr_maxx[gi];
  maxy = par.gr_maxy[gi];
  maxz = par.gr_maxz[gi];

  return 0;
}

int level_return_start(int level,Globals &par)
{
  if ( level >= maxlevels  || level < 0 ) {
    return -1;
  } else {
    return par.levelp[level];
  }
}

int level_mkall_dead(int level,Globals &par)
{
  int rc;
  // Find the beginning of level
  int gi = level_return_start(level,par);

  while ( grid_return_existence(gi,par) ) {
    par.gr_alive[gi] = DEAD;
    gi = par.gr_sibling[gi];
  }

  return 0;
}

int compute_error(std::vector<double> &error,int nx0, int ny0, int nz0,
                                double minx0,double miny0,double minz0,double h)
{
    // initialize some positive error
    for (int k=0;k<nz0;k++) {
      double z = minz0 + k*h;
    for (int j=0;j<ny0;j++) {
      double y = miny0 + j*h;
    for (int i=0;i<nx0;i++) {
      error[i+nx0*(j+ny0*k)] = 0.0;
      double x = minx0 + i*h + 2.25;
      double r = sqrt(x*x+y*y+z*z);
      if ( pow(r-1.0,2) <= 0.5*0.5 && r > 0 ) {
        error[i+nx0*(j+ny0*k)] = pow((r-1.0)*(r-1.0)-0.5*0.5,3)*8.0*(1.0-r)/pow(0.5,8)/r;
        if ( error[i+nx0*(j+ny0*k)] < 0.0 ) error[i+nx0*(j+ny0*k)] = 0.0;
      } 

      x = minx0 + i*h - 2.25;
      r = sqrt(x*x+y*y+z*z);
      if ( pow(r-1.0,2) <= 0.5*0.5 && r > 0 ) {
        error[i+nx0*(j+ny0*k)] = pow((r-1.0)*(r-1.0)-0.5*0.5,3)*8.0*(1.0-r)/pow(0.5,8)/r;
        if ( error[i+nx0*(j+ny0*k)] < 0.0 ) error[i+nx0*(j+ny0*k)] = 0.0;
      }

    } } }
}

int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,int minj,int mink,
                  int nxl,int nyl,int nzl,
                  int nx,int ny,int nz)
{
  int il,jl,kl;
  // combine local grid error into global grid error
  for (int k=0;k<nx;k++) {
    kl = k + mink;    
  for (int j=0;j<ny;j++) {
    jl = j + minj;    
  for (int i=0;i<nx;i++) {
    il = i + mini;    
    if ( error[il + nxl*(jl + nyl*kl)] < localerror[i+nx*(j+ny*k)] ) {
      error[il + nxl*(jl + nyl*kl)] = localerror[i+nx*(j+ny*k)];
    }
  } } }
  
  return 0;
}

