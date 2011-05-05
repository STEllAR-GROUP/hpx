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
  std::vector<double> refine_level;
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
extern "C" {void FNAME(level_makeflag)(double *flag,double *error,int *level,
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

const int maxlevels = 25;

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
  std::cout << " bound_width   : " <<  bound_width << std::endl;
  for (int i=0;i<allowedl;i++) {
    char tmpname[80];
    std::cout << " refine_level_" << i << " : " << par.refine_level[i] << std::endl;
  }

  par.nx0 = nx0;
  par.ny0 = ny0;
  par.nz0 = nz0;
  par.allowedl = allowedl;
  par.ethreshold = ethreshold;
  par.ghostwidth = ghostwidth;
  par.bound_width = bound_width;
  par.maxx0 = maxx0;
  par.minx0 = minx0;
  par.maxy0 = maxy0;
  par.miny0 = miny0;
  par.maxz0 = maxz0;
  par.minz0 = minz0;
  par.numprocs = numprocs;
  par.shadow = shadow;

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

  std::vector<double> error,flag; 
  error.resize(nx0*ny0*nz0);
  flag.resize(nx0*ny0*nz0);

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

  double scalar = par.refine_level[0];
  int level = 0;
  FNAME(load_scal_mult3d)(&*error.begin(),&*error.begin(),&scalar,&nx0,&ny0,&nz0);
  FNAME(level_makeflag)(&*flag.begin(),&*error.begin(),&level,
                                                 &minx0,&miny0,&minz0,&h,
                                                 &nx0,&ny0,&nz0,&ethreshold);
  {
    int shape[3];
    std::vector<double> coord;
    coord.resize(nx0+ny0+nz0);
    for (int i=0;i<nx0;i++) {
      coord[i] = minx0 + i*h; 
    }
    for (int i=0;i<ny0;i++) {
      coord[i+nx0] = miny0 + i*h; 
    }
    for (int i=0;i<nz0;i++) {
      coord[i+nx0+ny0] = minz0 + i*h; 
    }
    
   shape[0] = nx0;
   shape[1] = ny0;
   shape[2] = nz0;
   gft_out_full("flag",0.0,shape,"x|y|z", 3,&*coord.begin(),&*flag.begin());
  }

  // level_cluster
  int numbox = 1;
  int clusterstyle = 0;
  double minefficiency = .9;
  int mindim = 6;
  int refine_factor = 2;
  double time = 0.0;
  std::vector<double> sigi,sigj,sigk;
  std::vector<double> asigi,asigj,asigk;
  std::vector<double> lapi,lapj,lapk;
  std::vector<double> alapi,alapj,alapk;
  std::vector<int> b_minx,b_maxx,b_miny,b_maxy,b_minz,b_maxz;

   sigi.resize(nx0); sigj.resize(ny0); sigk.resize(nz0);
  asigi.resize(nx0);asigj.resize(ny0);asigk.resize(nz0);
   lapi.resize(nx0); lapj.resize(ny0); lapk.resize(nz0);
  alapi.resize(nx0);alapj.resize(ny0);alapk.resize(nz0);
 
  int maxbboxsize = 1000;
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
                       &minx0,&maxx0,
                       &miny0,&maxy0,
                       &minz0,&maxz0,
                       &numbox,&nx0,&ny0,&nz0,
                       &clusterstyle,&minefficiency,&mindim,
                       &ghostwidth,&refine_factor, 
                       &minx0,&miny0,&minz0, 
                       &maxx0,&maxy0,&maxz0);

  for (int i=0;i<numbox;i++) {
    std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
    std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
    std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;
  }

  // Take the set of subgrids to create and then output the same set but domain decomposed
  std::vector<double> tmp_mini,tmp_maxi,tmp_minj,tmp_maxj,tmp_mink,tmp_maxk;
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
    std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
    std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
    std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;
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

