//  Copyright (c) 2009-2011 Matthew Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <hpx/hpx.hpp>

const int maxgids = 1000;

#include <examples/adaptive1d/dataflow/dynamic_stencil_value.hpp>
#include <examples/adaptive1d/dataflow/functional_component.hpp>
#include <examples/adaptive1d/dataflow/dataflow_stencil.hpp>
#include <examples/adaptive1d/stencil/stencil.hpp>
#include <examples/adaptive1d/stencil/stencil_data.hpp>
#include <examples/adaptive1d/stencil/stencil_functions.hpp>
#include <examples/adaptive1d/stencil/logging.hpp>

using hpx::components::adaptive1d::parameter;
using hpx::naming::id_type;

int compute_error(std::vector<double> &error,int nx0,
                                double minx0,
                                double maxx0,
                                double h,double t,
                                int gi,
                boost::shared_ptr<std::vector<id_type> > &result_data,
                                parameter &par);
int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,
                  int nxl, int nx);
int grid_return_existence(int gridnum,parameter &par);
int grid_find_bounds(int gi,double &minx,double &maxx,
                            parameter &par);
int level_return_start(int level,parameter &par);
int level_return_start(int level,parameter &par);
int level_find_bounds(int level, double &minx, double &maxx,
                                 parameter &par);
int compute_numrows(parameter &par);
int compute_rowsize(parameter &par);
int increment_gi(int level,int nx,
                 double lminx, double lmaxx,
                 double hl,int refine_factor,parameter &par);
bool intersection(double xmin,double xmax,double xmin2,double xmax2);
bool floatcmp_le(double const& x1, double const& x2);
int floatcmp(double const& x1, double const& x2);

// level_refine {{{
int level_refine(int level,parameter &par,boost::shared_ptr<std::vector<id_type> > &result_data, double time)
{
  int rc;
  int ghostwidth = par->ghostwidth;
  int bound_width = par->num_neighbors;
  double ethreshold = par->ethreshold;
  double minx0 = par->minx0;
  double maxx0 = par->maxx0;
  double h = par->h;
  int clusterstyle = 0;
  double minefficiency = 0.9;
  int mindim = 6;
  int refine_factor = 2;

  // local vars
  std::vector<int> b_minx,b_maxx,b_miny,b_maxy,b_minz,b_maxz;
  std::vector<double> tmp_mini,tmp_maxi,tmp_minj,tmp_maxj,tmp_mink,tmp_maxk;
  int numbox;

  // hard coded parameters -- eventually to be made into full parameters
  int maxbboxsize = 1000000;

  int maxlevel = par->allowedl;
  if ( level == maxlevel ) {
    return 0;
  }
 
  double minx,maxx;
  double hl;
  int gi;
  if ( level == -1 ) {
    // we're creating the coarse grid
    minx = minx0;
    maxx = maxx0;
    gi = -1;
    hl = h * refine_factor;
  } else {
    // find the bounds of the level
    rc = level_find_bounds(level,minx,maxx,par);

    // find the grid index of the beginning of the level
    gi = level_return_start(level,par);

    // grid spacing for the level
    hl = par->gr_h[gi];
  }

  int nxl   =  (int) ((maxx-minx)/hl+0.5);
  nxl++;

  std::vector<double> error,localerror,flag;
  error.resize(nxl);
  flag.resize(nxl);

  if ( level == -1 ) {
    b_minx.resize(maxbboxsize);
    b_maxx.resize(maxbboxsize);

    tmp_mini.resize(maxbboxsize);
    tmp_maxi.resize(maxbboxsize);

    int numbox = 1;
    b_minx[0] = 1;
    b_maxx[0] = nxl;
    //FNAME(level_clusterdd)(&*tmp_mini.begin(),&*tmp_maxi.begin(),
    //                       &*b_minx.begin(),&*b_maxx.begin(),
    //                       &numbox,&numprocs,&maxbboxsize,
    //                       &ghostwidth,&refine_factor,&mindim,
    //                       &bound_width);

    //std::cout << " numbox post DD " << numbox << std::endl;
    for (int i=0;i<numbox;i++) {
      if (i == numbox-1 ) {
        par->gr_sibling.push_back(-1);
      } else {
        par->gr_sibling.push_back(i+1);
      }

      int nx = (b_maxx[i] - b_minx[i])*refine_factor+1;

      double lminx = minx + (b_minx[i]-1)*hl;
      double lmaxx = minx + (b_maxx[i]-1)*hl;

      par->gr_minx.push_back(lminx);
      par->gr_maxx.push_back(lmaxx);
      par->gr_h.push_back(hl/refine_factor);
      par->gr_nx.push_back(nx);
    }

    return 0;
  } else {
    // loop over all grids in level and get its error data
    gi = level_return_start(level,par);

    while ( grid_return_existence(gi,par) ) {
      int nx = par->gr_nx[gi];

      localerror.resize(nx);

      double lminx = par->gr_minx[gi];
      double lmaxx = par->gr_maxx[gi];

      rc = compute_error(localerror,nx,
                         lminx, lmaxx,
                         par->gr_h[gi],time,gi,result_data,par);

      int mini = (int) ((lminx - minx)/hl+0.5);

      // sanity check
      if ( floatcmp(lminx-(minx + mini*hl),0.0) == 0 ||
           floatcmp(lmaxx-(minx + (mini+nx-1)*hl),0.0) == 0 ) {
        std::cerr << " Index problem " << std::endl;
        std::cerr << " lminx " << lminx << " " << minx + mini*hl << std::endl;
        std::cerr << " lmaxx " << lminx << " " << minx + (mini+nx-1)*hl << std::endl;
        exit(0);
      }

      rc = level_combine(error,localerror,
                         mini,nxl,nx);

      gi = par->gr_sibling[gi];
    }
  }

  double scalar = 1.0;
  //FNAME(load_scal_mult)(&*error.begin(),&*error.begin(),&scalar,&nxl);
  int gw = par->ghostwidth;
  //FNAME(level_makeflag_simple)(&*flag.begin(),&*error.begin(),&level,
  //                                               &minx,&h,&nxl,&ethreshold,&gw);

  // level_cluster
  std::vector<double> sigi;
  std::vector<double> asigi;
  std::vector<double> lapi;
  std::vector<double> alapi;

   sigi.resize(nxl);
  asigi.resize(nxl);
   lapi.resize(nxl);
  alapi.resize(nxl);

  b_minx.resize(maxbboxsize);
  b_maxx.resize(maxbboxsize);
#if 0
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
#endif
  // TEST
  numbox = 0;
  //std::cout << " pre DD numbox TEST " << numbox << std::endl;

   // Figure out the domain decomposition you want here

  //for (int i=0;i<numbox;i++) {
  //  std::cout << " bbox : " << b_minx[i] << " " << b_maxx[i] << std::endl; 
  //  std::cout << "      : " << b_miny[i] << " " << b_maxy[i] << std::endl; 
  //  std::cout << "      : " << b_minz[i] << " " << b_maxz[i] << std::endl; 
  //}

  //std::cout << " numbox post DD " << numbox << std::endl;
  int prev_tgi = 0;
  int tgi = 0;
  for (int i=0;i<numbox;i++) {
    //std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
    //std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
    //std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;
    int nx = (b_maxx[i] - b_minx[i])*refine_factor+1;

    double lminx = minx + (b_minx[i]-1)*hl;
    double lmaxx = minx + (b_maxx[i]-1)*hl;

    if ( i != 0 ) {
      prev_tgi = tgi;
    }

    // look for matching grid on level
    tgi = increment_gi(level,nx,lminx,lmaxx,hl,refine_factor,par);

    if ( i == 0 ) {
      par->levelp[level+1] = tgi;
    }

    if (i == numbox-1) {
      par->gr_sibling[prev_tgi] = tgi;
      par->gr_sibling[tgi] = -1;
    } else if (i != 0 ) {
      par->gr_sibling[prev_tgi] = tgi;
    }

  }

  return 0;
}
// }}}

// compute_error {{{
int compute_error(std::vector<double> &error,int nx0,
                                double minx0,
                                double maxx0,
                                double h,double t,
                                int gi,
                boost::shared_ptr<std::vector<id_type> > &result_data,
                                parameter &par)
{
    // initialize some positive error
    if ( t < 1.e-8 ) {
      for (int i=0;i<nx0;i++) {
        double x = minx0 + i*h;

        // Provide initial error
        double Phi = exp(-x*x);
        error[i] = Phi;
      }
    } else {
      // This is the old mesh structure; we need the error for the new mesh structure
      // Some re-assembly is necessary
      // go through all of the old mesh gi's; see if they overlap this new mesh
      for (int step=0;step<par->prev_gi.size();step++) {
        // see if the new gi is the same as the old
        int gi = par->prev_gi[step];
        if ( gi != -1 ) {
          if ( floatcmp(minx0,par->gr_minx[gi]) && 
               floatcmp(h,par->gr_h[gi]) && 
               nx0 == par->gr_nx[gi] 
             ) {
            hpx::components::access_memory_block<hpx::components::adaptive1d::stencil_data>
                result( hpx::components::stubs::memory_block::get((*result_data)[par->gi2item[gi]]) );
            for (int i=0;i<nx0;i++) {
              error[i] = result->value_[i].error;
            }
          } else if ( intersection(minx0,maxx0,
                                   par->gr_minx[gi],par->gr_maxx[gi]) &&
                      floatcmp(h,par->gr_h[gi])
                    ) { 
            hpx::components::access_memory_block<hpx::components::adaptive1d::stencil_data>
                result( hpx::components::stubs::memory_block::get((*result_data)[par->gi2item[gi]]) );

            // find the intersection index
            double x1 = (std::max)(minx0,par->gr_minx[gi]); 
            double x2 = (std::min)(maxx0,par->gr_maxx[gi]); 
  
            int isize = (int) ( (x2-x1)/h );
  
            int istart_dst = (int) ( (x1 - minx0)/h );
  
            int istart_src = (int) ( (x1 - par->gr_minx[gi])/h );
  
            for (int ii=0;ii<=isize;ii++) {
              int i = ii + istart_dst;
  
              int si = ii + istart_src;
              BOOST_ASSERT(i < error.size());
              BOOST_ASSERT(si < result->value_.size());
              error[i] = result->value_[si].error;
            }

          }
        }
      }
    }

    return 0;
}
// }}}

// level_combine {{{
int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,
                  int nxl, int nx)
{
  int il;
  // combine local grid error into global grid error
  for (int i=0;i<nx;i++) {
    il = i + mini;
    if ( error[il] < localerror[i] ) {
      error[il] = localerror[i];
    }
  }

  return 0;
}
// }}}

// grid_return_existence {{{
int grid_return_existence(int gridnum,parameter &par)
{
  int maxsize = par->gr_minx.size();
  if ( (gridnum >= 0 ) && (gridnum < maxsize) ) {
    return 1;
  } else {
    return 0;
  }
}
// }}}

// grid_find_bounds {{{
int grid_find_bounds(int gi,double &minx,double &maxx,
                            parameter &par)
{
  minx = par->gr_minx[gi];
  maxx = par->gr_maxx[gi];

  return 0;
}
// }}}

// level_return_start {{{
int level_return_start(int level,parameter &par)
{
  int maxsize = par->levelp.size();
  if ( level >= maxsize || level < 0 ) {
    return -1;
  } else {
    return par->levelp[level];
  }
}
// }}}

// level_find_bounds {{{
int level_find_bounds(int level, double &minx, double &maxx,
                                 parameter &par)
{
  int rc;
  minx = 0.0;
  maxx = 0.0;

  double tminx,tmaxx;

  int gi = level_return_start(level,par);

  if ( !grid_return_existence(gi,par) ) {
    std::cerr << " level_find_bounds PROBLEM: level doesn't exist " << level << std::endl;
    exit(0);
  }

  rc = grid_find_bounds(gi,minx,maxx,par);

  gi = par->gr_sibling[gi];
  while ( grid_return_existence(gi,par) ) {
    rc = grid_find_bounds(gi,tminx,tmaxx,par);
    if ( tminx < minx ) minx = tminx;
    if ( tmaxx > maxx ) maxx = tmaxx;
    gi = par->gr_sibling[gi];
  }

  return 0;
}
// }}}

// compute_numrows {{{
int compute_numrows(parameter &par)
{
    // for each row, record what the lowest level on the row is
    std::size_t num_rows = 1 << par->allowedl;

    // account for prolongation and restriction (which is done every other step)
    if (par->allowedl > 0)
        num_rows += (1 << par->allowedl) / 2;

    num_rows *= 2; // we take two timesteps in the mesh
    par->num_rows = num_rows;

    int ii = -1; 
    for (std::size_t i = 0; i < num_rows; ++i)
    {
        if (((i + 5) % 3) != 0 || (par->allowedl == 0))
            ii++;

        std::size_t level = 0;
        for (std::size_t j = par->allowedl; j>=0; --j)
        {
            if ((ii % (1 << j)) == 0)
            {
                level = par->allowedl - j;
                par->level_row.push_back(level);
                break;
            }
        }
    }
    return 0;
}
// }}}

// compute_rowsize {{{
int compute_rowsize(parameter &par)
{
    // discover each rowsize
    std::size_t count;
    par->rowsize.resize(par->allowedl+1);
    for (std::size_t i=0;i<=par->allowedl;i++) {
      count = 0; 
      int gi = level_return_start(i,par);
      count++;
      gi = par->gr_sibling[gi];
      while ( grid_return_existence(gi,par) ) {
        count++;
        gi = par->gr_sibling[gi];
      }
      par->rowsize[i] = count;
    } 
    for (std::size_t i=0;i<=par->allowedl;i++) {
      for (std::size_t j=i+1;j<=par->allowedl;j++) {
        par->rowsize[i] += par->rowsize[j];
      }
    }

    // here we create a correspondence getween the gi number used in 'had' 
    // and the index number used in hpx
    par->item2gi.resize(par->rowsize[0]);
    par->gi2item.resize(maxgids);
    count = 0;
    for (int i=par->allowedl;i>=0;i--) {
      int gi = level_return_start(i,par);
      par->item2gi[count] = gi; 
      par->gi2item[gi] = count; 
      count++;
      gi = par->gr_sibling[gi];
      while ( grid_return_existence(gi,par) ) {
        par->item2gi[count] = gi; 
        par->gi2item[gi] = count; 
        count++;
        gi = par->gr_sibling[gi];
      }
    } 

    return 0;
}
// }}}

// increment_gi {{{
int increment_gi(int level,int nx,
                 double lminx, double lmaxx,
                 double hl,int refine_factor,parameter &par)
{
    int gi_tmp = level_return_start(level+1,par);
    bool found = false;
    while ( grid_return_existence(gi_tmp,par) ) {
      int lnx = par->gr_nx[gi_tmp];
      if ( floatcmp(lminx,par->gr_minx[gi_tmp]) &&
           nx == lnx ) {
        found = true;
        break;
      }
      gi_tmp = par->gr_sibling[gi_tmp];
    }
    if ( found ) return gi_tmp;

    // none found; this is a new gi
    int gi = par->gr_h.size();
    par->gr_minx.resize(gi+1);
    par->gr_maxx.resize(gi+1);
    par->gr_h.resize(gi+1);
    par->gr_nx.resize(gi+1);
    par->gr_sibling.resize(gi+1);

    par->gr_minx[gi]  = lminx;
    par->gr_maxx[gi]  = lmaxx;
    par->gr_h[gi]     = hl/refine_factor;
    par->gr_nx[gi]    = nx;
    
    return gi;

};
// }}}

// intersection {{{
bool intersection(double xmin,double xmax,double xmin2,double xmax2) 
{
  double pa[1],ea[1];
  static double const half = 0.5;
  pa[0] = half*(xmax + xmin);
  ea[0] = xmax - pa[0];

  double pb[1],eb[1];
  pb[0] = half*(xmax2 + xmin2);
  eb[0] = xmax2 - pb[0];

  double T[1];
  T[0] = pb[0] - pa[0];

  if ( floatcmp_le(fabs(T[0]),ea[0] + eb[0]) ) {
    return true;
  } else {
    return false;
  }
}
// }}}

// floatcmp_le {{{
bool floatcmp_le(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;

  if ( x1 < x2 ) return true;

  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}

// floatcmp {{{
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
// }}}
