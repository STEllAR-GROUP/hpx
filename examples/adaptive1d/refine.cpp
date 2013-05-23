//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <hpx/hpx.hpp>

#include <examples/adaptive1d/dataflow/dynamic_stencil_value.hpp>
#include <examples/adaptive1d/dataflow/functional_component.hpp>
#include <examples/adaptive1d/dataflow/dataflow_stencil.hpp>
#include <examples/adaptive1d/stencil/stencil.hpp>
#include <examples/adaptive1d/stencil/stencil_data.hpp>
#include <examples/adaptive1d/stencil/stencil_functions.hpp>
#include <examples/adaptive1d/stencil/logging.hpp>
#include <examples/adaptive1d/refine.hpp>

// level_bbox {{{
// the purpose of this routine is to find the boundaries of contiguous refinement regions
// this is used in the prolongation routines
int level_bbox(int level,parameter &par)
{
  int rc;
  int ghostwidth = par->ghostwidth;
//  double ethreshold = par->ethreshold;

  int gi;
  double minx,maxx;

  // find the bounds of the level
  rc = level_find_bounds(level,minx,maxx,par);

  // find the grid index of the beginning of the level
  gi = level_return_start(level,par);

  double h = par->gr_h[gi];

  std::vector<int> level_gi;

  while ( grid_return_existence(gi,par) ) {
    level_gi.push_back(gi);
    gi = par->gr_sibling[gi];
  }

  // Check each element of level_gi to see if they touch
  for (std::size_t i=0;i<level_gi.size();i++) {
    double lminx = par->gr_minx[ level_gi[i] ];
    double lmaxx = par->gr_maxx[ level_gi[i] ];
    gi = level_return_start(level,par);
    bool left_boundary = true;
    bool right_boundary = true;
    while ( grid_return_existence(gi,par) ) {
      if ( gi != level_gi[i] ) {
        double lminx2 = par->gr_minx[ gi ];
        double lmaxx2 = par->gr_maxx[ gi ];
        if ( ballpark(lminx,lmaxx2,2*h) ) {
          left_boundary = false;
          par->gr_left_neighbor[ level_gi[i] ] = gi;
        }
        if ( ballpark(lminx2,lmaxx,2*h) ) {
          right_boundary = false;
          par->gr_right_neighbor[ level_gi[i] ] = gi;
        }
      }
      if ( !right_boundary && !left_boundary ) break;
      gi = par->gr_sibling[gi];
    }
    if ( left_boundary ) par->gr_lbox[ level_gi[i] ] = true;
    if ( right_boundary ) par->gr_rbox[ level_gi[i] ] = true;

    // get prolongation neighbors
    if ( left_boundary || right_boundary ) {
      gi = level_return_start(level-1,par);
      while ( grid_return_existence(gi,par) ) {
        // look for intersection with ghostwidth region
        double lminx2 = par->gr_minx[ gi ];
        double lmaxx2 = par->gr_maxx[ gi ];
        if ( intersection(lminx,lminx+ghostwidth*h,
                         lminx2,lmaxx2) ) {
          par->gr_left_neighbor[ level_gi[i] ] = gi;
        }
        if ( intersection(lmaxx-ghostwidth*h,lmaxx,
                         lminx2,lmaxx2) ) {
          par->gr_right_neighbor[ level_gi[i] ] = gi;
        }
        gi = par->gr_sibling[gi];
      }
    }

  }


  return 0;
}

// }}}

// level_refine {{{
int level_refine(int level,parameter &par,boost::shared_ptr<std::vector<id_type> > &result_data, double time)
{
  int rc;
  int ghostwidth = par->ghostwidth;
  double ethreshold = par->ethreshold;
  double minx0 = par->minx0;
  double maxx0 = par->maxx0;
  double h = par->h;
  int refine_factor = 2;

  // local vars
  std::vector<int> b_minx,b_maxx;
  std::vector<int> ddminx,ddmaxx;

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
    hl = h;
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

  std::vector<double> error,localerror;
  std::vector<int> flag;
  error.resize(nxl);
  flag.resize(nxl);

  if ( level == -1 ) {
    int numbox = nxl/par->grain_size;
    b_minx.resize(numbox);
    b_maxx.resize(numbox);

    std::size_t grain_size;
    grain_size = par->grain_size;
    for (int i=0;i<numbox;i++) {
      b_minx[i] = i*grain_size;
      if ( i == numbox-1 ) {
        grain_size = nxl - (numbox-1)*par->grain_size;
      }
      b_maxx[i] = b_minx[i] + grain_size;

      if (i == numbox-1 ) {
        // indicate the last of the level -- no more siblings
        par->gr_sibling.push_back(-1);
      } else {
        par->gr_sibling.push_back(i+1);
      }
      int nx = (b_maxx[i] - b_minx[i]);

      double lminx = minx + b_minx[i]*hl;
      double lmaxx = lminx + hl*(grain_size-1);

      par->gr_minx.push_back(lminx);
      par->gr_maxx.push_back(lmaxx);
      par->gr_h.push_back(hl);
      par->gr_lbox.push_back(false);
      par->gr_rbox.push_back(false);
      par->gr_left_neighbor.push_back(-1);
      par->gr_right_neighbor.push_back(-1);
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

  level_makeflag_simple(flag,error,nxl,ethreshold);
  // level_cluster
  int hgw = (ghostwidth+1)/2;
  int maxfind = 0;
  for (std::size_t i=0;i<flag.size();i++) {
    if ( flag[i] == 1 && maxfind == 0 ) {
      b_minx.push_back(i);
      maxfind = 1;
    }
    if ( flag[i] == 0 && maxfind == 1 ) {
      b_maxx.push_back(i-1);
      maxfind = 0;
    }
  }

  // add in ghostzones
  for (std::size_t i=0;i<b_maxx.size();i++) {
    b_minx[i] -= hgw;
    b_maxx[i] += hgw;
  }

  // Determine the domain decomposition
  for (std::size_t i=0;i<b_maxx.size();i++) {
    std::size_t grain_size;
    grain_size = par->grain_size;

    BOOST_ASSERT(b_maxx[i] > b_minx[i]);
    std::size_t nx = (b_maxx[i] - b_minx[i])*refine_factor+1;
    if ( par->grain_size > nx ) {
      ddminx.push_back(b_minx[i]);
      ddmaxx.push_back(b_maxx[i]);
    } else {
      int numddbox = nx/par->grain_size;
      // break up b_minx/b_maxx in numddbox portions
      int tnx = b_maxx[i] - b_minx[i];
      int grain_size = tnx/numddbox;
      for (int j=0;j<numddbox;j++) {
        ddminx.push_back(j*grain_size + b_minx[i]);
        if ( j == numddbox-1 ) {
          grain_size = tnx - (numddbox-1)*grain_size;
        }
        ddmaxx.push_back(ddminx[ddminx.size()-1] + grain_size-1);
      }
    }
  }
  BOOST_ASSERT(ddmaxx.size() == ddminx.size());

  int prev_tgi = 0;
  int tgi = 0;
  for (std::size_t i=0;i<ddmaxx.size();i++) {
    int nx = (ddmaxx[i] - ddminx[i])*refine_factor+1;

    double lminx = minx + ddminx[i]*hl;
    double lmaxx = minx + ddmaxx[i]*hl;

    if ( i != 0 ) {
      prev_tgi = tgi;
    }

    // look for matching grid on level
    tgi = increment_gi(level,nx,lminx,lmaxx,hl,refine_factor,par);

    if ( i == 0 ) {
      par->levelp[level+1] = tgi;
    }

    if (i == ddmaxx.size()-1) {
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

      double x1 = 0.5*par->x0;
      double dx_u1 = 9999.;
      for (int i=0;i<nx0;i++) {
        double x = minx0 + i*h;
        if ( -x1 <= x && x <= x1 ) {
          dx_u1 = par->amp*(1.0-pow(tanh(x/(par->id_sigma*par->id_sigma)),2))/
                                  (par->id_sigma*par->id_sigma);
        } else if ( x >= x1 && x <= par->x0 + x1 ) {
          dx_u1 = -par->amp*(1.0-pow(tanh( (x-par->x0)/(par->id_sigma*par->id_sigma)),2))/
                                  (par->id_sigma*par->id_sigma);
        } else if ( x <= -x1 ) {
          dx_u1 = -par->amp*(1.0-pow(tanh( (x-2*par->x0)/(par->id_sigma*par->id_sigma)),2))/
                                  (par->id_sigma*par->id_sigma);
        } else if ( x >= par->x0 + x1 ) {
          dx_u1 = par->amp*(1.0-pow(tanh( (x-2*par->x0)/(par->id_sigma*par->id_sigma)),2))/
                                  (par->id_sigma*par->id_sigma);
        } else {
          BOOST_ASSERT(false);
        }

        // Provide initial error
        error[i] = fabs(dx_u1);
      }
    } else {
      // This is the old mesh structure; we need the error for the new mesh structure
      // Some re-assembly is necessary
      // go through all of the old mesh gi's; see if they overlap this new mesh
      int prev_size = par->prev_gi.size();
      for (int step=0;step<prev_size;step++) {
        // see if the new gi is the same as the old
        int gi = par->prev_gi[step];
        if ( gi != -1 ) {
          if ( floatcmp(minx0,par->gr_minx[gi]) &&
               floatcmp(h,par->gr_h[gi]) &&
               nx0 == (int) par->gr_nx[gi]
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
              BOOST_ASSERT(i < (int) error.size());
              BOOST_ASSERT(si < (int) result->value_.size());
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
    par->gr_lbox.resize(gi+1);
    par->gr_rbox.resize(gi+1);
    par->gr_left_neighbor.resize(gi+1);
    par->gr_right_neighbor.resize(gi+1);
    par->gr_nx.resize(gi+1);
    par->gr_sibling.resize(gi+1);

    par->gr_minx[gi]  = lminx;
    par->gr_maxx[gi]  = lmaxx;
    par->gr_h[gi]     = hl/refine_factor;
    par->gr_lbox[gi]  = false;
    par->gr_rbox[gi]  = false;
    par->gr_left_neighbor[gi] = -1;
    par->gr_right_neighbor[gi] = -1;
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

// ballpark {{{
int ballpark(double const& x1, double const& x2,double const& epsilon) {
  // compare two floating point numbers
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}

// level_makeflag_simple {{{
int level_makeflag_simple(std::vector<int> &flag,std::vector<double> &error,int nxl,double ethreshold)
{
  double SMALLNUMBER = 1.e-12;
  int FLAG_REFINE = 1;
  int FLAG_NOREFINE = 0;
  for (int i=0;i<nxl;i++) {
    if ( error[i] >= ethreshold-SMALLNUMBER ) {
      flag[i] = FLAG_REFINE;
    } else {
      flag[i] = FLAG_NOREFINE;
    }
  }
  return 0;
}
// }}}
