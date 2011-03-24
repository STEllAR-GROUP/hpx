//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <cmath>

//#include "../amr_c/stencil.hpp"
#include "../amr_c/stencil_data.hpp"
#include "../amr_c/stencil_functions.hpp"
#include "../had_config.hpp"
#include <stdio.h>

#include <boost/scoped_array.hpp>

///////////////////////////////////////////////////////////////////////////////
// windows needs to initialize MPFR in each shared library
#if defined(BOOST_WINDOWS) 

#include "../init_mpfr.hpp"

namespace hpx { namespace components { namespace amr 
{
    // initialize mpreal default precision
    init_mpfr init_;
}}}
#endif

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
template <int flag, int dim>
inline had_double_type& phi(nodedata& nd)
{
    return nd.phi[flag][dim];
}

template <int flag, int dim>
inline had_double_type& phi(nodedata* nd)
{
    return nd->phi[flag][dim];
}

template <int flag, typename Array>
inline void 
calcrhs(struct nodedata * rhs, Array const& vecval, 
    had_double_type const& dx, bool boundary, 
    int const i, int const j, int const k, Par const& par)
{
  int const n = 3*par.granularity;

  if ( !boundary ) {
    rhs->phi[0][0] = phi<flag, 4>(vecval[i+n*(j+n*k)]); 

    rhs->phi[0][1] = - 0.5*(phi<flag, 4>(vecval[i+1+n*(j+n*k)]) - phi<flag, 4>(vecval[i-1+n*(j+n*k)]))/dx;

    rhs->phi[0][2] = - 0.5*(phi<flag, 4>(vecval[i+n*(j+1+n*k)]) - phi<flag, 4>(vecval[i+n*(j-1+n*k)]))/dx;

    rhs->phi[0][3] = - 0.5*(phi<flag, 4>(vecval[i+n*(j+n*(k+1))]) - phi<flag, 4>(vecval[i+n*(j+n*(k-1))]))/dx;

    rhs->phi[0][4] = - 0.5*(phi<flag, 1>(vecval[i+1+n*(j+n*k)]) - phi<flag, 1>(vecval[i-1+n*(j+n*k)]))/dx
                - 0.5*(phi<flag, 2>(vecval[i+n*(j+1+n*k)]) - phi<flag, 2>(vecval[i+n*(j-1+n*k)]))/dx
                - 0.5*(phi<flag, 3>(vecval[i+n*(j+n*(k+1))]) - phi<flag, 3>(vecval[i+n*(j+n*(k-1))]))/dx;
  } 
  else {
    // dirichlet
    rhs->phi[0][0] = 0.0;
    rhs->phi[0][1] = 0.0;
    rhs->phi[0][2] = 0.0;
    rhs->phi[0][3] = 0.0;
    rhs->phi[0][4] = 0.0;
  }
  return;
}

///////////////////////////////////////////////////////////////////////////////
// local functions
inline int floatcmp(had_double_type const& x1, had_double_type const& x2) 
{
  // compare to floating point numbers
  static had_double_type const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return 1;
  } else {
    return 0;
  }
}

inline had_double_type initial_chi(had_double_type const& r,Par const& par) 
{
  return par.amp*exp( -(r-par.R0)*(r-par.R0)/(par.delta*par.delta) );   
}

inline had_double_type initial_Phi(had_double_type const& r,Par const& par) 
{
  // Phi is the r derivative of chi
  static had_double_type const c_m2 = -2.;
  return par.amp*exp( -(r-par.R0)*(r-par.R0)/(par.delta*par.delta) ) * ( c_m2*(r-par.R0)/(par.delta*par.delta) );
}

inline std::size_t findlevel3D(std::size_t step, std::size_t item, 
                               std::size_t &a, std::size_t &b, std::size_t &c, Par const& par)
{
  int ll = par.level_row[step];
  // discover what level to which this point belongs
  int level = -1;
  if ( ll == par.allowedl ) {
    level = ll;
    // get 3D coordinates from 'i' value
    // i.e. i = a + nx*(b+c*nx);
    int tmp_index = item/par.nx[ll];
    c = tmp_index/par.nx[ll];
    b = tmp_index%par.nx[ll];
    a = item - par.nx[ll]*(b+c*par.nx[ll]);
    BOOST_ASSERT(item == a + par.nx[ll]*(b+c*par.nx[ll]));
  } else {
    if ( item < par.rowsize[par.allowedl] ) {
      level = par.allowedl;
    } else {
      for (int j=par.allowedl-1;j>=ll;j--) {
        if ( item < par.rowsize[j] && item >= par.rowsize[j+1] ) {
          level = j;
          break;
        }
      }
    }

    if ( level < par.allowedl ) {
      int tmp_index = (item - par.rowsize[level+1])/par.nx[level];
      c = tmp_index/par.nx[level];
      b = tmp_index%par.nx[level];
      a = (item-par.rowsize[level+1]) - par.nx[level]*(b+c*par.nx[level]);
      BOOST_ASSERT(item-par.rowsize[level+1] == a + par.nx[level]*(b+c*par.nx[level]));
    } else {
      int tmp_index = item/par.nx[level];
      c = tmp_index/par.nx[level];
      b = tmp_index%par.nx[level];
      a = item - par.nx[level]*(b+c*par.nx[level]);
    }
  }
  BOOST_ASSERT(level >= 0);
  return level;
}


///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    Par const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;

    val->x_.resize(par.granularity);
    val->y_.resize(par.granularity);
    val->z_.resize(par.granularity);
    val->value_.resize(par.granularity*par.granularity*par.granularity);

    //number of values per stencil_data
    nodedata node;

    // find out the step row are dealing with 
    int ll = par.level_row[row];

    // find out the level we are at
    std::size_t a,b,c;
    val->level_ = findlevel3D(row,item,a,b,c,par);    

    int level = val->level_;
    had_double_type dx = par.dx0/pow(2.0,level);

    static had_double_type const c_0 = 0.0;
    static had_double_type const c_7 = 7.0;
    static had_double_type const c_6 = 6.0;
    static had_double_type const c_8 = 8.0;

    for (int i=0;i<par.granularity;i++) {
      had_double_type x = par.min[level] + (a*par.granularity + i)*dx;
      had_double_type y = par.min[level] + (b*par.granularity + i)*dx;
      had_double_type z = par.min[level] + (c*par.granularity + i)*dx;

      val->x_[i] = x;
      val->y_[i] = y;
      val->z_[i] = z;
    }

    for (int k=0;k<par.granularity;k++) {
      had_double_type z = val->z_[k];
    for (int j=0;j<par.granularity;j++) {
      had_double_type y = val->y_[j];
    for (int i=0;i<par.granularity;i++) {
      had_double_type x = val->x_[i];

      had_double_type r = sqrt(x*x+y*y+z*z);

      if ( pow(r-par.R0,2) <= par.delta*par.delta && r > 0 ) {
        had_double_type Phi = par.amp*pow((r-par.R0)*(r-par.R0)
                             -par.delta*par.delta,4)/pow(par.delta,8)/r;
        had_double_type D1Phi = par.amp*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                 (c_7*r*r-c_6*r*par.R0+par.R0*par.R0+par.delta*par.delta)*
                                 x/(r*r)/pow(par.delta,8);
        had_double_type D2Phi  = par.amp*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                 (c_7*r*r-c_6*r*par.R0+par.R0*par.R0+par.delta*par.delta)*
                                 y/(r*r)/pow(par.delta,8); 
        had_double_type D3Phi = par.amp*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                 (c_7*r*r-c_6*r*par.R0+par.R0*par.R0+par.delta*par.delta)*
                                 z/(r*r)/pow(par.delta,8);
        had_double_type D4Phi = par.amp_dot*pow((r-par.R0)*(r-par.R0)-par.delta*par.delta,3)*
                                c_8*(par.R0-r)/pow(par.delta,8)/r; 

        node.phi[0][0] = Phi;
        node.phi[0][1] = D1Phi;
        node.phi[0][2] = D2Phi;
        node.phi[0][3] = D3Phi;
        node.phi[0][4] = D4Phi;
      } else {
        node.phi[0][0] = c_0;
        node.phi[0][1] = c_0;
        node.phi[0][2] = c_0;
        node.phi[0][3] = c_0;
        node.phi[0][4] = c_0;
      }

      val->value_[i+par.granularity*(j+k*par.granularity)] = node;
    }}}

    return 1;
}

int rkupdate(hpx::memory::default_vector< nodedata* >::type const& vecval, stencil_data* result, 
  bool boundary,
  int *bbox, int compute_index, 
  had_double_type const& dt, had_double_type const& dx, had_double_type const& tstep,
  int level, Par const& par)
{
    // allocate some temporary arrays for calculating the rhs
    nodedata rhs;
    boost::scoped_array<nodedata> work(new nodedata[vecval.size()]);
    boost::scoped_array<nodedata> work2(new nodedata[vecval.size()]);
    boost::scoped_array<nodedata> work3(new nodedata[vecval.size()]);

    static had_double_type const c_0_75 = 0.75;
    static had_double_type const c_0_25 = 0.25;
    static had_double_type const c_2_3 = had_double_type(2.)/had_double_type(3.);
    static had_double_type const c_1_3 = had_double_type(1.)/had_double_type(3.);

    int const n = 3*par.granularity;
    int const n2 = par.granularity;

    std::size_t i_start,i_end,j_start,j_end,k_start,k_end;
    std::size_t i_sf,i_ef,j_sf,j_ef,k_sf,k_ef;

    result->timestep_ = tstep;
    for (int timestep = par.time_granularity; timestep > 0; timestep-- ) {

      if ( !boundary ) {
        i_start = n2 - 3*timestep + 1 ;
        i_end = 2*n2 + 3*timestep - 1;
        j_start = i_start;
        j_end = i_end;
        k_start = i_start;
        k_end = i_end;

        i_sf = n2 -3*(timestep-1);
        i_ef = 2*n2 +3*(timestep-1);
        j_sf = i_sf;
        j_ef = i_ef;
        k_sf = i_sf;
        k_ef = i_ef;
      } 
      else {
        if ( bbox[0] == 1 ) {
          i_start = n2;
          i_sf = n2;
        } else {
          i_start = n2-3*timestep + 1;
          i_sf = n2-3*(timestep-1);
        }
    
        if ( bbox[1] == 1 ) {
          i_end = 2*n2;
          i_ef = 2*n2;
        } else {
          i_end = 2*n2+3*timestep - 1;
          i_ef = 2*n2+3*(timestep-1);
        }
    
        if ( bbox[2] == 1 ) {
          j_start = n2;
          j_sf = n2;
        } else {
          j_start = n2-3*timestep + 1;
          j_sf = n2-3*(timestep-1);
        }

        if ( bbox[3] == 1 ) {
          j_end = 2*n2;
          j_ef = 2*n2;
        } else {
          j_end = 2*n2+3*timestep - 1;
          j_ef = 2*n2+3*(timestep-1);
        }
    
        if ( bbox[4] == 1 ) {
          k_start = n2;
          k_sf = n2;
        } else {
          k_start = n2-3*timestep + 1;
          k_sf = n2-3*(timestep-1);
        }

        if ( bbox[5] == 1 ) {
          k_end = 2*n2;
          k_ef = 2*n2;
        } else {
          k_end = 2*n2+3*timestep - 1;
          k_ef = 2*n2+3*(timestep-1);
        }
      }

      // -------------------------------------------------------------------------
      // iter 0
      for (int k=k_start; k<k_end;k++) {
      for (int j=j_start; j<j_end;j++) {
      for (int i=i_start; i<i_end;i++) {
        if ( timestep == par.time_granularity ) {
          calcrhs<0>(&rhs,vecval,dx,boundary,i,j,k,par); 
        } else {
          calcrhs<1>(&rhs,work3,dx,boundary,i,j,k,par); 
        }

        nodedata& nd = work[i+n*(j+n*k)];
        if ( timestep == par.time_granularity ) {
          nodedata const *ndvecval = vecval[i+n*(j+n*k)];
          for (int ll=0;ll<num_eqns;ll++) {
            nd.phi[0][ll] = ndvecval->phi[0][ll];
            nd.phi[1][ll] = ndvecval->phi[0][ll] + rhs.phi[0][ll]*dt; 
          }
        } else {
          nodedata& nd3 = work3[i+n*(j+n*k)];
          for (int ll=0;ll<num_eqns;ll++) {
            nd.phi[0][ll] = nd3.phi[0][ll];
            nd.phi[1][ll] = nd3.phi[0][ll] + rhs.phi[0][ll]*dt; 
          }
        }
      }}}

      // -------------------------------------------------------------------------
      // iter 1
      for (int k=k_start; k<k_end;k++) {
      for (int j=j_start; j<j_end;j++) {
      for (int i=i_start; i<i_end;i++) {
        calcrhs<1>(&rhs,work,dx,boundary,i,j,k,par); 
        nodedata& nd = work[i+n*(j+n*k)];
        nodedata& nd2 = work2[i+n*(j+n*k)];
        for (int ll=0;ll<num_eqns;ll++) {
          nd2.phi[1][ll] = c_0_75*nd.phi[0][ll] + 
            c_0_25*nd.phi[1][ll] + c_0_25*rhs.phi[0][ll]*dt;
        }
      }}}

      // -------------------------------------------------------------------------
      // iter 2
      int ii,jj,kk;
      for (int k=k_sf; k<k_ef;k++) {
      for (int j=j_sf; j<j_ef;j++) {
      for (int i=i_sf; i<i_ef;i++) {
        calcrhs<1>(&rhs,work2,dx,boundary,i,j,k,par); 

        ii = i - n2;
        jj = j - n2;
        kk = k - n2;

        nodedata& nd = work[i+n*(j+n*k)];
        nodedata& nd2 = work2[i+n*(j+n*k)];

        if ( timestep == 1 ) {
          nodedata& ndresult = result->value_[ii + n2*(jj + n2*kk)];
          // last local rk update
          for (int ll=0;ll<num_eqns;ll++) {
            ndresult.phi[0][ll] = c_1_3*nd.phi[0][ll] + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
          }
        } else {
          nodedata& nd3 = work3[i+n*(j+n*k)];
          // more timesteps to compute
          for (int ll=0;ll<num_eqns;ll++) {
            nd3.phi[0][ll] = c_1_3*nd.phi[0][ll] + c_2_3*(nd2.phi[1][ll] + rhs.phi[0][ll]*dt);
          }
        }
      }}}

      result->timestep_ += 1.0/pow(2.0,level);

    }

    return 1;
}

