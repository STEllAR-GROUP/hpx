//  Copyright (c) 2007-2010 Hartmut Kaiser
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

#define UGLIFY 1

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

void calcrhs(struct nodedata * rhs,
                std::vector< nodedata > const& vecval,
                std::vector< had_double_type > const& vecx,
                int flag, had_double_type const& dx, int size,
                bool boundary, int *bbox,int compute_index, Par const& par);

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

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    Par const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;
    val->cycle_ = 0;
    val->iter_ = 0;
    val->gw_iter_ = 0;
    val->ghostwidth_ = 0;

    val->granularity = par.granularity;
    val->x_.resize(par.granularity);
    val->value_.resize(par.granularity);

    //number of values per stencil_data
    nodedata node;

    // find out what level we are at
    std::size_t level = -1;
    for (int j=0;j<=par.allowedl;j++) {
      if (item >= par.level_begin[j] && item < par.level_end[j] ) {
        level = j; 
        break;
      }    
    }

    // identify the ghostwidth points
    for (int i=0;i<par.ghostwidth_array.size();i++) {
      if ( item == par.ghostwidth_array[i] ) {
        val->ghostwidth_ = 1; 
        level--;
        break;
      }
    }
    BOOST_ASSERT(level >= 0);

    val->level_= level;
    had_double_type dx = par.dx0/pow(2.0,(int) level);

    had_double_type r_start = 0.0;
    for (int j=par.allowedl;j>level;j--) {
      r_start += (par.alt_level_end[j]-par.alt_level_begin[j])*par.granularity*par.dx0/pow(2.0,j);
    }
    for (int j=par.alt_level_begin[level];j<item;j++) {
      r_start += dx*par.granularity;
    }

    static had_double_type const c_0 = 0.0;
    static had_double_type const c_0_5 = 0.5;

    for (int i=0;i<par.granularity;i++) {
      had_double_type r = r_start + i*dx;

      had_double_type chi = initial_chi(r,par);
      had_double_type Phi = initial_Phi(r,par);
      had_double_type Pi  = c_0;
      had_double_type Energy = c_0_5* r*r * (Pi*Pi + Phi*Phi) - r*r * pow(chi, par.PP+1)/(par.PP+1);

      val->x_[i] = r;

      node.phi[0][0] = chi;
      node.phi[0][1] = Phi;
      node.phi[0][2] = Pi;
      node.phi[0][3] = Energy;

      val->value_[i] = node;
    }

    return 1;
}

int rkupdate(std::vector< nodedata > const& vecval, stencil_data* result, 
  std::vector< had_double_type > const& vecx, int size, bool boundary,
  int *bbox, int compute_index, 
  had_double_type const& dt, had_double_type const& dx, had_double_type const& timestep,
  int iter, int level, Par const& par)
{
  // allocate some temporary arrays for calculating the rhs
  nodedata rhs,work;

  static had_double_type const c_1 = 1.;
  static had_double_type const c_2 = 2.;
  static had_double_type const c_0_75 = 0.75;
  static had_double_type const c_0_5 = 0.5;
  static had_double_type const c_0_25 = 0.25;

  static had_double_type const c_4_3 = had_double_type(4.)/had_double_type(3.);
  static had_double_type const c_2_3 = had_double_type(2.)/had_double_type(3.);
  static had_double_type const c_1_3 = had_double_type(1.)/had_double_type(3.);

  had_double_type tmp;

  if ( iter == 0 ) {
    for (int j=0;  j<result->granularity;j++) {
      calcrhs(&rhs,vecval,vecx,0,dx,size,boundary,bbox,j+compute_index,par);
      for (int i=0; i<num_eqns; i++) {
        work.phi[0][i] = vecval[j+compute_index].phi[0][i];
#ifndef UGLIFY
        work.phi[1][i] = vecval[j+compute_index].phi[0][i] + rhs.phi[0][i]*dt;
#else
        // uglify
        work.phi[1][i] = dt;
        work.phi[1][i] *= rhs.phi[0][i];
        work.phi[1][i] += vecval[j+compute_index].phi[0][i];
#endif
      }
      result->value_[j] = work;

    }
    if ( boundary && bbox[0] == 1 ) {
      // chi
#ifndef UGLIFY
      result->value_[0].phi[1][0] = c_4_3*result->value_[1].phi[1][0]
                                   -c_1_3*result->value_[2].phi[1][0];
#else
      // uglify
      result->value_[0].phi[1][0] = c_4_3*result->value_[1].phi[1][0];
      result->value_[0].phi[1][0] -= c_1_3*result->value_[2].phi[1][0];
#endif

      // Pi
#ifndef UGLIFY
      result->value_[0].phi[1][2] = c_4_3*result->value_[1].phi[1][2]
                                   -c_1_3*result->value_[2].phi[1][2];
#else
      // uglify
      result->value_[0].phi[1][2] = c_4_3*result->value_[1].phi[1][2];
      result->value_[0].phi[1][2] -= -c_1_3*result->value_[2].phi[1][2];
#endif

      // Phi
      result->value_[1].phi[1][1] = c_0_5*result->value_[2].phi[1][1];
    }
    // no timestep update-- this is just a part of an rk subcycle
    result->timestep_ = timestep;
  } 
  else if ( iter == 1 ) {
    for (int j=0; j<result->granularity; j++) {
      calcrhs(&rhs,vecval,vecx,1,dx,size,boundary,bbox,j+compute_index,par);
      for (int i=0; i<num_eqns; i++) {
        work.phi[0][i] = vecval[j+compute_index].phi[0][i];
#ifndef UGLIFY
        work.phi[1][i] = c_0_75*vecval[j+compute_index].phi[0][i]
                        +c_0_25*vecval[j+compute_index].phi[1][i] + c_0_25*rhs.phi[0][i]*dt;
#else
        // uglify
        tmp = dt;
        tmp *= c_0_25;
        tmp *= rhs.phi[0][i];
        work.phi[1][i] = vecval[j+compute_index].phi[1][i];
        work.phi[1][i] *= c_0_25;
        work.phi[1][i] += tmp;
        tmp = c_0_75;
        tmp *= vecval[j+compute_index].phi[0][i];
        work.phi[1][i] += tmp;
#endif
      }
      result->value_[j] = work;
    }

    if ( boundary && bbox[0] == 1 ) {
      // chi
#ifndef UGLIFY
      result->value_[0].phi[1][0] = c_4_3*result->value_[1].phi[1][0]
                                   -c_1_3*result->value_[2].phi[1][0];
#else
      // uglify
      result->value_[0].phi[1][0] = c_4_3*result->value_[1].phi[1][0];
      result->value_[0].phi[1][0] -= c_1_3*result->value_[2].phi[1][0];
#endif

      // Pi
#ifndef UGLIFY
      result->value_[0].phi[1][2] = c_4_3*result->value_[1].phi[1][2]
                                   -c_1_3*result->value_[2].phi[1][2];
#else
      // uglify
      result->value_[0].phi[1][2] = c_4_3*result->value_[1].phi[1][2];
      result->value_[0].phi[1][2] -= c_1_3*result->value_[2].phi[1][2];
#endif

      // Phi
      result->value_[1].phi[1][1] = c_0_5*result->value_[2].phi[1][1];
    }
    // no timestep update-- this is just a part of an rk subcycle
    result->timestep_ = timestep;
  } 
  else if ( iter == 2 ) {
    for (int j=0; j<result->granularity; j++) {
      calcrhs(&rhs,vecval,vecx,1,dx,size,boundary,bbox,j+compute_index,par);
      for (int i=0; i<num_eqns; i++) {
#ifndef UGLIFY
        work.phi[0][i] = c_1_3*vecval[j+compute_index].phi[0][i]
                        +c_2_3*(vecval[j+compute_index].phi[1][i] + rhs.phi[0][i]*dt);
#else
        // uglify
        tmp = c_1_3;
        tmp *= vecval[j+compute_index].phi[0][i];
        work.phi[0][i] = dt;
        work.phi[0][i] *= rhs.phi[0][i];
        work.phi[0][i] += vecval[j+compute_index].phi[1][i];
        work.phi[0][i] *= c_2_3;
        work.phi[0][i] += tmp;
#endif
      }
      result->value_[j] = work;
    }

    if ( boundary && bbox[0] == 1 ) {
      // chi
#ifndef UGLIFY
      result->value_[0].phi[0][0] = c_4_3*result->value_[1].phi[0][0]
                                   -c_1_3*result->value_[2].phi[0][0];
#else
      // uglify
      result->value_[0].phi[0][0] = c_4_3*result->value_[1].phi[0][0];
      result->value_[0].phi[0][0] -= c_1_3*result->value_[2].phi[0][0];
#endif
      // Pi
#ifndef UGLIFY
      result->value_[0].phi[0][2] = c_4_3*result->value_[1].phi[0][2]
                                   -c_1_3*result->value_[2].phi[0][2];
#else
      // uglify
      result->value_[0].phi[0][2] = c_4_3*result->value_[1].phi[0][2];
      result->value_[0].phi[0][2] -= c_1_3*result->value_[2].phi[0][2];
#endif
      // Phi
      result->value_[1].phi[0][1] = c_0_5*result->value_[2].phi[0][1];
    } 

    // timestep update
#ifndef UGLIFY
    result->timestep_ = timestep + 1.0/pow(2.0,level);
#else
    // uglify
    tmp = pow(c_2,level);
    result->timestep_ = c_1;
    result->timestep_ /= tmp;
    result->timestep_ += timestep;
#endif
  } 
  else {
    printf(" PROBLEM : invalid iter flag %d\n",iter);
    return 0;
  }
  return 1;
}

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
void calcrhs(struct nodedata * rhs,
                std::vector< nodedata > const& vecval,
                std::vector< had_double_type > const& vecx,
                int flag, had_double_type const& dx, int size,
                bool boundary, int *bbox,int compute_index, Par const& par)
{
  static had_double_type const c_m1 = -1.;
  static had_double_type const c_2 = 2.;
  static had_double_type const c_3 = 3.;
  static had_double_type const c_4 = 4.;
  static had_double_type const c_6 = 6.;
  static had_double_type const c_15 = 15.;
  static had_double_type const c_20 = 20.;
  static had_double_type const c_64 = 64.;
  static had_double_type const c_0 = 0.;

  had_double_type const& dr = dx;
  had_double_type const& r = vecx[compute_index];
  had_double_type const& chi = vecval[compute_index].phi[flag][0];
  had_double_type const& Phi = vecval[compute_index].phi[flag][1];
  had_double_type const& Pi =  vecval[compute_index].phi[flag][2];
  had_double_type diss_chi = c_0;
  had_double_type diss_Phi = c_0;
  had_double_type diss_Pi = c_0;
  had_double_type tmp = c_0;
  had_double_type tmp1 = c_0;
  had_double_type tmp2 = c_0;
  had_double_type tmp3 = c_0;

  // the compute_index is not physical boundary; all points in stencilsize
  // are available for computing the rhs.

  // Add  dissipation if size = 7
  if ( compute_index + 3 < size && compute_index - 3 >= 0 ) { 
#ifndef UGLIFY
    diss_chi = c_m1/(c_64*dr)*(  -vecval[compute_index-3].phi[flag][0]
                             +c_6*vecval[compute_index-2].phi[flag][0]
                            -c_15*vecval[compute_index-1].phi[flag][0]
                            +c_20*chi //vecval[compute_index  ].phi[flag][0]
                            -c_15*vecval[compute_index+1].phi[flag][0]
                             +c_6*vecval[compute_index+2].phi[flag][0]
                                 -vecval[compute_index+3].phi[flag][0] );
#else
    // uglify
    diss_chi -= vecval[compute_index+3].phi[flag][0];
    tmp = vecval[compute_index+2].phi[flag][0];
    tmp *= c_6;
    diss_chi += tmp;
    tmp = vecval[compute_index+1].phi[flag][0];
    tmp *= c_15;
    diss_chi -= tmp;
    tmp = chi;
    tmp *= c_20;
    diss_chi += tmp;
    tmp = vecval[compute_index-1].phi[flag][0];
    tmp *= c_15;
    diss_chi -= tmp;
    tmp = vecval[compute_index-2].phi[flag][0];
    tmp *= c_6;
    diss_chi += tmp;
    diss_chi -= vecval[compute_index-3].phi[flag][0];
    diss_chi *= c_m1;
    diss_chi /= c_64;
    diss_chi /= dr;
#endif
    
#ifndef UGLIFY
    diss_Phi = c_m1/(c_64*dr)*(  -vecval[compute_index-3].phi[flag][1]
                             +c_6*vecval[compute_index-2].phi[flag][1]
                            -c_15*vecval[compute_index-1].phi[flag][1]
                            +c_20*Phi //vecval[compute_index  ].phi[flag][1]
                            -c_15*vecval[compute_index+1].phi[flag][1]
                             +c_6*vecval[compute_index+2].phi[flag][1]
                                 -vecval[compute_index+3].phi[flag][1] );
#else
    // uglify
    diss_Phi -= vecval[compute_index+3].phi[flag][1];
    tmp = vecval[compute_index+2].phi[flag][1];
    tmp *= c_6;
    diss_Phi += tmp;
    tmp = vecval[compute_index+1].phi[flag][1];
    tmp *= c_15;
    diss_Phi -= tmp;
    tmp = Phi;
    tmp *= c_20;
    diss_Phi += tmp;
    tmp = vecval[compute_index-1].phi[flag][1];
    tmp *= c_15;
    diss_Phi -= tmp;
    tmp = vecval[compute_index-2].phi[flag][1];
    tmp *= c_6;
    diss_Phi += tmp;
    diss_Phi -= vecval[compute_index-3].phi[flag][1];
    diss_Phi *= c_m1;
    diss_Phi /= c_64;
    diss_Phi /= dr;
#endif

#ifndef UGLIFY
    diss_Pi  = c_m1/(c_64*dr)*(  -vecval[compute_index-3].phi[flag][2]
                             +c_6*vecval[compute_index-2].phi[flag][2]
                            -c_15*vecval[compute_index-1].phi[flag][2]
                            +c_20*Pi //vecval[compute_index  ].phi[flag][2]
                            -c_15*vecval[compute_index+1].phi[flag][2]
                             +c_6*vecval[compute_index+2].phi[flag][2]
                                 -vecval[compute_index+3].phi[flag][2] );
#else
    // uglify
    diss_Pi -= vecval[compute_index+3].phi[flag][2];
    tmp = vecval[compute_index+2].phi[flag][2];
    tmp *= c_6;
    diss_Pi += tmp;
    tmp = vecval[compute_index+1].phi[flag][2];
    tmp *= c_15;
    diss_Pi -= tmp;
    tmp = Pi;
    tmp *= c_20;
    diss_Pi += tmp;
    tmp = vecval[compute_index-1].phi[flag][2];
    tmp *= c_15;
    diss_Pi -= tmp;
    tmp = vecval[compute_index-2].phi[flag][2];
    tmp *= c_6;
    diss_Pi += tmp;
    diss_Pi -= vecval[compute_index-3].phi[flag][2];
    diss_Pi *= c_m1;
    diss_Pi /= c_64;
    diss_Pi /= dr;
#endif
  }


  if ( compute_index + 1 < size && compute_index - 1 >= 0 ) { 

    had_double_type const& chi_np1 = vecval[compute_index+1].phi[flag][0];
    had_double_type const& chi_nm1 = vecval[compute_index-1].phi[flag][0];

#ifndef UGLIFY
    rhs->phi[0][0] = Pi + par.eps*diss_chi; // chi rhs
#else
    // uglify
    rhs->phi[0][0] = diss_chi;
    rhs->phi[0][0] *= par.eps;
    rhs->phi[0][0] += Pi;
#endif

    had_double_type const& Pi_np1 = vecval[compute_index+1].phi[flag][2];
    had_double_type const& Pi_nm1 = vecval[compute_index-1].phi[flag][2];

    had_double_type const& Phi_np1 = vecval[compute_index+1].phi[flag][1];
    had_double_type const& Phi_nm1 = vecval[compute_index-1].phi[flag][1];

#ifndef UGLIFY
    rhs->phi[0][1] = (Pi_np1 - Pi_nm1)/(c_2*dr) + par.eps*diss_Phi; // Phi rhs
#else
    // uglify
    rhs->phi[0][1] = diss_Phi;
    rhs->phi[0][1] *= par.eps;
    tmp = Pi_np1;
    tmp -= Pi_nm1;
    tmp /= c_2;
    tmp /= dr;
    rhs->phi[0][1] += tmp;
#endif

#ifndef UGLIFY
    had_double_type const& r2_Phi_np1 = (r+dr)*(r+dr)*Phi_np1;
#else
    //uglify
    tmp = r;
    tmp += dr;
    tmp1 = tmp;
    tmp1 *= tmp;
    tmp1 *= Phi_np1;
    had_double_type const& r2_Phi_np1 = tmp1;
#endif

#ifndef UGLIFY
    had_double_type const& r2_Phi_nm1 = (r-dr)*(r-dr)*Phi_nm1;
#else
    // uglify
    tmp2 = r;
    tmp2 -= dr;
    tmp3 = tmp2;
    tmp3 *= tmp2;
    tmp3 *= Phi_nm1;
    had_double_type const& r2_Phi_nm1 = tmp3;
#endif

#ifndef UGLFIY
    rhs->phi[0][2] = c_3*( r2_Phi_np1 - r2_Phi_nm1 )/( pow(r+dr,3) - pow(r-dr,3) ) + pow(chi,par.PP) + par.eps*diss_Pi; // Pi rhs
#else
    // uglify
    rhs->phi[0][2] = diss_Pi;
    rhs->phi[0][2] *= par.eps;
    rhs->phi[0][2] += pow(chi,par.PP);
    tmp = r2_Phi_np1;
    tmp -= r2_Phi_nm1;
    tmp1 = r;
    tmp1 += dr;
    tmp2 = r;
    tmp2 -= dr;
    tmp3 = pow(tmp1,3);
    tmp3 -= pow(tmp2,3);
    tmp /= tmp3;
    tmp *= c_3;
    rhs->phi[0][2] += tmp;
#endif

    rhs->phi[0][3] = c_0; // Energy rhs

  } 
  else {
    // tapered point or boundary ( boundary case taken care of below )
    rhs->phi[0][0] = c_0; // chi rhs -- chi is set by quadratic fit
    rhs->phi[0][1] = c_0; // Phi rhs -- Phi-dot is always zero at r=0
    rhs->phi[0][2] = c_0; // Pi rhs -- chi is set by quadratic fit
    rhs->phi[0][3] = c_0; // Energy rhs -- analysis variable
  }

  if (boundary ) {
    // boundary -- look at the bounding box (bbox) to decide which boundary it is
    if ( bbox[0] == 1 && compute_index == 0 ) {
      // we are at the left boundary  -- values are determined by quadratic fit, not evolution

      rhs->phi[0][0] = c_0; // chi rhs -- chi is set by quadratic fit
      rhs->phi[0][1] = c_0; // Phi rhs -- Phi-dot is always zero at r=0
      rhs->phi[0][2] = c_0; // Pi rhs -- chi is set by quadratic fit
      rhs->phi[0][3] = c_0; // Energy rhs -- analysis variable
    }
    if (bbox[1] == 1 && compute_index == size-1) {

      had_double_type const& Phi_nm1 = vecval[size-2].phi[flag][1];
      had_double_type const& Phi_nm2 = vecval[size-3].phi[flag][1];

      had_double_type const& Pi_nm1 = vecval[size-2].phi[flag][2];
      had_double_type const& Pi_nm2 = vecval[size-3].phi[flag][2];

      // we are at the right boundary 
      rhs->phi[0][0] = Pi;  // chi rhs
#ifndef UGLIFY
      rhs->phi[0][1] = -(c_3*Phi - c_4*Phi_nm1 + Phi_nm2)/(c_2*dr) - Phi/r;    // Phi rhs
#else
      // uglify
      rhs->phi[0][1] = Phi;
      rhs->phi[0][1] *= c_3;
      tmp = Phi_nm1;
      tmp *= c_4;
      rhs->phi[0][1] -= tmp;
      rhs->phi[0][1] += Phi_nm2;
      rhs->phi[0][1] /= c_2;
      rhs->phi[0][1] /= dr;
      rhs->phi[0][1] *= c_m1;
      tmp = Phi;
      tmp /= r;
      rhs->phi[0][1] -= tmp;
#endif

#ifndef UGLIFY
      rhs->phi[0][2] = -Pi/r - (c_3*Pi - c_4*Pi_nm1 + Pi_nm2)/(c_2*dr);      // Pi rhs
#else
      // uglify
      rhs->phi[0][2] = Pi;
      rhs->phi[0][2] *= c_3;
      tmp = Pi_nm1;
      tmp *= c_4;
      rhs->phi[0][2] -= tmp;
      rhs->phi[0][2] += Pi_nm2;
      rhs->phi[0][2] /= c_2;
      rhs->phi[0][2] /= dr;
      rhs->phi[0][2] *= c_m1;
      tmp = Pi;
      tmp /= r;
      rhs->phi[0][2] -= tmp;
#endif

      rhs->phi[0][3] = c_0; // Energy rhs -- analysis variable
    }
  }
}
