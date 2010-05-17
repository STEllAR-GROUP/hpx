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

// local functions
int floatcmp(had_double_type x1,had_double_type x2) {
  // compare to floating point numbers
  had_double_type epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return 1;
  } else {
    return 0;
  }
}

void calcrhs(struct nodedata * rhs,
                stencil_data ** vecval,
                int flag, had_double_type dx, int size,
                bool boundary,int *bbox,int compute_index, Par const& par);

had_double_type initial_chi(had_double_type r,Par const& par) {
  had_double_type chi = par.amp*exp( -(r-par.R0)*(r-par.R0)/(par.delta*par.delta) );   
  return chi;
}

had_double_type initial_Phi(had_double_type r,Par const& par) {

  // Phi is the r derivative of chi
  had_double_type Phi = par.amp*exp( -(r-par.R0)*(r-par.R0)/(par.delta*par.delta) ) * ( -2.*(r-par.R0)/(par.delta*par.delta) );

  return Phi;
}

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    int level, had_double_type x, Par const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;
    val->cycle_ = 0;
    val->level_= level;
    val->iter_ = 0;
    val->refine_= false;
    val->right_alloc_ = 0;
    val->left_alloc_ = 0;
    val->overwrite_alloc_ = 0;

    had_double_type dx;
    had_double_type xcoord;

    dx = par.dx0/pow(2.0,level);
    if ( level == 0 ) {
      xcoord = par.minx0 + item*dx;
    } else {
      // for tapered mesh
      if (maxitems%2 == 0) {
        printf("had_amr_test.cpp : generate initial data: Problem Level %d !\n",level);
        exit(0);
      } else {
        xcoord = x + (item-((maxitems-1)/2))*dx;
      }
    }

    had_double_type chi,Phi,Pi,Energy,r;

    r = xcoord;
    chi = initial_chi(r,par);
    Phi = initial_Phi(r,par);
    Pi  = 0.0;
    Energy = 0.5*r*r*(Pi*Pi + Phi*Phi) - r*r*pow(chi,par.PP+1)/(par.PP+1);

    if ( floatcmp(r,0.0) == 0 && r < 0.0 ) {
      chi = -99999999.0;
      Phi = -99999999.0;
      Pi  = -99999999.0;
      Energy = -99999999.0;
    } else {
      chi = initial_chi(r,par);
      Phi = initial_Phi(r,par);
      Pi  = 0.0;
      Energy = 0.5*r*r*(Pi*Pi + Phi*Phi) - r*r*pow(chi,par.PP+1)/(par.PP+1);
    }

    val->x_ = r;
    val->value_.phi[0][0] = chi;
    val->value_.phi[0][1] = Phi;
    val->value_.phi[0][2] = Pi;
    val->value_.phi[0][3] = Energy;

    return 1;
}

int rkupdate(stencil_data ** vecval,stencil_data* result,int size,bool boundary,int *bbox,int compute_index, Par const& par)
{
  // copy over the level info
  result->level_ = vecval[0]->level_;

  // count the subcycle
  result->cycle_ = vecval[0]->cycle_ + 1;

  // copy over index information
  result->max_index_ = vecval[compute_index]->max_index_;
  result->index_ = vecval[compute_index]->index_;

  // allocate some temporary arrays for calculating the rhs
  nodedata rhs;
  int i;

  had_double_type dt = par.dt0/pow(2.0,(int) vecval[0]->level_);
  had_double_type dx = par.dx0/pow(2.0,(int) vecval[0]->level_);

  // Sanity check
  //for (i=0;i<size;i++) {
  //  std::cout << vecval[i]->x_ << std::endl;
  //}
  // std::cout << " iter " << vecval[0]->iter_ << std::endl;
  //  std::cout << "\n" << std::endl;
  if ( size >= 2 ) {
    if ( floatcmp(vecval[1]->x_ - vecval[0]->x_,dx) == 0 ) {
      std::cout <<" PROBLEM with dx: "<<  dx << " x[1] " << vecval[1]->x_ << " x[0] " << vecval[0]->x_ << std::endl;
      return 0;
    }
  }

  if ( par.integrator == 1 ) { // rk3

    if ( vecval[0]->iter_ == 0 ) {
      // increment subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      // Energy
      had_double_type chi = result->value_.phi[0][0];
      had_double_type Phi = result->value_.phi[0][1];
      had_double_type Pi  = result->value_.phi[0][2];
      had_double_type r   = result->x_;
      result->value_.phi[0][3] = 0.5*r*r*(Pi*Pi + Phi*Phi) - r*r*pow(chi,par.PP+1.0)/(par.PP+1.0);

      calcrhs(&rhs,vecval,0,dx,size,boundary,bbox,compute_index,par);

      for (i=0;i<num_eqns;i++) {
        result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
        result->value_.phi[1][i] = vecval[compute_index]->value_.phi[0][i] + rhs.phi[0][i]*dt;
      }
      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 1 || vecval[0]->iter_ == 3) {
      // increment subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      // apply BC's nearat r=0
      if ( boundary && bbox[0] == 1 ) {
        // chi
        result->value_.phi[1][0] = 4./3*vecval[compute_index+1]->value_.phi[1][0]
                                  -1./3*vecval[compute_index+2]->value_.phi[1][0];

        // Pi
        result->value_.phi[1][2] = 4./3*vecval[compute_index+1]->value_.phi[1][2]
                                  -1./3*vecval[compute_index+2]->value_.phi[1][2];

      } else if ( boundary && bbox[0] == 2 ) {
        // Phi
        result->value_.phi[1][1] = 0.5*vecval[compute_index+1]->value_.phi[1][1];
      } else {
        for (i=0;i<num_eqns;i++) {
          result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
          result->value_.phi[1][i] = vecval[compute_index]->value_.phi[1][i];
        }
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 2 ) {
      // increment subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,vecval,1,dx,size,boundary,bbox,compute_index,par);

      for (i=0;i<num_eqns;i++) {
        result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
        result->value_.phi[1][i] = 0.75*vecval[compute_index]->value_.phi[0][i]
                                  +0.25*vecval[compute_index]->value_.phi[1][i] + 0.25*rhs.phi[0][i]*dt;
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 4 ) {
      // increment subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,vecval,1,dx,size,boundary,bbox,compute_index,par);

      for (i=0;i<num_eqns;i++) {
        result->value_.phi[0][i] = 1./3*vecval[compute_index]->value_.phi[0][i]
                                 +2./3*(vecval[compute_index]->value_.phi[1][i] + rhs.phi[0][i]*dt);
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 5 ) {
      // reset rk subcycle counter
      result->iter_ = 0;

      // apply BC's nearat r=0
      if ( boundary && bbox[0] == 1 ) {
        // chi
        result->value_.phi[0][0] = 4./3*vecval[compute_index+1]->value_.phi[0][0]
                                  -1./3*vecval[compute_index+2]->value_.phi[0][0];

        // Pi
        result->value_.phi[0][2] = 4./3*vecval[compute_index+1]->value_.phi[0][2]
                                  -1./3*vecval[compute_index+2]->value_.phi[0][2];

      } else if ( boundary && bbox[0] == 2 ) {
        // Phi
        result->value_.phi[0][1] = 0.5*vecval[compute_index+1]->value_.phi[0][1];
      } else {
        for (i=0;i<num_eqns;i++) {
          result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
          result->value_.phi[1][i] = vecval[compute_index]->value_.phi[1][i];
        }
      }
      // Now comes the timestep update
      result->timestep_ = vecval[0]->timestep_ + 1.0/pow(2.0,(int) vecval[0]->level_);
    } else {
      printf(" PROBLEM : invalid iter flag %d\n",vecval[0]->iter_);
      return 0;
    }
  } else { 
    printf(" PROBLEM : invalid integrator %d\n",par.integrator);
    return 0;
  }

  return 1;
}

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
void calcrhs(struct nodedata * rhs,
                stencil_data ** vecval,
                int flag, had_double_type dx, int size,
                bool boundary,int *bbox,int compute_index, Par const& par)
{

  had_double_type dr = dx;
  had_double_type r = vecval[compute_index]->x_;
  had_double_type chi = vecval[compute_index]->value_.phi[flag][0];
  had_double_type Phi = vecval[compute_index]->value_.phi[flag][1];
  had_double_type Pi =  vecval[compute_index]->value_.phi[flag][2];
  had_double_type diss_chi = 0.0;
  had_double_type diss_Phi = 0.0;
  had_double_type diss_Pi = 0.0;

  if ( !boundary ) {
    // the compute_index is not physical boundary; all points in stencilsize
    // are available for computing the rhs.

    // Add  dissipation if size = 7
    if ( size == 7 ) { 
      diss_chi = -1./(64.*dr)*(  -vecval[compute_index-3]->value_.phi[flag][0]
                              +6.*vecval[compute_index-2]->value_.phi[flag][0]
                             -15.*vecval[compute_index-1]->value_.phi[flag][0]
                             +20.*vecval[compute_index  ]->value_.phi[flag][0]
                             -15.*vecval[compute_index+1]->value_.phi[flag][0]
                              +6.*vecval[compute_index+2]->value_.phi[flag][0]
                                 -vecval[compute_index+3]->value_.phi[flag][0] );
      diss_Phi = -1./(64.*dr)*(  -vecval[compute_index-3]->value_.phi[flag][1]
                              +6.*vecval[compute_index-2]->value_.phi[flag][1]
                             -15.*vecval[compute_index-1]->value_.phi[flag][1]
                             +20.*vecval[compute_index  ]->value_.phi[flag][1]
                             -15.*vecval[compute_index+1]->value_.phi[flag][1]
                              +6.*vecval[compute_index+2]->value_.phi[flag][1]
                                 -vecval[compute_index+3]->value_.phi[flag][1] );
      diss_Pi  = -1./(64.*dr)*(  -vecval[compute_index-3]->value_.phi[flag][2]
                              +6.*vecval[compute_index-2]->value_.phi[flag][2]
                             -15.*vecval[compute_index-1]->value_.phi[flag][2]
                             +20.*vecval[compute_index  ]->value_.phi[flag][2]
                             -15.*vecval[compute_index+1]->value_.phi[flag][2]
                              +6.*vecval[compute_index+2]->value_.phi[flag][2]
                                 -vecval[compute_index+3]->value_.phi[flag][2] );
    }


    had_double_type chi_np1 = vecval[compute_index+1]->value_.phi[flag][0];
    had_double_type chi_nm1 = vecval[compute_index-1]->value_.phi[flag][0];

    rhs->phi[0][0] = Pi + par.eps*diss_chi; // chi rhs

    had_double_type Pi_np1 = vecval[compute_index+1]->value_.phi[flag][2];
    had_double_type Pi_nm1 = vecval[compute_index-1]->value_.phi[flag][2];

    had_double_type Phi_np1 = vecval[compute_index+1]->value_.phi[flag][1];
    had_double_type Phi_nm1 = vecval[compute_index-1]->value_.phi[flag][1];

    rhs->phi[0][1] = (Pi_np1 - Pi_nm1)/(2.*dr) + par.eps*diss_Phi; // Phi rhs

    had_double_type r2_Phi_np1 = (r+dr)*(r+dr)*Phi_np1;
    had_double_type r2_Phi_nm1 = (r-dr)*(r-dr)*Phi_nm1;


    rhs->phi[0][2] = 3.*( r2_Phi_np1 - r2_Phi_nm1 )/( pow(r+dr,3) - pow(r-dr,3) ) + pow(chi,par.PP) + par.eps*diss_Pi; // Pi rhs

    rhs->phi[0][3] = 0.; // Energy rhs
  } else {
    // boundary -- look at the bounding box (bbox) to decide which boundary it is
    if ( bbox[0] == 1 ) {
      // we are at the left boundary  -- values are determined by quadratic fit, not evolution

      rhs->phi[0][0] = 0.0; // chi rhs -- chi is set by quadratic fit
      rhs->phi[0][1] = 0.0; // Phi rhs -- Phi-dot is always zero at r=0
      rhs->phi[0][2] = 0.0; // Pi rhs -- chi is set by quadratic fit
      rhs->phi[0][3] = 0.0; // Energy rhs -- analysis variable

    } else if (bbox[1] == 1) {

      had_double_type Phi_nm1 = vecval[compute_index-1]->value_.phi[flag][1];
      had_double_type Phi_nm2 = vecval[compute_index-2]->value_.phi[flag][1];

      had_double_type Pi_nm1 = vecval[compute_index-1]->value_.phi[flag][2];
      had_double_type Pi_nm2 = vecval[compute_index-2]->value_.phi[flag][2];

      // we are at the right boundary 
      rhs->phi[0][0] = Pi;  // chi rhs
      rhs->phi[0][1] = -(3.*Phi - 4.*Phi_nm1 + Phi_nm2)/(2.*dr) - Phi/r;    // Phi rhs
      rhs->phi[0][2] = -Pi/r - (3.*Pi - 4.*Pi_nm1 + Pi_nm2)/(2.*dr);      // Pi rhs
      rhs->phi[0][3] = 0.0; // Energy rhs -- analysis variable
    }
  }
}

had_double_type interp_quad(had_double_type y1,had_double_type y2,had_double_type y3,had_double_type y4,had_double_type y5,
                            had_double_type x, 
                            had_double_type x1,had_double_type x2,had_double_type x3,had_double_type x4,had_double_type x5) {
  had_double_type xx1 = x - x1;
  had_double_type xx2 = x - x2;
  had_double_type xx3 = x - x3;
  had_double_type xx4 = x - x4;
  had_double_type xx5 = x - x5;
  had_double_type result = xx2*xx3*xx4*xx5*y1/( (x1-x2)*(x1-x3)*(x1-x4)*(x1-x5) )
                + xx1*xx3*xx4*xx5*y2/( (x2-x1)*(x2-x3)*(x2-x4)*(x2-x5) )
                + xx1*xx2*xx4*xx5*y3/( (x3-x1)*(x3-x2)*(x3-x4)*(x3-x5) )
                + xx1*xx2*xx3*xx5*y4/( (x4-x1)*(x4-x2)*(x4-x3)*(x4-x5) )
                + xx1*xx2*xx3*xx4*y5/( (x5-x1)*(x5-x2)*(x5-x3)*(x5-x4) );

  return result;
}                 

int interpolation(had_double_type dst_x,struct nodedata *dst,
                  had_double_type * x_val, int xsize,
                  nodedata * n_val, int nsize)
{
  int i,j,start_index;

  // sanity check
  if ( x_val[0] > dst_x || x_val[xsize-1] < dst_x ) return 0;

  // specific to spherical symmetry
  if ( dst_x < 0.0 ) {
    for (i=0;i<num_eqns;i++) {
      dst->phi[0][i] = 1.e8;
      dst->phi[1][i] = 1.e8;
    }
    return 1;
  }

  // quad interp at AMR boundaries
  // find the point nearest dst_x
  for (i=0;i<xsize;i++) {
    if ( x_val[i] > dst_x ) break;
  }

  if ( i > 1 && i < xsize-2 ) {
    start_index = i-2;
  } else if ( i <= 1 ) {
    start_index = 0;
  } else if ( i >= xsize-2 ) {
    start_index = xsize-5;
  } else {
    // this shouldn't happen
    return 0;
  }
  j = i;

  // linear interpolation at boundaries
  for (i=0;i<num_eqns;i++) {
    dst->phi[0][i] = 0.5*(n_val[j-1].phi[0][i] + n_val[j].phi[0][i]);
#if 0
    dst->phi[0][i] = interp_quad(n_val[start_index].phi[0][i],
                                 n_val[start_index+1].phi[0][i],
                                 n_val[start_index+2].phi[0][i],
                                 n_val[start_index+3].phi[0][i],
                                 n_val[start_index+4].phi[0][i],
                                 dst_x,
                                 x_val[start_index],
                                 x_val[start_index+1],
                                 x_val[start_index+2],
                                 x_val[start_index+3],
                                 x_val[start_index+4]);
#endif
  }

  return 1;
}

bool refinement(stencil_data ** vecval, int size, struct nodedata *dst,int level,had_double_type r,int compute_index,bool boundary, int *bbox,Par const& par)
{
//#if 0
  had_double_type grad1,grad2,grad3,grad4;
  had_double_type dx = par.dx0/pow(2.0,(int) vecval[0]->level_);

  if ( r < par.fmr_radius ) return true;
  else return false;

  if ( compute_index > 0 && compute_index < size-1 && !boundary ) {
    // gradient detector
    grad1 = (vecval[compute_index+1]->value_.phi[0][0] - vecval[compute_index-1]->value_.phi[0][0])/(2.*dx);
    grad2 = (vecval[compute_index+1]->value_.phi[0][1] - vecval[compute_index-1]->value_.phi[0][1])/(2.*dx);
    grad3 = (vecval[compute_index+1]->value_.phi[0][2] - vecval[compute_index-1]->value_.phi[0][2])/(2.*dx);
    grad4 = (vecval[compute_index+1]->value_.phi[0][3] - vecval[compute_index-1]->value_.phi[0][3])/(2.*dx);

    if ( sqrt( grad1*grad1 + grad2*grad2 + grad3*grad3 + grad4*grad4 ) > par.ethreshold ) return true;
    else return false;
  } if ( boundary && bbox[0] == 1 ) {
    // gradient detector
    grad1 = (vecval[compute_index+1]->value_.phi[0][0] - vecval[compute_index]->value_.phi[0][0])/(dx);
    grad2 = (vecval[compute_index+1]->value_.phi[0][1] - vecval[compute_index]->value_.phi[0][1])/(dx);
    grad3 = (vecval[compute_index+1]->value_.phi[0][2] - vecval[compute_index]->value_.phi[0][2])/(dx);
    grad4 = (vecval[compute_index+1]->value_.phi[0][3] - vecval[compute_index]->value_.phi[0][3])/(dx);

    if ( sqrt( grad1*grad1 + grad2*grad2 + grad3*grad3 + grad4*grad4 ) > par.ethreshold ) return true;
    return false;
  } else {
    return false;
  }
//#endif 

#if 0
  // simple amplitude refinement
  had_double_type threshold;
  if ( level == 0 ) threshold = 0.15;
  if ( level == 1 ) threshold = 0.25;
  if ( level == 2 ) threshold = 0.21;
  if ( level == 3 ) threshold = 0.33;
  if ( level == 4 ) threshold = 0.45;

  if ( dst->phi[0][0] > threshold || 
       dst->phi[0][1] > threshold || 
       dst->phi[0][2] > threshold || 
       dst->phi[0][3] > threshold ) return true;
  else return false;
#endif
}


