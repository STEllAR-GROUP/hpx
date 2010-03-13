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
    if ( r < 0.0 ) {
      chi = -9999.0;
      Phi = -9999.0;
      Pi  = -9999.0;
      Energy = -9999.0;
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
  if ( floatcmp(vecval[1]->x_ - vecval[0]->x_,dx) == 0 ) {
    std::cout <<" PROBLEM with dx: "<<  dx << std::endl;
    return 0;
  }

  if ( par.integrator == 0 ) {  // Euler
    printf(" PROBLEM: not implemented\n");
    return 0;
  } else if ( par.integrator == 1 ) { // rk3

    if ( vecval[0]->iter_ == 0 ) {
      // increment rk subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,vecval,0,dx,size,boundary,bbox,compute_index,par);

      for (i=0;i<num_eqns;i++) {
        result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
        result->value_.phi[1][i] = vecval[compute_index]->value_.phi[0][i] + rhs.phi[0][i]*dt;
      }

      if ( boundary && bbox[0] == 1 ) {
        // quadratic fit chi
        result->value_.phi[1][0] =
                              4./3*vecval[compute_index+1]->value_.phi[0][0]
                             -1./3*vecval[compute_index+2]->value_.phi[0][0];
        // quadratic fit Pi
        result->value_.phi[1][2] =
                              4./3*vecval[compute_index+1]->value_.phi[0][2]
                             -1./3*vecval[compute_index+2]->value_.phi[0][2];
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 1 ) {
      // increment rk subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,vecval,1,dx,size,boundary,bbox,compute_index,par);

      for (i=0;i<num_eqns;i++) {
        result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
        result->value_.phi[1][i] = 0.75*vecval[compute_index]->value_.phi[0][i]
                                  +0.25*vecval[compute_index]->value_.phi[1][i] + 0.25*rhs.phi[0][i]*dt;
      }

      if ( boundary && bbox[0] == 1 ) {
        // quadratic fit chi
        result->value_.phi[1][0] = 
                             4./3*vecval[compute_index+1]->value_.phi[1][0]
                             -1./3*vecval[compute_index+2]->value_.phi[1][0];
        // quadratic fit Pi
        result->value_.phi[1][2] =
                              4./3*vecval[compute_index+1]->value_.phi[1][2]
                             -1./3*vecval[compute_index+2]->value_.phi[1][2];
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 2 ) {
      calcrhs(&rhs,vecval,1,dx,size,boundary,bbox,compute_index,par);

      // reset rk subcycle counter
      result->iter_ = 0;

      for (i=0;i<num_eqns;i++) {
        result->value_.phi[0][i] = 1./3*vecval[compute_index]->value_.phi[0][i]
                                 +2./3*(vecval[compute_index]->value_.phi[1][i] + rhs.phi[0][i]*dt);
      }

      if ( boundary && bbox[0] == 1 ) {
        // quadratic fit chi
        result->value_.phi[0][0] = 
                              4./3*vecval[compute_index+1]->value_.phi[1][0]
                             -1./3*vecval[compute_index+2]->value_.phi[1][0];
        // quadratic fit Pi
        result->value_.phi[0][2] = 
                              4./3*vecval[compute_index+1]->value_.phi[1][2]
                             -1./3*vecval[compute_index+2]->value_.phi[1][2];
      }

      // Energy
      had_double_type chi = result->value_.phi[0][0];
      had_double_type Phi = result->value_.phi[0][1];
      had_double_type Pi  = result->value_.phi[0][2];
      had_double_type r   = result->x_;
      result->value_.phi[0][3] = 0.5*r*r*(Pi*Pi + Phi*Phi) - r*r*pow(chi,par.PP+1.0)/(par.PP+1.0);

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

  if ( r < 0.0 ) {
    // ignore these points 
    rhs->phi[0][0] = 0.0;
    rhs->phi[0][1] = 0.0;
    rhs->phi[0][2] = 0.0;
    rhs->phi[0][3] = 0.0;
    return;
  }

  if ( !boundary ) {
    // the compute_index is not physical boundary; all points in stencilsize
    // are available for computing the rhs.

    rhs->phi[0][0] = Pi; // chi rhs

    had_double_type Pi_np1 = vecval[compute_index+1]->value_.phi[flag][2];
    had_double_type Pi_nm1 = vecval[compute_index-1]->value_.phi[flag][2];

    rhs->phi[0][1] = (Pi_np1 - Pi_nm1)/(2.*dr); // Phi rhs

    had_double_type Phi_np1 = vecval[compute_index+1]->value_.phi[flag][1];
    had_double_type Phi_nm1 = vecval[compute_index-1]->value_.phi[flag][1];

    had_double_type r2_Phi_np1 = (r+dr)*(r+dr)*Phi_np1;
    had_double_type r2_Phi_nm1 = (r-dr)*(r-dr)*Phi_nm1;

    rhs->phi[0][2] = 3.*( r2_Phi_np1 - r2_Phi_nm1 )/( pow(r+dr,3) - pow(r-dr,3) ) + pow(chi,par.PP); // Pi rhs

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
      if ( size != 4 ) fprintf(stderr,"Problem: not enough points for boundary condition\n");

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

int interpolation(struct nodedata *dst,struct nodedata *src1,struct nodedata *src2)
{
  int i;
  // linear interpolation at boundaries
  for (i=0;i<num_eqns;i++) {
    dst->phi[0][i] = 0.5*(src1->phi[0][i] + src2->phi[0][i]);
    dst->phi[1][i] = 0.5*(src1->phi[1][i] + src2->phi[1][i]);
  }

  return 1;
}

bool refinement(stencil_data ** vecval, int size, struct nodedata *dst,int level,Par const& par)
{
//#if 0
  had_double_type grad1,grad2,grad3,grad4;
  int index;
  had_double_type dx = par.dx0/pow(2.0,(int) vecval[0]->level_);

  if ( vecval[0]->x_ < 5.0 ) {
    return true;
  }

  // gradient detector
  if ( size%2 == 1 ) {
    index = (size-1)/2;
    grad1 = (vecval[index+1]->value_.phi[0][0] - vecval[index-1]->value_.phi[0][0])/(2.*dx);
    grad2 = (vecval[index+1]->value_.phi[0][1] - vecval[index-1]->value_.phi[0][1])/(2.*dx);
    grad3 = (vecval[index+1]->value_.phi[0][2] - vecval[index-1]->value_.phi[0][2])/(2.*dx);
    grad4 = (vecval[index+1]->value_.phi[0][3] - vecval[index-1]->value_.phi[0][3])/(2.*dx);
    if ( sqrt( grad1*grad1 + grad2*grad2 + grad3*grad3 + grad4*grad4 ) > par.ethreshold ) return true;
    else return false;
  } else {
    return false;
  }
//#endif 

#if 0
  // simple amplitude refinement
  had_double_type threshold;
  if ( level == 0 ) return true;
  if ( level == 1 ) threshold = 0.005;
  if ( level == 2 ) threshold = 0.01;
  if ( level == 3 ) threshold = 0.03;
  if ( level == 4 ) threshold = 0.035;

  if ( dst->phi[0][0] > threshold || 
       dst->phi[0][1] > threshold || 
       dst->phi[0][2] > threshold || 
       dst->phi[0][3] > threshold ) return true;
  else return false;
#endif
}


