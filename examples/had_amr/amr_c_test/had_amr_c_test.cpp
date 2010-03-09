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

    val->x_ = xcoord;
    val->value_.phi[0][0] = par.energy*exp(-(xcoord-8.0)*(xcoord-8.0));   // u
    val->value_.phi[0][1] = 0.0;                               // psi

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
  had_double_type r1,r2,y1,y2,A,B,C;

  had_double_type dt = par.dt0/pow(2.0,(int) vecval[0]->level_);
  had_double_type dx = par.dx0/pow(2.0,(int) vecval[0]->level_);

  // Sanity check
  if ( floatcmp(vecval[1]->x_ - vecval[0]->x_,dx) == 0 ) {
    printf(" PROBLEM with dx: %g %g : x1 %g x2 %g\n",vecval[1]->x_ - vecval[0]->x_,dx,vecval[1]->x_,vecval[0]->x_);
    return 0;
  }

  if ( par.integrator == 0 ) {  // Euler
    calcrhs(&rhs,vecval,0,dx,size,boundary,bbox,compute_index,par);

    // iter is kept to be zero for Euler
    result->iter_ = 0;

    for (i=0;i<num_eqns;i++) {
      result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i] + rhs.phi[0][i]*dt;
    }

    // set left boundary by quadratic
    if ( boundary && bbox[0] == 1 ) {
      for (i=0;i<num_eqns;i++) {
        y1 = vecval[compute_index+1]->value_.phi[0][i];
        y2 = vecval[compute_index+2]->value_.phi[0][i];
        r1 = vecval[compute_index+1]->x_;
        r2 = vecval[compute_index+2]->x_;

        // y = A*r^2 + B*r + C
        B = 0; // functions are even; at r=0 dphi/dr = 0
        A = (y2-y1)/(r2*r2 - r1*r1);
        C = -A*r1*r1 + y1;
        // at r=0
        result->value_.phi[0][i] = C;
      }
    }

    result->timestep_ = vecval[0]->timestep_ + 1.0/pow(2.0,(int) vecval[0]->level_);
  } else if ( par.integrator == 1 ) { // rk3

    if ( vecval[0]->iter_ == 0 ) {
      // increment rk subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,vecval,0,dx,size,boundary,bbox,compute_index,par);

      // set left boundary by quadratic
      if ( boundary && bbox[0] == 1 ) {
        for (i=0;i<num_eqns;i++) {
          y1 = vecval[compute_index+1]->value_.phi[0][i];
          y2 = vecval[compute_index+2]->value_.phi[0][i];
          r1 = vecval[compute_index+1]->x_;
          r2 = vecval[compute_index+2]->x_;
  
          // y = A*r^2 + B*r + C
          B = 0; // functions are even; at r=0 dphi/dr = 0
          A = (y2-y1)/(r2*r2 - r1*r1);
          C = -A*r1*r1 + y1;
          // at r=0
          result->value_.phi[1][i] = C;

          // this is not used in evolution, but put it here anyways for clarity
          result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
        }
      } else {
        for (i=0;i<num_eqns;i++) {
          result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
          result->value_.phi[1][i] = vecval[compute_index]->value_.phi[0][i] + rhs.phi[0][i]*dt;
        }
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 1 ) {
      // increment rk subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,vecval,1,dx,size,boundary,bbox,compute_index,par);

      if ( boundary && bbox[0] == 1 ) {
        for (i=0;i<num_eqns;i++) {
          y1 = vecval[compute_index+1]->value_.phi[0][i];
          y2 = vecval[compute_index+2]->value_.phi[0][i];
          r1 = vecval[compute_index+1]->x_;
          r2 = vecval[compute_index+2]->x_;
  
          // y = A*r^2 + B*r + C
          B = 0; // functions are even; at r=0 dphi/dr = 0
          A = (y2-y1)/(r2*r2 - r1*r1);
          C = -A*r1*r1 + y1;
          // at r=0
          result->value_.phi[1][i] = C;

          // this is not used in evolution, but put it here anyways for clarity
          result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
        }
      } else {
        for (i=0;i<num_eqns;i++) {
          result->value_.phi[0][i] = vecval[compute_index]->value_.phi[0][i];
          result->value_.phi[1][i] = 0.75*vecval[compute_index]->value_.phi[0][i]
                                    +0.25*vecval[compute_index]->value_.phi[1][i] + 0.25*rhs.phi[0][i]*dt;
        }
      }

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 2 ) {
      calcrhs(&rhs,vecval,1,dx,size,boundary,bbox,compute_index,par);

      // reset rk subcycle counter
      result->iter_ = 0;

      if ( boundary && bbox[0] == 1 ) {
        for (i=0;i<num_eqns;i++) {
          y1 = vecval[compute_index+1]->value_.phi[0][i];
          y2 = vecval[compute_index+2]->value_.phi[0][i];
          r1 = vecval[compute_index+1]->x_;
          r2 = vecval[compute_index+2]->x_;
  
          // y = A*r^2 + B*r + C
          B = 0; // functions are even; at r=0 dphi/dr = 0
          A = (y2-y1)/(r2*r2 - r1*r1);
          C = -A*r1*r1 + y1;
          // at r=0
          result->value_.phi[0][i] = C;
        }
      } else {
        for (i=0;i<num_eqns;i++) {
          result->value_.phi[0][i] = 1./3*vecval[compute_index]->value_.phi[0][i]
                                   +2./3*(vecval[compute_index]->value_.phi[1][i] + rhs.phi[0][i]*dt);
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

  // variable: 0  -- u
  // variable: 1  -- psi

  had_double_type x = vecval[compute_index]->x_;

  if ( !boundary ) {
    // the compute_index is not physical boundary; all points in stencilsize
    // are available for computing the rhs.
    rhs->phi[0][0] = vecval[compute_index]->value_.phi[flag][1];

    rhs->phi[0][1] = (      vecval[compute_index+1]->value_.phi[flag][0] 
                       - 2.*vecval[compute_index]->value_.phi[flag][0] 
                          + vecval[compute_index-1]->value_.phi[flag][0] )/(dx*dx)
                 + 2./x * ( vecval[compute_index+1]->value_.phi[flag][0] - vecval[compute_index-1]->value_.phi[flag][0] )/dx
                      + pow(vecval[compute_index]->value_.phi[flag][0],7.0);
  } else {
    // boundary -- look at the bounding box (bbox) to decide which boundary it is
    if ( bbox[0] == 1 ) {
      // we are at the left boundary  -- values are determined by quadratic fit, not evolution
      if ( size != 4 ) fprintf(stderr,"Problem: not enough points for boundary condition\n");
      rhs->phi[0][0] = 0.0;
      rhs->phi[0][1] = 0.0;

    } else if (bbox[1] == 1) {
      // we are at the right boundary -- outflow
      rhs->phi[0][0] = -(vecval[compute_index]->value_.phi[flag][0] - vecval[compute_index-1]->value_.phi[flag][0])/dx;
      rhs->phi[0][1] = -(vecval[compute_index]->value_.phi[flag][1] - vecval[compute_index-1]->value_.phi[flag][1])/dx;
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
  had_double_type grad1,grad2;
  int index;
  had_double_type dx = par.dx0/pow(2.0,(int) vecval[0]->level_);

  // gradient detector
  if ( size%2 == 1 ) {
    index = (size-1)/2;
    grad1 = (vecval[index+1]->value_.phi[0][0] - vecval[index-1]->value_.phi[0][0])/(2.*dx);
    grad2 = (vecval[index+1]->value_.phi[0][1] - vecval[index-1]->value_.phi[0][1])/(2.*dx);
    if ( sqrt( grad1*grad1 + grad2*grad2 ) > par.ethreshold ) return true;
    else return false;
  } else {
    return false;
  }
//#endif 

#if 0
  // simple amplitude refinement
  had_double_type threshold;
  if ( level == 0 ) return true;
  if ( level == 1 ) threshold = 0.015;
  if ( level == 2 ) threshold = 0.025;
  if ( level == 3 ) threshold = 0.03;
  if ( level == 4 ) threshold = 0.035;

  if ( dst->phi[0][1] > threshold || dst->phi[0][0] > threshold ) return true;
  else return false;
#endif
}


