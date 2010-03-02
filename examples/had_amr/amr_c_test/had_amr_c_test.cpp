//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <math.h>

//#include "../amr_c/stencil.hpp"
#include "../amr_c/stencil_data.hpp"
#include "../amr_c/stencil_functions.hpp"
#include "../had_config.hpp"
#include <stdio.h>

// local functions
int floatcmp(double x1,double x2) {
  // compare to floating point numbers
  double epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return 1;
  } else {
    return 0;
  }
}

int calcrhs(struct nodedata * rhs,
                had_double_type *phi,
                had_double_type *x, double dx, int size,
                bool boundary, int *bbox,int compute_index, Par const& par);

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    int level, double x, Par const& par)
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

    double dx;
    double xcoord;

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
    val->value_.phi0 = exp(-xcoord*xcoord);
    //val->value_ = xcoord;

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
  had_double_type *phi;
  had_double_type *x;
  had_double_type *phi_np1;
  nodedata rhs;
  int i;

  phi = (double *) malloc(sizeof(had_double_type*)*size);
  phi_np1 = (double *) malloc(sizeof(had_double_type*)*size);
  x = (double *) malloc(sizeof(had_double_type*)*size);
  double dt = par.dt0/pow(2.0,(int) vecval[0]->level_);
  double dx = par.dx0/pow(2.0,(int) vecval[0]->level_);

  // assign temporary arrays
  for (i=0;i<size;i++) {
    phi[i] = vecval[i]->value_.phi0;
    phi_np1[i] = vecval[i]->value_.phi1;
    x[i] = vecval[i]->x_;
  }

  // Sanity check
  if ( floatcmp(x[1] - x[0],dx) == 0 ) {
    printf(" PROBLEM with dx: %g %g\n",x[1]-x[0],dx);
    return 0;
  }

  if ( par.integrator == 0 ) {  // Euler
    calcrhs(&rhs,phi,x,dx,size,boundary,bbox,compute_index,par);

    // iter is kept to be zero for Euler
    result->iter_ = 0;

    result->value_.phi0 = phi[compute_index] + rhs.phi0*dt;
    result->timestep_ = vecval[0]->timestep_ + 1.0/pow(2.0,(int) vecval[0]->level_);
  } else if ( par.integrator == 1 ) { // rk3

    if ( vecval[0]->iter_ == 0 ) {
      // increment rk subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,phi,x,dx,size,boundary,bbox,compute_index,par);

      result->value_.phi0 = phi[compute_index];
      result->value_.phi1 = phi[compute_index] + rhs.phi0*dt;

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 1 ) {
      // increment rk subcycle counter
      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,phi_np1,x,dx,size,boundary,bbox,compute_index,par);

      result->value_.phi0 = phi[compute_index];
      result->value_.phi1 = 0.75*phi[compute_index]+0.25*phi_np1[compute_index] + 0.25*rhs.phi0*dt;

      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;
    } else if ( vecval[0]->iter_ == 2 ) {
      calcrhs(&rhs,phi_np1,x,dx,size,boundary,bbox,compute_index,par);

      // reset rk subcycle counter
      result->iter_ = 0;

      result->value_.phi0 = 1./3*phi[compute_index]+2./3*(phi_np1[compute_index] + rhs.phi0*dt);

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

  free(phi);
  free(x);
  free(phi_np1);

  return 1;
}

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
int calcrhs(struct nodedata * rhs,
                had_double_type *phi,
                had_double_type *x, double dx, int size,
                bool boundary,int *bbox,int compute_index, Par const& par)
{
  if ( !boundary ) {
    // the compute_index is not physical boundary; all points in stencilsize
    // are available for computing the rhs.
    rhs->phi0 = -(phi[compute_index] - phi[compute_index-1])/dx;
  } else {
    // boundary -- look at the bounding box (bbox) to decide which boundary it is
    if ( bbox[0] == 1 ) {
      // we are at the left boundary
      rhs->phi0 = 0;
    } else if (bbox[1] == 1) {
      // we are at the right boundary
      rhs->phi0 = -(phi[compute_index] - phi[compute_index-1])/dx;
    }
  }
}

int interpolation(struct nodedata *dst,struct nodedata *src1,struct nodedata *src2)
{
  // linear interpolation at boundaries
  dst->phi0 = 0.5*(src1->phi0 + src2->phi0);
  dst->phi1 = 0.5*(src1->phi1 + src2->phi1);

  return 1;
}

bool refinement(struct nodedata *dst,int level)
{
  double threshold;
  if ( level == 0 ) threshold = 0.05;
  if ( level == 1 ) threshold = 0.15;
  if ( level == 2 ) threshold = 0.25;
  if ( level == 3 ) threshold = 0.3;
  if ( level == 4 ) threshold = 0.35;

  if ( dst->phi0 > threshold ) return true;
  else return false;
}


