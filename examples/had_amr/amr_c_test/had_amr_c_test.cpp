//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <math.h>

//#include "../amr_c/stencil.hpp"
#include "../amr_c/stencil_data.hpp"
#include "../amr_c/stencil_functions.hpp"
#include "../had_config.hpp"

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
                int column, Par const& par);

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

int rkupdate(stencil_data ** vecval,stencil_data* result,int size,int column, Par const& par)
{
  // copy over the level info
  result->level_ = vecval[0]->level_;

  // count the subcycle
  result->cycle_ = vecval[0]->cycle_ + 1;

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
    result->timestep_ = vecval[0]->timestep_ + 1.0/pow(2.0,(int) vecval[0]->level_);
    calcrhs(&rhs,phi,x,dx,size,column,par);

    // iter is kept to be zero for Euler
    result->iter_ = 0;

    if ( size%2 == 1 ) {
      // the middle point
      result->max_index_ = vecval[(size-1)/2]->max_index_;
      result->index_ = vecval[(size-1)/2]->index_;
      result->value_.phi0 = phi[(size-1)/2] + rhs.phi0*dt;
    } else {
    // boundary
      if ( column == 0 ) {
        result->max_index_ = vecval[0]->max_index_;
        result->index_ = vecval[0]->index_;
        result->value_.phi0 = phi[0] + rhs.phi0*dt;
      } else {
        result->max_index_ = vecval[1]->max_index_;
        result->index_ = vecval[1]->index_;
        result->value_.phi0 = phi[1] + rhs.phi0*dt;
      }
    }
  } else if ( par.integrator == 1 ) { // rk3

    if ( vecval[0]->iter_ == 0 ) {
      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;

      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,phi,x,dx,size,column,par);

      if ( size%2 == 1 ) {
        // the middle point
        result->max_index_ = vecval[(size-1)/2]->max_index_;
        result->index_ = vecval[(size-1)/2]->index_;
        result->value_.phi0 = phi[(size-1)/2];
        result->value_.phi1 = phi[(size-1)/2] + rhs.phi0*dt;
      } else {
      // boundary
        if ( column == 0 ) {
          result->max_index_ = vecval[0]->max_index_;
          result->index_ = vecval[0]->index_;
          result->value_.phi0 = phi[0];
          result->value_.phi1 = phi[0] + rhs.phi0*dt;
        } else {
          result->max_index_ = vecval[1]->max_index_;
          result->index_ = vecval[1]->index_;
          result->value_.phi0 = phi[1];
          result->value_.phi1 = phi[1] + rhs.phi0*dt;
        }
      }

    } else if ( vecval[0]->iter_ == 1 ) {
      // no timestep update-- this is just a part of an rk subcycle
      result->timestep_ = vecval[0]->timestep_;

      result->iter_ = vecval[0]->iter_ + 1;

      calcrhs(&rhs,phi_np1,x,dx,size,column,par);

      if ( size%2 == 1 ) {
        // the middle point
        result->max_index_ = vecval[(size-1)/2]->max_index_;
        result->index_ = vecval[(size-1)/2]->index_;
        result->value_.phi0 = phi[(size-1)/2];
        result->value_.phi1 = 0.75*phi[(size-1)/2]+0.25*phi_np1[(size-1)/2] + rhs.phi0*dt;
      } else {
      // boundary
        if ( column == 0 ) {
          result->max_index_ = vecval[0]->max_index_;
          result->index_ = vecval[0]->index_;
          result->value_.phi0 = phi[0];
          result->value_.phi1 = 0.75*phi[0]+0.25*phi_np1[0] + rhs.phi0*dt;
        } else {
          result->max_index_ = vecval[1]->max_index_;
          result->index_ = vecval[1]->index_;
          result->value_.phi0 = phi[1];
          result->value_.phi1 = 0.75*phi[1]+0.25*phi_np1[1] + rhs.phi0*dt;
        }
      }

    } else if ( vecval[0]->iter_ == 2 ) {
      calcrhs(&rhs,phi_np1,x,dx,size,column,par);
      result->iter_ = 0;

      if ( size%2 == 1 ) {
        // the middle point
        result->max_index_ = vecval[(size-1)/2]->max_index_;
        result->index_ = vecval[(size-1)/2]->index_;
        result->value_.phi0 = 1./3*phi[(size-1)/2]+2./3*(phi_np1[(size-1)/2] + rhs.phi0*dt);
      } else {
      // boundary
        if ( column == 0 ) {
          result->max_index_ = vecval[0]->max_index_;
          result->index_ = vecval[0]->index_;
          result->value_.phi0 = 1./3*phi[0]+2./3*(phi_np1[0] + rhs.phi0*dt);
        } else {
          result->max_index_ = vecval[1]->max_index_;
          result->index_ = vecval[1]->index_;
          result->value_.phi0 = 1./3*phi[1]+2./3*(phi_np1[1] + rhs.phi0*dt);
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

  free(phi);
  free(x);
  free(phi_np1);

  return 1;
}

// This is a pointwise calculation: compute the rhs for point result given input values in array phi
int calcrhs(struct nodedata * rhs,
                had_double_type *phi,
                had_double_type *x, double dx, int size,
                int column, Par const& par)
{
  if ( size%2 == 1 ) {
    int midpoint = (size-1)/2;
    rhs->phi0 = -(phi[midpoint] - phi[midpoint-1])/dx;
  } else {
    // boundary
    if ( column == 0 ) {
      rhs->phi0 = 0;
    } else {
      rhs->phi0 = -(phi[1] - phi[0])/dx;
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


