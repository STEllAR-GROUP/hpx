//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <math.h>

//#include "../amr_c/stencil.hpp"
#include "../amr_c/stencil_data.hpp"
#include "../amr_c/stencil_functions.hpp"

#include "rand.hpp"

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    int level, double x, Par const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;
    val->level_= level;
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
      if (maxitems != 9) {
        printf("had_amr_test.cpp line 35: Problem Level %d !\n",level);
        exit(0);
      }
      xcoord = x + (item-4)*dx;
    }

    val->x_ = xcoord;
    val->value_ = exp(-xcoord*xcoord);
    //val->value_ = xcoord;

    return 1;
}

int rkupdate(stencil_data ** vecval,stencil_data* result,int size,
             int numsteps,Par const& par,int gidsize,int column)
{
  result->timestep_ = vecval[0]->timestep_ + 1.0/pow(2.0,(int) vecval[0]->level_);
  result->level_ = vecval[0]->level_;

  double dt = par.dt0;
  double dx = par.dx0;

  if ( size == 3 ) {
    result->value_ = vecval[1]->value_ - dt/dx*(vecval[1]->value_ - vecval[0]->value_);
    result->max_index_ = vecval[1]->max_index_;
    result->index_ = vecval[1]->index_;
  } else if ( size == 5 ) {
    result->value_ = vecval[2]->value_ - dt/dx*(vecval[2]->value_ - vecval[1]->value_);
    result->max_index_ = vecval[2]->max_index_;
    result->index_ = vecval[2]->index_;
  } else if ( size == 2 ) {
    if ( column == 0 ) {
      result->value_ = vecval[0]->value_;
      result->max_index_ = vecval[0]->max_index_;
      result->index_ = vecval[0]->index_;
    } else {
      result->value_ = vecval[1]->value_ - dt/dx*(vecval[1]->value_ - vecval[0]->value_);
      result->max_index_ = vecval[1]->max_index_;
      result->index_ = vecval[1]->index_;
    }
  }

  if (gidsize < 5) result->refine_ = false;

  return 1;
}

int interpolation()
{
  return 1;
}

bool refinement(double value,int level,int gidsize)
{
  if (gidsize < 5) return false;

  double threshold;
  if ( level == 0 ) threshold = 0.05;
  if ( level == 1 ) threshold = 0.15;
  if ( level == 2 ) threshold = 0.25;
  if ( level == 3 ) threshold = 0.3;
  if ( level == 4 ) threshold = 0.35;

  if ( value > threshold ) return true;
  else return false;
}


