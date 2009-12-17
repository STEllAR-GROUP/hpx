//  Copyright (c) 2007-2009 Hartmut Kaiser
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
    //val->value_ = exp(-xcoord*xcoord);
    val->value_ = xcoord;

    return 1;
}

///////////////////////////////////////////////////////////////////////////
int evaluate_timestep(stencil_data const* left, stencil_data const* middle, 
    stencil_data const* right, stencil_data* result, int numsteps, Par const& par,int gidsize)
{
    // the middle point is our direct predecessor

    result->max_index_ = middle->max_index_;
    result->index_ = middle->index_;
    result->timestep_ = middle->timestep_ + 1.0/pow(2.0,(int) middle->level_);
    result->level_ = middle->level_;
    result->refine_ = true;
    //if (gidsize < 5) result->refine_ = false;

    double sum = 0;
    double dt = par.dt0;
    double dx = par.dx0;

   // result->value_ = middle->value_ - dt/dx*(middle->value_ - left->value_);
    result->value_ = middle->value_;

    //if ( result->value_ > 0.2 ) result->refine_ = true;
    //else result->refine_ = false;

    return 1;
}

///////////////////////////////////////////////////////////////////////////
int evaluate_left_bdry_timestep(stencil_data const* middle, stencil_data const* right, 
                           stencil_data* result, int numsteps,Par const& par)
{
    // the middle point is our direct predecessor

    result->max_index_ = middle->max_index_;
    result->index_ = middle->index_;
    result->timestep_ = middle->timestep_ + 1.0/pow(2.0,(int) middle->level_);
    result->level_ = middle->level_;
    result->refine_ = false;
    /*
    result->value_ = 0.25 * left->value_ + 0.75 * right->value_;
    */
    double sum = 0;
    double dt = par.dt0;
    double dx = par.dx0;

    // boundary condition
    result->value_ = middle->value_;

    return 1;
}

///////////////////////////////////////////////////////////////////////////
int evaluate_right_bdry_timestep(stencil_data const* left, stencil_data const* middle, 
                           stencil_data* result, int numsteps,Par const& par)
{
    // the middle point is our direct predecessor

    result->max_index_ = middle->max_index_;
    result->index_ = middle->index_;
    result->timestep_ = middle->timestep_ + 1.0/pow(2.0,(int) middle->level_);
    result->level_ = middle->level_;
    result->refine_ = false;
    /*
    result->value_ = 0.25 * left->value_ + 0.75 * right->value_;
    */
    double sum = 0;
    double dt = par.dt0;
    double dx = par.dx0;

    //result->value_ = middle->value_ - dt/dx*(middle->value_ - left->value_);
    result->value_ = middle->value_;

    return 1;
}

int interpolation()
{
  return 1;
}

