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
    Par const& par)
{
    // provide initial data for the given data value 
    val->max_index_ = maxitems;
    val->index_ = item;
    val->timestep_ = 0;
    val->level_= 0;
    val->refine_= false;
    val->right_alloc_ = 0;
    /*
    if (item < (int)(maxitems / 3.) || item >= (int)(2. * maxitems / 3.))
        val->value_ = 0;
    else
        val->value_ = pow(item - 1./3., 4.) * pow(item - 2./3., 4.);
    */
    //val->value_ = random_numbers();
    double dx0 = par.dx0;

    double xcoord = par.minx0 + item*dx0;
    val->x_ = xcoord;
    val->value_ = exp(-xcoord*xcoord);

    return 1;
}

///////////////////////////////////////////////////////////////////////////
int evaluate_timestep(stencil_data const* left, stencil_data const* middle, 
    stencil_data const* right, stencil_data* result, int numsteps,Par const& par)
{
    // the middle point is our direct predecessor

    result->max_index_ = middle->max_index_;
    result->index_ = middle->index_;
    result->timestep_ = middle->timestep_ + 1;
    result->level_ = middle->level_;
    result->refine_ = true;
    /*
    result->value_ = 0.25 * left->value_ + 0.75 * right->value_;
    */
    double sum = 0;
    long n = work[middle->timestep_*nzones+middle->index_/zone];
    double dt = par.dt0;
    double dx = par.dx0;

    //printf("Point %d, iter %d: work=%ld\n", middle->index_, middle->timestep_, n);
    //for (t = 0; t < n; t++)
    //    sum += left->value_+middle->value_ ;
    result->value_ = middle->value_ - dt/dx*(middle->value_ - left->value_);

    return 1;
}

///////////////////////////////////////////////////////////////////////////
int evaluate_left_bdry_timestep(stencil_data const* middle, stencil_data const* right, 
                           stencil_data* result, int numsteps,Par const& par)
{
    // the middle point is our direct predecessor

    result->max_index_ = middle->max_index_;
    result->index_ = middle->index_;
    result->timestep_ = middle->timestep_ + 1;
    result->level_ = middle->level_;
    result->refine_ = false;
    /*
    result->value_ = 0.25 * left->value_ + 0.75 * right->value_;
    */
    double sum = 0;
    long n = work[middle->timestep_*nzones+middle->index_/zone];
    double dt = par.dt0;
    double dx = par.dx0;

    //printf("Point %d, iter %d: work=%ld\n", middle->index_, middle->timestep_, n);
    //for (t = 0; t < n; t++)
    //    sum += middle->value_+right->value_;
    //result->value_ = 0.5*(middle->value_+right->value_);
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
    result->timestep_ = middle->timestep_ + 1;
    result->level_ = middle->level_;
    result->refine_ = false;
    /*
    result->value_ = 0.25 * left->value_ + 0.75 * right->value_;
    */
    double sum = 0;
    long n = work[middle->timestep_*nzones+middle->index_/zone];
    double dt = par.dt0;
    double dx = par.dx0;

    //printf("Point %d, iter %d: work=%ld\n", middle->index_, middle->timestep_, n);
    //for (t = 0; t < n; t++)
    //    sum += left->value_+middle->value_;
    //result->value_ = 0.5*(left->value_+middle->value_);
    result->value_ = middle->value_ - dt/dx*(middle->value_ - left->value_);

    return 1;
}

int interpolation()
{
  return 1;
}

