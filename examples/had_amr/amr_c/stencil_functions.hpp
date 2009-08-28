//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(AMR_C_FUNCTIONS_FEB_16_2009_0141PM)
#define AMR_C_FUNCTIONS_FEB_16_2009_0141PM

#include <hpx/config/export_definitions.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    /// The function \a generate_initial_data will be called to initialize the 
    /// given instance of 'stencil_data' 
    HPX_COMPONENT_EXPORT int generate_initial_data(
        stencil_data* data, int item, int maxitems, int row);

    /// The function \a evaluate_timestep will be called to compute the result data
    /// for the given timestep
    HPX_COMPONENT_EXPORT int evaluate_timestep(stencil_data const* left, 
        stencil_data const* middle, stencil_data const* right, 
        stencil_data* result, int numsteps);

    HPX_COMPONENT_EXPORT int evaluate_left_bdry_timestep(
        stencil_data const* middle, stencil_data const* right, 
        stencil_data* result, int numsteps);

    HPX_COMPONENT_EXPORT int evaluate_right_bdry_timestep(stencil_data const* left, 
        stencil_data const* middle,
        stencil_data* result, int numsteps);

}}}

#endif
