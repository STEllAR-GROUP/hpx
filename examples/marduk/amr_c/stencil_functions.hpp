//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(AMR_C_FUNCTIONS_FEB_16_2009_0141PM)
#define AMR_C_FUNCTIONS_FEB_16_2009_0141PM

#include <hpx/config/export_definitions.hpp>
#include "../parameter.hpp"

namespace hpx { namespace components { namespace mesh 
{

///////////////////////////////////////////////////////////////////////////////
/// The function \a generate_initial_data will be called to initialize the 
/// given instance of 'stencil_data' 
HPX_COMPONENT_EXPORT int generate_initial_data(
    stencil_data* data, int item, int maxitems, int row, detail::parameter const& par);

/// The function \a evaluate_timestep will be called to compute the result data
/// for the given timestep
HPX_COMPONENT_EXPORT int rkupdate(std::vector<nodedata*> const& val, 
    stencil_data* result,  
    bool boundary, int *bbox, int compute_index, 
    double_type const&, double_type const&, double_type const&,
    int level, detail::parameter const& par);

}}}

#endif

