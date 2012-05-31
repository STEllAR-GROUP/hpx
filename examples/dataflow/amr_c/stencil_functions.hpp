//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(AMR_C_FUNCTIONS_FEB_16_2009_0141PM)
#define AMR_C_FUNCTIONS_FEB_16_2009_0141PM

#include <hpx/config/export_definitions.hpp>
#include "../parameter.hpp"

namespace hpx { namespace components { namespace amr
{

///////////////////////////////////////////////////////////////////////////////
/// The function \a generate_initial_data will be called to initialize the
/// given instance of 'stencil_data'
HPX_LIBRARY_EXPORT int generate_initial_data(
    stencil_data* data, int item, int maxitems, int row, detail::parameter const& par);

/// The function \a evaluate_timestep will be called to compute the result data
/// for the given timestep
HPX_LIBRARY_EXPORT int rkupdate(std::vector<access_memory_block<stencil_data> > const&val,
    stencil_data *result,
    std::vector<int> &src, std::vector<int> &vsrc,double dt,double dx,double t,
    int nx0, int ny0, int nz0,
    double minx0, double miny0, double minz0,detail::parameter const& par);

}}}

#endif

