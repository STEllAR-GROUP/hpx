//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM)
#define HPX_COMPONENTS_AMR_STENCIL_DATA_NOV_10_2008_0719PM

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    struct timestep_data
    {
        int max_index_;   // overall number of datapoints
        int index_;       // sequential number of this datapoint (0 <= index_ < max_values_)
        int timestep_;    // current time step
        double value_;    // current value
    };

}}}

#endif
