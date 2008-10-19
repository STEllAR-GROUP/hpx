//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/components/amr_test/stencil.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil

    // Compute the result value for the current time step
    double stencil::eval(double x, double y, double z)
    {
        ++timestep_;
        return (x + y + z) / 3;
    }

    // Return, whether the current time step is the final one
    bool stencil::is_last_timestep() const
    {
        return timestep_ == 2;
    }

}}}

