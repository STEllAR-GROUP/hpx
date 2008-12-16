//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>

#include "amr_test/logging.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    void logging::logentry(timestep_data const& val)
    {
        boost::mutex::scoped_lock l(mtx_);

        std::cout << val.max_index_ << ", " << val.index_ << ", " 
                  << val.timestep_ << ", "<< val.value_ << std::endl;
    }

}}}}

