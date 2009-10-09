//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>

#include "logging.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    void logging::logentry(stencil_data const& val, int row)
    {
        mutex_type::scoped_lock l(mtx_);

        std::cout << " AMR Level: " << val.level_ << "   Timestep: " <<  val.timestep_ << "   refine?: " << val.refine_ << "   row: " << row << "   index: " << val.index_ << "    Value: " << val.value_ << std::endl;   
      //  std::cout << row << ", " << val.index_ << ", " << val.level_ << ": " 
      //            << val.max_index_ << ", " << val.timestep_ << ", " 
      //            << val.refine_ << ", " 
      //            << val.value_ << std::endl;
    }

}}}}

