//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>
#include <hpx/components/amr_test/logging.hpp>

#include "stencil_data.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    threads::thread_state logging::logentry(threads::thread_self& self, 
        applier::applier& appl, naming::id_type const& memblock_gid)
    {
        // start asynchronous get operations
        components::memory_block mb(appl, memblock_gid, true);

        // get the input memory_block_data instance
        access_memory_block<timestep_data> val (mb.get(self));

        std::cout << val->max_index_ << ", " << val->index_ << ", " 
                  << val->timestep_ << ", "<< val->value_ << std::endl;

        return threads::terminated;
    }

}}}

