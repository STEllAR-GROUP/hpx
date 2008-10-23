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
    threads::thread_state stencil::eval(threads::thread_self& self, 
        applier::applier& appl, naming::id_type* result, 
        naming::id_type const& gid1, naming::id_type const& gid2, 
        naming::id_type const& gid3)
    {
        // count timesteps
        ++timestep_;

        // start asynchronous get operations
        lcos::future_value<memory_block_data> mf1(stubs::memory_block::get_async(appl, gid1));
        lcos::future_value<memory_block_data> mf2(stubs::memory_block::get_async(appl, gid2));
        lcos::future_value<memory_block_data> mf3(stubs::memory_block::get_async(appl, gid3));

        // get all input memory_block_data instances
        memory_data<double> const val1 (mf1.get_result(self));
        memory_data<double> const val2 (mf2.get_result(self));
        memory_data<double> const val3 (mf3.get_result(self));

        // create new dataset and store result value
        memory_block retval(memory_block::create(self, appl, appl.get_prefix()));

        // do the actual computation
        memory_data<double> r(retval.get(self));
        *r = (*val1 + *val2 + *val3) / 3;

        *result = retval.get_gid();
        return threads::terminated;
    }

    // Return, whether the current time step is the final one
    threads::thread_state stencil::is_last_timestep(threads::thread_self&, 
            applier::applier&, bool* result)
    {
        *result = (timestep_ == 2) ? true : false;
        return threads::terminated;
    }

}}}

